# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""

# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import json
import requests
import time
import traceback

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

import evaluate
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import torch
import torch.nn as nn
import torch.distributed as dist
from copy import deepcopy
# torch.backends.cuda.matmul.allow_tf32 = False

from anybcq.utils.swap_linear import set_precision_model
from anybcq.quantization.anybcq import AnyBCQ
from anybcq.utils.analyzer import get_analyzer
from anybcq.quantization.cached_loader import DataCacheWrapper, StopForwardException

from arguments import ModelArguments, DataTrainingArguments


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.51.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)


class DeviceAwareTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        device = self.args.device
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }


def _move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    return value


def _tensor_payload(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, tuple):
        for item in obj:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Expected tensor-like payload, got {type(obj)}")


def _send_metadata(meta: dict, dst: int):
    payload = json.dumps(meta).encode("utf-8")
    size = torch.tensor([len(payload)], dtype=torch.int64, device="cuda")
    dist.send(size, dst=dst)
    byte_tensor = torch.tensor(list(payload), dtype=torch.uint8, device="cuda")
    dist.send(byte_tensor, dst=dst)


def _recv_metadata(src: int):
    size = torch.empty((1,), dtype=torch.int64, device="cuda")
    dist.recv(size, src=src)
    nbytes = int(size.item())
    byte_tensor = torch.empty((nbytes,), dtype=torch.uint8, device="cuda")
    dist.recv(byte_tensor, src=src)
    payload = bytes(byte_tensor.cpu().tolist())
    return json.loads(payload.decode("utf-8"))


_DTYPE_TO_CODE = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
    torch.int64: 3,
    torch.int32: 4,
    torch.bool: 5,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}


def _send_tensor(tensor: torch.Tensor, dst: int):
    dtype_code = _DTYPE_TO_CODE.get(tensor.dtype, None)
    if dtype_code is None:
        raise ValueError(f"Unsupported tensor dtype for send: {tensor.dtype}")
    rank_dtype = torch.tensor(
        [tensor.dim(), dtype_code], dtype=torch.int64, device=tensor.device
    )
    shape = torch.tensor(tensor.shape, dtype=torch.int64, device=tensor.device)
    dist.send(rank_dtype, dst=dst)
    dist.send(shape, dst=dst)
    dist.send(tensor.contiguous(), dst=dst)


def _recv_tensor(src: int, device: torch.device):
    rank_dtype = torch.empty((2,), dtype=torch.int64, device=device)
    dist.recv(rank_dtype, src=src)
    ndim = int(rank_dtype[0].item())
    dtype_code = int(rank_dtype[1].item())
    dtype = _CODE_TO_DTYPE.get(dtype_code, None)
    if dtype is None:
        raise ValueError(f"Unsupported received dtype code: {dtype_code}")
    shape = torch.empty((ndim,), dtype=torch.int64, device=device)
    dist.recv(shape, src=src)
    recv_shape = tuple(int(v) for v in shape.tolist())
    tensor = torch.empty(recv_shape, dtype=dtype, device=device)
    dist.recv(tensor, src=src)
    return tensor


def _strip_batch_dim(tensor: torch.Tensor):
    if tensor.dim() >= 1 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def _capture_first_layer_context(model, all_layers, micro_inputs, device):
    wrapped = DataCacheWrapper(all_layers[0])
    all_layers[0] = wrapped
    try:
        model(**micro_inputs)
    except StopForwardException:
        pass
    finally:
        all_layers[0] = wrapped.module

    hidden_in = wrapped.inp_data.detach()
    other = {}
    other_data = wrapped.other_data or {}
    for key in ("attention_mask", "position_ids", "cache_position"):
        value = other_data.get(key, None)
        if isinstance(value, torch.Tensor):
            other[key] = value.detach()
    del wrapped
    torch.cuda.empty_cache()
    return hidden_in, other


def _forward_layer_stack(base_model, layers, hidden_states, other):
    out = hidden_states
    attention_mask = other.get("attention_mask", None)
    position_ids = other.get("position_ids", None)
    cache_position = other.get("cache_position", None)

    model_backbone = getattr(base_model, "model", None)
    for layer in layers:
        kwargs = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        if cache_position is not None:
            kwargs["cache_position"] = cache_position

        rope = None
        if model_backbone is not None and hasattr(model_backbone, "rotary_emb"):
            rope = model_backbone.rotary_emb
        elif hasattr(layer, "self_attn") and hasattr(layer.self_attn, "rotary_emb"):
            rope = layer.self_attn.rotary_emb

        if rope is not None and position_ids is not None:
            cos, sin = rope(out, position_ids)
            kwargs["position_embeddings"] = (cos, sin)

        if kwargs:
            layer_out = layer(hidden_states=out, **kwargs)
        else:
            layer_out = layer(hidden_states=out)
        out = _tensor_payload(layer_out)

    return out


class StageCalibrationDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class StagePTQModel(nn.Module):
    def __init__(self, base_model, layer_start, layer_end):
        super().__init__()
        backbone = getattr(base_model, "model", base_model)
        object.__setattr__(self, "_rope_owner", backbone)
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.layers = nn.ModuleList(
            [backbone.layers[i] for i in range(layer_start, layer_end)]
        )

    def forward(
        self,
        input_ids=None,
        hidden_states=None,
        attention_mask=None,
        position_ids=None,
        cache_position=None,
        **kwargs,
    ):
        if hidden_states is None:
            raise ValueError(
                "StagePTQModel requires hidden_states in pipeline mode."
            )

        other = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "cache_position": cache_position,
        }
        return _forward_layer_stack(self._rope_owner, self.layers, hidden_states, other)


def _send_stage_packet(dst, hidden_states, other):
    meta = {
        "done": False,
        "has_attention_mask": other.get("attention_mask") is not None,
        "has_position_ids": other.get("position_ids") is not None,
        "has_cache_position": other.get("cache_position") is not None,
    }
    _send_metadata(meta, dst=dst)
    _send_tensor(hidden_states, dst=dst)
    for key in ("attention_mask", "position_ids", "cache_position"):
        tensor = other.get(key, None)
        if tensor is not None:
            _send_tensor(tensor, dst=dst)


def _recv_stage_packet(src, device):
    meta = _recv_metadata(src=src)
    if meta.get("done", False):
        return True, None, None
    hidden_states = _recv_tensor(src=src, device=device)
    other = {}
    if meta.get("has_attention_mask", False):
        other["attention_mask"] = _recv_tensor(src=src, device=device)
    if meta.get("has_position_ids", False):
        other["position_ids"] = _recv_tensor(src=src, device=device)
    if meta.get("has_cache_position", False):
        other["cache_position"] = _recv_tensor(src=src, device=device)
    return False, hidden_states, other


def _send_stage_done(dst):
    _send_metadata({"done": True}, dst=dst)


def build_pipeline_stage_dataloader(
    model,
    all_layers,
    stage_layers,
    source_dataloader,
    stage_rank,
    world_size,
    microbatch_size,
    per_device_train_batch_size,
):
    device = torch.device("cuda", torch.cuda.current_device())
    is_first = stage_rank == 0
    is_last = stage_rank == (world_size - 1)
    left = stage_rank - 1
    right = stage_rank + 1
    local_items = []

    if is_first:
        with torch.no_grad():
            for batch in source_dataloader:
                batch_size = int(batch["input_ids"].shape[0])
                for i in range(0, batch_size, microbatch_size):
                    j = min(batch_size, i + microbatch_size)
                    for sample_idx in range(i, j):
                        sample = {}
                        for key, value in batch.items():
                            if isinstance(value, torch.Tensor):
                                sample[key] = value[sample_idx : sample_idx + 1]
                        sample.pop("labels", None)
                        micro_gpu = {
                            k: _move_to_device(v, device) for k, v in sample.items()
                        }
                        hidden_in, other = _capture_first_layer_context(
                            model=model,
                            all_layers=all_layers,
                            micro_inputs=micro_gpu,
                            device=device,
                        )
                        stage_hidden = _forward_layer_stack(
                            base_model=model,
                            layers=stage_layers,
                            hidden_states=hidden_in,
                            other=other,
                        )
                        if not is_last:
                            _send_stage_packet(
                                dst=right,
                                hidden_states=stage_hidden,
                                other=other,
                            )

                        local_item = {
                            "hidden_states": _strip_batch_dim(hidden_in).detach().cpu()
                        }
                        for key in ("attention_mask", "position_ids", "cache_position"):
                            value = other.get(key, None)
                            if isinstance(value, torch.Tensor):
                                local_item[key] = _strip_batch_dim(value).detach().cpu()
                        local_items.append(local_item)

                        del stage_hidden, hidden_in, other, micro_gpu, sample
                        torch.cuda.empty_cache()

        if not is_last:
            _send_stage_done(dst=right)
    else:
        with torch.no_grad():
            while True:
                is_done, hidden_in, other = _recv_stage_packet(src=left, device=device)
                if is_done:
                    if not is_last:
                        _send_stage_done(dst=right)
                    break

                stage_hidden = _forward_layer_stack(
                    base_model=model,
                    layers=stage_layers,
                    hidden_states=hidden_in,
                    other=other,
                )
                if not is_last:
                    _send_stage_packet(
                        dst=right,
                        hidden_states=stage_hidden,
                        other=other,
                    )

                local_item = {"hidden_states": _strip_batch_dim(hidden_in).detach().cpu()}
                for key in ("attention_mask", "position_ids", "cache_position"):
                    value = other.get(key, None)
                    if isinstance(value, torch.Tensor):
                        local_item[key] = _strip_batch_dim(value).detach().cpu()
                local_items.append(local_item)

                del stage_hidden, hidden_in, other
                torch.cuda.empty_cache()

    stage_dataset = StageCalibrationDataset(local_items)
    stage_loader = DataLoader(
        stage_dataset,
        shuffle=False,
        collate_fn=default_data_collator,
        batch_size=per_device_train_batch_size,
    )
    return stage_loader, len(local_items)


def resolve_chunk_layer_range(model_args, num_layers):
    layer_start = model_args.layer_start
    layer_end = num_layers if model_args.layer_end == -1 else model_args.layer_end
    chunk_index = model_args.chunk_index

    if model_args.num_chunks > 1:
        if chunk_index == -1:
            raise ValueError("--chunk_index must be set when --num_chunks > 1")
        chunk_size = math.ceil(num_layers / model_args.num_chunks)
        layer_start = chunk_index * chunk_size
        layer_end = min(num_layers, layer_start + chunk_size)

    if layer_start < 0 or layer_start >= num_layers:
        raise ValueError(
            f"Invalid layer_start={layer_start} for model with {num_layers} layers."
        )
    if layer_end <= layer_start or layer_end > num_layers:
        raise ValueError(
            f"Invalid layer_end={layer_end}; expected in [{layer_start + 1}, {num_layers}]."
        )

    return layer_start, layer_end, chunk_index, model_args.num_chunks


def resolve_stage_layer_range(stage_rank, num_stages, num_layers):
    chunk_size = math.ceil(num_layers / num_stages)
    start = stage_rank * chunk_size
    end = min(num_layers, start + chunk_size)
    if start >= num_layers or end <= start:
        raise ValueError(
            f"Stage rank {stage_rank} has empty layer range for num_layers={num_layers}, num_stages={num_stages}."
        )
    return start, end


def _env_int(*keys, default):
    for key in keys:
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"Environment variable {key} must be an integer.") from exc
    return default


def resolve_dist_state(model_args):
    if not model_args.dist_ptq:
        return None

    stage_rank = model_args.stage_rank
    if stage_rank == -1:
        stage_rank = _env_int("RANK", "SLURM_PROCID", default=0)

    world_size = _env_int("WORLD_SIZE", "SLURM_NTASKS", default=model_args.num_stages)
    if model_args.num_stages > 1 and world_size != model_args.num_stages:
        raise ValueError(
            f"--num_stages ({model_args.num_stages}) does not match WORLD_SIZE/SLURM_NTASKS ({world_size})."
        )
    if stage_rank < 0 or stage_rank >= world_size:
        raise ValueError(
            f"Invalid stage_rank={stage_rank} for world_size={world_size}."
        )

    local_rank = _env_int("LOCAL_RANK", "SLURM_LOCALID", default=0)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=model_args.dist_backend,
            init_method=model_args.dist_init_method,
            rank=stage_rank,
            world_size=world_size,
        )

    return {
        "stage_rank": stage_rank,
        "world_size": world_size,
        "local_rank": local_rank,
    }


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    dist_state = resolve_dist_state(model_args)
    # local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # torch.cuda.set_device(local_rank)
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    from data_utils import get_dataset

    raw_datasets = get_dataset(data_args, model_args)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "attn_implementation": "eager",  # for bug from transformers version up
        "use_cache": False,  # for bug from attention mask shape(https://github.com/huggingface/transformers/blob/0466fd5ca25fc6cc3d44ef4b690f2e701cf6f28a/src/transformers/models/llama/modeling_llama.py#L718)
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(
            dict((p.data_ptr(), p.numel()) for p in model.parameters()).values()
        )
        logger.info(
            f"Training new model from scratch - Total size={n_params / 2**20:.2f}M params"
        )

    model.resize_token_embeddings(len(tokenizer))

    print("********************")
    print("*** PLM is used! ***")
    print("********************")

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base"
    )

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(desc="grouping texts together"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    if dist_state is not None:
        logger.info(
            "Distributed PTQ context initialized: rank=%s world_size=%s local_rank=%s",
            dist_state["stage_rank"],
            dist_state["world_size"],
            dist_state["local_rank"],
        )

    # Quantization: FlexRound for GPT Causal LM tasks
    quantization_chunk_metadata = None
    if model_args.quantization:
        train_dataset = lm_datasets["train"]
        if not training_args.do_eval:
            eval_dataset = lm_datasets["validation"]

        if "train" in model_args.quantization_dataset:
            quantization_dataset = lm_datasets["train"]
            print("########## reconstruction with train dataset ########### ")
        else:  # 'eval' in  model_args.quantization_dataset:
            quantization_dataset = lm_datasets["validation"]
            print("########## reconstruction with validation dataset ########### ")

        if model_args.num_samples is not None:
            origin_len = len(quantization_dataset)
            max_eval_samples = min(origin_len, model_args.num_samples)
            quantization_dataset = quantization_dataset.select(range(max_eval_samples))
            print("###### Using sample of total dataset ####### ")
            print("{} of {}".format(len(quantization_dataset), origin_len))
        else:
            print("######## Using all total dataset ########")

        quantization_source_dataloader = DataLoader(
            quantization_dataset,
            shuffle=False,
            collate_fn=default_data_collator,
            batch_size=training_args.per_device_train_batch_size,
        )

        # build quantization parameters
        wq_params = {
            "n_bits": model_args.n_bits_w,
            "group_size": model_args.group_size,
        }
        aq_params = {
            "n_bits": 16,
            "input_prob": model_args.input_prob,
        }
        recon_dtype = getattr(torch, model_args.recon_dtype)
        analyzer = get_analyzer(
            model, yaml_path=None, include_tokenizer=False, cpu_only=False
        )
        arch_config = analyzer.get_arch_config()
        num_layers = len(analyzer.get_layers())
        if model_args.dist_ptq:
            stage_rank = dist_state["stage_rank"]
            world_size = dist_state["world_size"]
            layer_start, layer_end = resolve_stage_layer_range(
                stage_rank=stage_rank,
                num_stages=world_size,
                num_layers=num_layers,
            )
            chunk_index = stage_rank
            num_chunks = world_size
        else:
            layer_start, layer_end, chunk_index, num_chunks = resolve_chunk_layer_range(
                model_args, num_layers
            )
        logger.info(
            "Quantization layer chunk: [%s, %s) over %s total layers",
            layer_start,
            layer_end,
            num_layers,
        )
        if model_args.dist_ptq:
            logger.info(
                "Distributed PTQ stage rank=%s/%s local_rank=%s",
                dist_state["stage_rank"],
                dist_state["world_size"],
                dist_state["local_rank"],
            )

        model_for_ptq = model
        stage_item_count = None
        if model_args.dist_ptq:
            stage_layers = [model.model.layers[i] for i in range(layer_start, layer_end)]
            stage_loader, stage_item_count = build_pipeline_stage_dataloader(
                model=model,
                all_layers=model.model.layers,
                stage_layers=stage_layers,
                source_dataloader=quantization_source_dataloader,
                stage_rank=stage_rank,
                world_size=world_size,
                microbatch_size=model_args.microbatch_size,
                per_device_train_batch_size=training_args.per_device_train_batch_size,
            )
            logger.info(
                "Stage %s built pipeline local dataset with %s items",
                stage_rank,
                stage_item_count,
            )
            quantization_source_dataloader = stage_loader
            stage_model = StagePTQModel(
                base_model=model,
                layer_start=layer_start,
                layer_end=layer_end,
            )
            stage_model.ptq_layers = stage_model.layers
            stage_model.config = model.config
            model_for_ptq = stage_model

        anybcq = AnyBCQ(
            model=model_for_ptq,
            data_loader=quantization_source_dataloader,
            w_lr1=model_args.w_lr1,
            w_lr2=model_args.w_lr2,
            w_lr3=model_args.w_lr3,
            iters=model_args.iters_w,
            add_bits=model_args.add_bits,
            num_samples=model_args.num_samples,
            wq_params=wq_params,
            aq_params=aq_params,
            batch_size=training_args.per_device_train_batch_size,
            torch_dtype=torch_dtype,
            recon_dtype=recon_dtype,
            input_prob=model_args.input_prob,
            asymmetric=model_args.asymmetric,
            train_beta=model_args.train_beta,
        )
        if model_args.dist_ptq:
            local_layer_start = 0
            local_layer_end = len(stage_layers)
        else:
            local_layer_start = layer_start
            local_layer_end = layer_end

        quantization_chunk_metadata = anybcq.minimize(
            layer_start=local_layer_start,
            layer_end=local_layer_end,
        )
        if model_args.dist_ptq:
            quantization_chunk_metadata["local_layer_start"] = local_layer_start
            quantization_chunk_metadata["local_layer_end"] = local_layer_end
            quantization_chunk_metadata["layer_start"] = layer_start
            quantization_chunk_metadata["layer_end"] = layer_end
        quantization_chunk_metadata["chunk_index"] = chunk_index
        quantization_chunk_metadata["num_chunks"] = num_chunks
        quantization_chunk_metadata["dist_ptq"] = bool(model_args.dist_ptq)
        quantization_chunk_metadata["stage_rank"] = (
            dist_state["stage_rank"] if model_args.dist_ptq else chunk_index
        )
        quantization_chunk_metadata["world_size"] = (
            dist_state["world_size"] if model_args.dist_ptq else num_chunks
        )
        quantization_chunk_metadata["num_layers"] = num_layers
        quantization_chunk_metadata["microbatch_size"] = model_args.microbatch_size
        quantization_chunk_metadata["pipeline_mode"] = (
            "activation_stream_v1" if model_args.dist_ptq else "stage_partition_v1"
        )
        if stage_item_count is not None:
            quantization_chunk_metadata["local_stage_items"] = stage_item_count

        if model_args.lm_head_quant:
            print(f">> Quantize LM_HEAD")
            from alice.functions.bcq.bcq import quantize

            lm_head_wq_params = {
                "qbits": 4,
                "rounds": 100,
                "group_size": 128,
                "transpose": False,
            }
            w_hat, binary, alpha, _ = quantize(
                model.lm_head.weight, **lm_head_wq_params
            )
            model.lm_head.weight.data.copy_(w_hat)

        # save model
        if training_args.output_dir is not None and model_args.save_model:
            if model_args.dist_ptq:
                model_to_save = model
            else:
                model_to_save = model
            model_to_save.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            # add new config parameters
            anybcq_configs = {
                "seed_precision": model_args.n_bits_w,
                "parent_precision": model_args.n_bits_w + model_args.add_bits,
                "group_size": model_args.group_size,
                "arch_config": arch_config,
            }
            if quantization_chunk_metadata is not None:
                anybcq_configs["chunk"] = quantization_chunk_metadata
            config.anybcq = anybcq_configs
            config.save_pretrained(training_args.output_dir)

    # Evaluation
    if model_args.skip_eval:
        logger.info("Skipping evaluation because --skip_eval is enabled.")
        if dist_state is not None and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    # Potential fix. Line below fixes "Expected all tensors to be on the same device."
    # Huggingface automatically wraps qnn model into DataParallelModel if _n_gpu > 1  during the evalaution step.
    # Reinitialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval
        else None,
    )

    logger.info("*** Quantized Model Evaluate ***")
    for precision in range(
        model_args.n_bits_w, model_args.n_bits_w + model_args.add_bits + 1
    ):
        set_precision_model(model, precision)
        print(f"<<< Setting Precision to {precision} for Evaluation >>>")

        # evaluation for validation dataset of calibration dataset
        print(f"<<< Dataset={data_args.dataset_name} Evaluation >>>")

        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval_qnn", metrics)
        trainer.save_metrics("eval_qnn", metrics)

        metrics_qnn_eval = dict(
            ("qnn_eval_" + key, value) for (key, value) in metrics.items()
        )

        metrics_data = json.dumps(metrics_qnn_eval)

        if "wiki" not in data_args.dataset_name:
            # evaluation for validation dataset of wikitext-2
            print(f"\n<<< Dataset=WikiText-2 Evaluation >>>")
            test_datasets = load_dataset(
                "wikitext",
                "wikitext-2-raw-v1",
                cache_dir=model_args.cache_dir,
                # use_auth_token=True if model_args.use_auth_token else None,
            )
            test_column_names = test_datasets["validation"].column_names
            # text_column_name = "text" if "text" in column_names else column_names[0]

            with training_args.main_process_first(desc="dataset map tokenization"):
                tokenized_test_datasets = test_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=test_column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )

            with training_args.main_process_first(desc="grouping texts together"):
                lm_test_datasets = tokenized_test_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )

            if "validation" not in tokenized_test_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_test_dataset = lm_test_datasets["validation"]

            logger.info("*** Quantized Model Evaluate on Wikitext ***")

            metrics_wikitext = trainer.evaluate(eval_dataset=eval_test_dataset)

            metrics_wikitext["eval_samples_wikitext"] = len(eval_test_dataset)
            try:
                perplexity_wikitext = math.exp(metrics_wikitext["eval_loss"])
            except OverflowError:
                perplexity_wikitext = float("inf")
            metrics_wikitext["perplexity_wikitext"] = perplexity_wikitext

            trainer.log_metrics("eval_qnn_wikitext", metrics_wikitext)
            trainer.save_metrics("eval_qnn_wikitext", metrics_wikitext)

            metrics_wikitext_qnn_eval = dict(
                ("qnn_eval_wikitext_" + key, value)
                for (key, value) in metrics_wikitext.items()
            )

            metrics_wikitext_data = json.dumps(metrics_wikitext_qnn_eval)

    if dist_state is not None and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
