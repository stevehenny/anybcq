from .helpers import dataloader
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional
from .helpers.utils import (
    vprint,
    logprint,
    get_tokenizer_type,
    name_splitter,
    base_model_name_to_hf_repo_name,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from anybcq.inference.AnyBCQForCausalLM import AnyBCQForCausalLM
from anybcq.inference.model_shard import ModelShard
from anybcq.quantization.cached_loader import DataCacheWrapper, StopForwardException
from anybcq.utils.analyzer import get_analyzer
import os
import json
import lm_eval
import tokenizers
import transformers

current_dir = os.path.dirname(os.path.realpath(__file__))


@dataclass
class DistInferenceState:
    stage_rank: int
    world_size: int
    local_rank: int
    process_group: object
    device: torch.device


@dataclass
class DistStageContext:
    model: nn.Module
    hf_model: nn.Module
    model_backbone: nn.Module
    all_layers: nn.ModuleList
    stage_layers: list[nn.Module]
    stage_rank: int
    world_size: int
    device: torch.device
    is_first: bool
    is_last: bool
    shard_comm: Optional[ModelShard]
    embed_module: Optional[nn.Module]
    norm_module: Optional[nn.Module]
    final_layer_norm_module: Optional[nn.Module]
    project_out_module: Optional[nn.Module]
    lm_head_module: Optional[nn.Module]


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


def _resolve_attr_path(root, path: str):
    module = root
    for name in path.split("."):
        module = getattr(module, name)
    return module


def _layer_range_for_stage(stage_rank: int, num_stages: int, num_layers: int):
    chunk_size = math.ceil(num_layers / num_stages)
    start = stage_rank * chunk_size
    end = min(num_layers, start + chunk_size)
    if start >= num_layers or end <= start:
        raise ValueError(
            f"Stage rank {stage_rank} has empty layer range for num_layers={num_layers}, num_stages={num_stages}."
        )
    return start, end


def _tensor_payload(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, tuple):
        for item in obj:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Expected tensor-like payload, got {type(obj)}")


def _capture_first_layer_context(model, all_layers, micro_inputs):
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
    return hidden_in, other


def _forward_layer_stack(base_model, layers, hidden_states, other):
    out = hidden_states
    attention_mask = other.get("attention_mask", None)
    position_ids = other.get("position_ids", None)
    cache_position = other.get("cache_position", None)

    model_backbone = getattr(base_model, "model", base_model)
    for layer in layers:
        kwargs = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        position_ids_for_layer = position_ids
        if position_ids_for_layer is None and cache_position is not None:
            if cache_position.dim() == 1:
                position_ids_for_layer = cache_position.unsqueeze(0)
            else:
                position_ids_for_layer = cache_position
        if position_ids_for_layer is None:
            position_ids_for_layer = torch.arange(
                out.shape[1], device=out.device, dtype=torch.long
            ).unsqueeze(0)
        else:
            position_ids_for_layer = position_ids_for_layer.to(
                device=out.device, dtype=torch.long
            )
            if position_ids_for_layer.dim() == 1:
                position_ids_for_layer = position_ids_for_layer.unsqueeze(0)
        if position_ids_for_layer.shape[0] == 1 and out.shape[0] > 1:
            position_ids_for_layer = position_ids_for_layer.expand(out.shape[0], -1)

        kwargs["position_ids"] = position_ids_for_layer
        if cache_position is not None:
            kwargs["cache_position"] = cache_position.to(device=out.device)

        raw_layer = getattr(layer, "module", layer)
        rope = None
        if hasattr(model_backbone, "rotary_emb"):
            rope = model_backbone.rotary_emb
        elif hasattr(raw_layer, "self_attn") and hasattr(raw_layer.self_attn, "rotary_emb"):
            rope = raw_layer.self_attn.rotary_emb

        if rope is not None:
            cos, sin = rope(out, position_ids_for_layer)
            kwargs["position_embeddings"] = (cos, sin)

        layer_out = layer(out, **kwargs)
        out = _tensor_payload(layer_out)
    return out


def _prepare_dist_stage_context(model, dist_state: DistInferenceState):
    hf_model = model.model if isinstance(model, AnyBCQForCausalLM) else model
    analyzer = get_analyzer(hf_model, include_tokenizer=False)
    arch = analyzer.get_arch_config()
    model_backbone = _resolve_attr_path(hf_model, arch["model_name"])
    all_layers = _resolve_attr_path(model_backbone, arch["layers_name"])

    layer_start, layer_end = _layer_range_for_stage(
        dist_state.stage_rank, dist_state.world_size, len(all_layers)
    )
    stage_layers = [all_layers[i] for i in range(layer_start, layer_end)]
    for layer in stage_layers:
        layer.to(dist_state.device)

    is_first = dist_state.stage_rank == 0
    is_last = dist_state.stage_rank == (dist_state.world_size - 1)

    embed_module = None
    for embed_name in ("embed_tokens", "wte", "tok_embeddings"):
        if hasattr(model_backbone, embed_name):
            embed_module = getattr(model_backbone, embed_name)
            break
    if is_first and embed_module is not None:
        embed_module.to(dist_state.device)
    if is_first and hasattr(model_backbone, "rotary_emb"):
        model_backbone.rotary_emb.to(dist_state.device)

    norm_module = getattr(model_backbone, "norm", None)
    final_layer_norm_module = getattr(model_backbone, "final_layer_norm", None)
    project_out_module = getattr(model_backbone, "project_out", None)
    lm_head_module = getattr(hf_model, "lm_head", None)
    if is_last:
        if norm_module is not None:
            norm_module.to(dist_state.device)
        if final_layer_norm_module is not None:
            final_layer_norm_module.to(dist_state.device)
        if project_out_module is not None:
            project_out_module.to(dist_state.device)
        if lm_head_module is not None:
            lm_head_module.to(dist_state.device)

    shard_comm = None
    if dist_state.world_size > 1:
        shard_comm = ModelShard(
            module=nn.Identity(),
            src_rank=(dist_state.stage_rank - 1) if not is_first else None,
            dst_rank=(dist_state.stage_rank + 1) if not is_last else None,
            process_group=dist_state.process_group,
            auto_init_process_group=False,
        )

    return DistStageContext(
        model=model,
        hf_model=hf_model,
        model_backbone=model_backbone,
        all_layers=all_layers,
        stage_layers=stage_layers,
        stage_rank=dist_state.stage_rank,
        world_size=dist_state.world_size,
        device=dist_state.device,
        is_first=is_first,
        is_last=is_last,
        shard_comm=shard_comm,
        embed_module=embed_module,
        norm_module=norm_module,
        final_layer_norm_module=final_layer_norm_module,
        project_out_module=project_out_module,
        lm_head_module=lm_head_module,
    )


def _apply_output_head(ctx: DistStageContext, hidden_states: torch.Tensor):
    out = hidden_states
    if ctx.norm_module is not None:
        out = ctx.norm_module(out)
    if ctx.final_layer_norm_module is not None:
        out = ctx.final_layer_norm_module(out)
    if ctx.project_out_module is not None:
        out = ctx.project_out_module(out)

    if ctx.lm_head_module is not None:
        return ctx.lm_head_module(out)
    if hasattr(ctx.model_backbone, "embed_out") and ctx.model_backbone.embed_out is not None:
        return ctx.model_backbone.embed_out(out)
    raise RuntimeError("Could not locate LM head for distributed perplexity evaluation.")


def _chunk_nll(logits: torch.Tensor, input_ids: torch.Tensor):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    vocab_size = shift_logits.size(-1)
    return F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="mean",
    )


def init_distributed_inference(
    num_stages: int,
    stage_rank: int = -1,
    dist_backend: str = "nccl",
    dist_init_method: str = "env://",
):
    resolved_rank = stage_rank
    if resolved_rank == -1:
        resolved_rank = _env_int("RANK", "SLURM_PROCID", default=0)

    world_size = _env_int("WORLD_SIZE", "SLURM_NTASKS", default=num_stages)
    if num_stages > 1 and world_size != num_stages:
        raise ValueError(
            f"--num_stages ({num_stages}) does not match WORLD_SIZE/SLURM_NTASKS ({world_size})."
        )
    if resolved_rank < 0 or resolved_rank >= world_size:
        raise ValueError(f"Invalid stage_rank={resolved_rank} for world_size={world_size}.")

    local_rank = _env_int("LOCAL_RANK", "SLURM_LOCALID", default=0)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")

    if not dist.is_initialized():
        dist.init_process_group(
            backend=dist_backend,
            init_method=dist_init_method,
            rank=resolved_rank,
            world_size=world_size,
        )

    return DistInferenceState(
        stage_rank=resolved_rank,
        world_size=world_size,
        local_rank=local_rank,
        process_group=dist.group.WORLD,
        device=device,
    )


def fake_pack(parent_path, verbose=True):
    # Load from non-packed parent model to simulate quantization
    # WARNING: This is for PPL research only, and should not be used for any other purpose
    import re

    logprint(
        verbose,
        f"Simulating Any-Precision model from non-packed parent model at {parent_path}",
    )

    if os.path.isdir("./cache/fake_packed"):
        for file in os.listdir("./cache/fake_packed"):
            if parent_path.split("/")[-1] in file:
                logprint(
                    verbose,
                    f"Faked packed model already exists for {parent_path.split('/')[-1]}. Skipping...",
                )
                return

    # Check if D&S quantization is used
    dns = parent_path.split("/")[-1].startswith("dns")

    fields = name_splitter(parent_path)
    # get the field wrapped in ()
    for field in fields:
        if field.startswith("(") and field.endswith(")"):
            base_model_name = field[1:-1]
            break
    else:
        raise ValueError(f"Could not find base model name in {parent_path}")
    original_model_repo = base_model_name_to_hf_repo_name(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(original_model_repo)

    logprint(verbose, f"Loading original model from {original_model_repo}")
    # Load the model from the original model repo
    model = AutoModelForCausalLM.from_pretrained(
        original_model_repo, torch_dtype=torch.float16, trust_remote_code=True
    )

    logprint(verbose, f"Loading quantized weights from {parent_path}")
    # Load the qweights
    files = os.listdir(parent_path + "/weights")
    layer_count = len(files)  # this should suffice
    qweights = [None] * layer_count
    for file in tqdm(files, desc="Loading qweights", disable=not verbose):
        # filenames should be 'l0.pt'
        l = int(re.match(r"l(\d+).pt", file).group(1))
        qweights[l] = torch.load(parent_path + "/weights/" + file)

    logprint(verbose, f"Loading LUTs from {parent_path}")
    # get a list of directories in the model_path
    dirs = os.listdir(parent_path)
    dirs.remove("weights")
    if dns:
        dirs.remove("sparse")
    luts = {}
    # Only the LUT directories should remain
    for lut_dir in dirs:
        # example: lut_3
        bit = int(re.match(r"lut_(\d+)", lut_dir).group(1))
        for file in tqdm(
            os.listdir(parent_path + "/" + lut_dir),
            desc=f"Loading {bit}-bit LUTs",
            disable=not verbose,
        ):
            # example: l0.pt
            l = int(re.match(r"l(\d+).pt", file).group(1))
            if bit not in luts:
                luts[bit] = [None] * layer_count
            luts[bit][l] = torch.load(parent_path + "/" + lut_dir + "/" + file)

    # Load D&S sparse weights if they exist
    sparse_model_weights = []
    if dns:
        logprint(verbose, f"D&S quantization detected. Loading sparse weights...")
        for l in range(layer_count):
            sparse_weights = torch.load(parent_path + f"/sparse/l{l}.pt")
            sparse_model_weights.append(sparse_weights)

    logprint(verbose, f"Replacing qweights with centroids from LUTs...")

    max_bit = max(luts.keys())

    for bit in luts:
        state_dict = model.state_dict()
        for l in tqdm(
            range(layer_count),
            desc=f"Replacing qweights with {bit}-bit centroids",
        ):
            qweight = qweights[l]
            lut = luts[bit][l]
            for module_name in qweight:
                full_param_name_suffix = f".{l}.{module_name}.weight"
                matching_keys = [
                    key
                    for key in state_dict.keys()
                    if key.endswith(full_param_name_suffix)
                ]
                assert len(matching_keys) == 1, (
                    f"Expected 1 matching key, got {len(matching_keys)}"
                )
                matching_key = matching_keys[0]

                module_qweight = qweight[module_name]
                module_lut = lut[module_name]
                module_weights = []
                for row_idx in range(module_qweight.shape[0]):
                    row_weights = []
                    for group_idx in range(module_qweight.shape[1]):
                        # fetch weights from the LUT
                        group_weights = module_lut[row_idx][group_idx][
                            module_qweight[row_idx][group_idx] >> (max_bit - bit)
                        ]
                        row_weights.append(torch.from_numpy(group_weights))
                    # join the group weights
                    row_weights = torch.cat(row_weights, dim=0)
                    module_weights.append(row_weights)
                module_weights = torch.stack(module_weights)
                # Add the sparse weights if they exist
                if dns:
                    sparse_weights = sparse_model_weights[l][module_name]
                    # get the indices of the sparse weights
                    sparse_indices = sparse_weights.indices()
                    # replace the weights with the sparse weights
                    module_weights[sparse_indices[0], sparse_indices[1]] = (
                        sparse_weights.values()
                    )
                state_dict[matching_key] = module_weights

        save_path = (
            f"./cache/fake_packed/fake_anyprec-p{bit}-{parent_path.split('/')[-1]}"
        )
        os.makedirs(save_path, exist_ok=True)
        torch.save(state_dict, save_path + "/pytorch_model.bin")
        tokenizer.save_pretrained(save_path)
        model.config.save_pretrained(save_path)
        logprint(verbose, f"{bit}-bit model saved to {save_path}")


@torch.no_grad()
def auto_model_load(
    model_path,
    is_fp16=False,
    device="cuda",
    dtype=torch.float16,
    verbose=True,
    new_vocab_size=None,
    dist_state: Optional[DistInferenceState] = None,
):
    """
    Args:
        model_path: path of the model to evaluate
        device: the device to use for evaluation, either 'cuda' or 'cpu'
        dtype: the dtype to use for evaluation, either torch.float16 or torch.float32
        verbose: whether to print progress

    Returns:
        (tokenizer, model) tuple loaded from the given path, with the given device and dtype.
    """
    logprint(verbose, "Loading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    target_device = dist_state.device if dist_state is not None else device
    new_vocab_size = len(tokenizer)
    if is_fp16:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True
        )
        if dist_state is None:
            model = model.to(target_device)
        logprint(verbose, "Loading full precision model...")
    else:
        model = AnyBCQForCausalLM.from_quantized(
            model_path, new_vocab_size=new_vocab_size
        )
        if dist_state is None:
            model = model.to(target_device)
    # if os.path.basename(model_path).startswith("anyprec-"):
    #     tokenizer = AutoTokenizer.from_pretrained(model_path)
    #     model = AnyBCQForCausalLM.from_quantized(model_path).to(device)
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(model_path)
    #     model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype,
    #                                                  trust_remote_code=True).to(device)

    if dist_state is None:
        loaded_device = model.device
    else:
        loaded_device = "cpu(partitioned runtime)"
    logprint(verbose, f"{model.__class__.__name__} model loaded to device: {loaded_device}")

    tokenizer_type = get_tokenizer_type(model_path)

    if tokenizer_type is None:
        logprint(
            verbose,
            f"Unknown tokenizer type for {model_path}. Cannot use cached input tokens.",
        )

    return tokenizer_type, tokenizer, model


@torch.no_grad()
def evaluate_ppl(
    model, tokenizer, testcases, verbose=True, chunk_size=2048, tokenizer_type=None
):
    """
    Args:
        model: model to evaluate
        tokenizer: tokenizer to use
        testcases: testcases names to evaluate on, passed on to dataloader.get_loaders
        verbose: whether to print progress
        chunk_size: the size of the chunks into which the test set is split
        tokenizer_type: set to llama, llama-2, or opt to use cached input tokens
                        for the corresponding test set

    Returns:
        A dictionary of perplexity scores, with keys being the testcases names and values being the perplexity scores.

    Note that the perplexity scores are calculated over non-overlapping chunks of the test set.
    """

    if isinstance(model, AnyBCQForCausalLM):
        is_anyprec = True
    else:
        is_anyprec = False

    model.eval()

    results = {}

    supported_bits = model.precisions if is_anyprec else [None]

    for bit in supported_bits:
        if is_anyprec:
            logprint(verbose, f"<<<< Setting model precision to {bit}-bit... >>>>")
            model.set_precision(bit)

        for testcase_name in testcases:
            vprint(
                verbose,
                f"---------------------- {testcase_name} ----------------------",
            )

            input_tokens = _load_input_tokens(
                tokenizer_type, testcase_name, tokenizer, verbose
            )

            input_tokens.to(model.device)

            logprint(verbose, "Calculating perplexity...")

            seq_len = input_tokens.input_ids.size(1)
            nsamples = seq_len // chunk_size  # floor(seq_len / chunk_size)

            neg_log_likelihoods = []
            for i in tqdm(range(nsamples), disable=not verbose):
                begin_loc = i * chunk_size

                input_ids = input_tokens.input_ids[
                    :, begin_loc : begin_loc + chunk_size
                ]

                # add BOS token for Gemma-7B
                # https://github.com/huggingface/transformers/issues/29250
                if "gemma" in model.config.architectures[0].lower():
                    # Mostly harmless to other models, but a slight drop in ppl is observed
                    # Hence, we only add the BOS token for Gemma models for now
                    input_ids[:, 0] = tokenizer.bos_token_id

                with torch.no_grad():
                    outputs = model(input_ids, labels=input_ids)
                    neg_log_likelihood = outputs.loss
                    neg_log_likelihoods.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(neg_log_likelihoods).mean())
            logprint(verbose, f"Perplexity: {ppl.item()}")

            results[f"{testcase_name}:{bit}-bit"] = ppl.item()

        if not is_anyprec:
            break

    return results


@torch.no_grad()
def evaluate_ppl_distributed(
    model,
    tokenizer,
    testcases,
    dist_state: DistInferenceState,
    verbose=True,
    chunk_size=2048,
    tokenizer_type=None,
):
    if dist_state is None:
        raise ValueError("dist_state is required for distributed perplexity evaluation.")
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized for distributed evaluation.")
    if chunk_size < 2:
        raise ValueError("chunk_size must be >= 2 for perplexity evaluation.")

    if isinstance(model, AnyBCQForCausalLM):
        is_anyprec = True
    else:
        is_anyprec = False

    model.eval()
    stage_ctx = _prepare_dist_stage_context(model, dist_state)
    results = {}
    supported_bits = model.precisions if is_anyprec else [None]

    for bit in supported_bits:
        if is_anyprec:
            if verbose and dist_state.stage_rank == 0:
                logprint(verbose, f"<<<< Setting model precision to {bit}-bit... >>>>")
            model.set_precision(bit)
        dist.barrier()

        for testcase_name in testcases:
            if verbose and dist_state.stage_rank == 0:
                vprint(
                    verbose,
                    f"---------------------- {testcase_name} ----------------------",
                )

            total_nll = torch.zeros((), dtype=torch.float64, device=stage_ctx.device)
            n_chunks = 0

            if stage_ctx.is_first:
                input_tokens = _load_input_tokens(
                    tokenizer_type, testcase_name, tokenizer, verbose and stage_ctx.is_first
                )
                seq_len = input_tokens.input_ids.size(1)
                nsamples = seq_len // chunk_size
                if nsamples == 0:
                    raise ValueError(
                        f"Testcase '{testcase_name}' is shorter than chunk_size={chunk_size}."
                    )

                iterator = range(nsamples)
                if verbose and stage_ctx.stage_rank == 0:
                    iterator = tqdm(iterator)

                for i in iterator:
                    begin_loc = i * chunk_size
                    input_ids = input_tokens.input_ids[
                        :, begin_loc : begin_loc + chunk_size
                    ].to(stage_ctx.device)

                    if "gemma" in model.config.architectures[0].lower():
                        input_ids[:, 0] = tokenizer.bos_token_id

                    micro_inputs = {"input_ids": input_ids}
                    hidden_in, other = _capture_first_layer_context(
                        model=stage_ctx.hf_model,
                        all_layers=stage_ctx.all_layers,
                        micro_inputs=micro_inputs,
                    )
                    other["input_ids"] = input_ids.detach()

                    stage_hidden = _forward_layer_stack(
                        base_model=stage_ctx.hf_model,
                        layers=stage_ctx.stage_layers,
                        hidden_states=hidden_in,
                        other=other,
                    )

                    if stage_ctx.is_last:
                        logits = _apply_output_head(stage_ctx, stage_hidden)
                        total_nll += _chunk_nll(logits, other["input_ids"]).double()
                        n_chunks += 1
                    else:
                        stage_ctx.shard_comm.send_packet(stage_hidden, side_tensors=other)

                    del stage_hidden, hidden_in, other

                if not stage_ctx.is_last:
                    stage_ctx.shard_comm.send_done()
            else:
                while True:
                    done, hidden_in, other = stage_ctx.shard_comm.recv_packet()
                    if done or hidden_in is None:
                        if not stage_ctx.is_last:
                            stage_ctx.shard_comm.send_done()
                        break

                    stage_hidden = _forward_layer_stack(
                        base_model=stage_ctx.hf_model,
                        layers=stage_ctx.stage_layers,
                        hidden_states=hidden_in,
                        other=other,
                    )

                    if stage_ctx.is_last:
                        if "input_ids" not in other:
                            raise ValueError(
                                "Distributed packet missing input_ids needed for perplexity loss."
                            )
                        logits = _apply_output_head(stage_ctx, stage_hidden)
                        total_nll += _chunk_nll(logits, other["input_ids"]).double()
                        n_chunks += 1
                    else:
                        stage_ctx.shard_comm.send_packet(stage_hidden, side_tensors=other)

                    del stage_hidden, hidden_in, other

            if stage_ctx.is_last:
                if n_chunks == 0:
                    raise RuntimeError(
                        f"Last stage processed zero chunks for testcase '{testcase_name}'."
                    )
                ppl_tensor = torch.exp(total_nll / n_chunks).to(dtype=torch.float64)
            else:
                ppl_tensor = torch.zeros((), dtype=torch.float64, device=stage_ctx.device)

            dist.broadcast(ppl_tensor, src=(stage_ctx.world_size - 1))
            if dist_state.stage_rank == 0:
                logprint(verbose, f"Perplexity: {float(ppl_tensor.item())}")
                results[f"{testcase_name}:{bit}-bit"] = float(ppl_tensor.item())
            dist.barrier()

        if not is_anyprec:
            break

    return results if dist_state.stage_rank == 0 else {}


@torch.no_grad()
def run_lm_eval(tokenizer, model, tasks, num_fewshot, verbose=True):
    """Run lm-eval on the given model and tasks and return the results.

    Receives an already initialized hf model, and a list of task names.
    """
    if isinstance(model, AnyBCQForCausalLM):
        is_anyprec = True
    else:
        is_anyprec = False

    model.eval()

    results = {}

    supported_bits = model.precisions if is_anyprec else [None]

    # for bit in supported_bits[-1:]:
    for bit in supported_bits:
        if is_anyprec:
            logprint(verbose, f"<<<< Setting model precision to {bit}-bit... >>>>")
            model.set_precision(bit)

        if "gsm8k_cot" in tasks:
            model_lm = lm_eval.models.huggingface.HFLM(
                pretrained=model, tokenizer=tokenizer, add_bos_token=True
            )
            print("=================Adding BOS token for GSM8K_COT=================")
        else:
            model_lm = lm_eval.models.huggingface.HFLM(
                pretrained=model, tokenizer=tokenizer
            )
        eval_results = lm_eval.simple_evaluate(
            model=model_lm, tasks=tasks, num_fewshot=num_fewshot, batch_size="auto"
        )

        if verbose:
            logprint(verbose, json.dumps(eval_results["results"], indent=4))

        for task in tasks:
            results[f"{task}:{bit}-bit"] = eval_results["results"][task]

        if not is_anyprec:
            break

    return results


def _load_input_tokens(tokenizer_type, testcase_name, tokenizer, verbose):
    """Load input tokens from cache if available, otherwise load from dataloader and save to cache."""
    input_tokens_cache_path = f"{current_dir}/input_tokens_cache/dataloader-{tokenizer_type}-{testcase_name}-test.pt"
    if tokenizer_type and os.path.exists(input_tokens_cache_path):
        logprint(
            verbose, f"Loading cached input tokens from {input_tokens_cache_path}..."
        )
        with torch.serialization.safe_globals(
            [
                tokenizers.Encoding,
                transformers.tokenization_utils_base.BatchEncoding,
            ]
        ):
            input_tokens = torch.load(input_tokens_cache_path)
    else:
        logprint(verbose, "Loading test set...")

        raw_text = dataloader.get_loaders(testcase_name)

        logprint(verbose, "Tokenizing test set...")

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
            logprint(verbose, "Added pad token to tokenizer.")
        input_tokens = tokenizer(
            raw_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        # save input_tokens to cache
        if tokenizer_type:
            logprint(verbose, f"Caching input tokens to {input_tokens_cache_path}...")
            # we must create the directory if it doesn't exist
            os.makedirs(os.path.dirname(input_tokens_cache_path), exist_ok=True)
            torch.save(input_tokens, input_tokens_cache_path)

    return input_tokens
