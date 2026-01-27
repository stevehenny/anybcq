# AnyBCQ
# Copyright (c) 2025-present NAVER Cloud Corp.
# referenced from https://github.com/yhhhli/BRECQ/blob/main/quant/data_utils.py
# Apache-2.0

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import gc


class CachedDataset(Dataset):
    @torch.inference_mode()
    def __init__(
        self,
        model,
        cali_data,
        input_prob: float = 0.0,
        num_samples: int = 1024,
        base_bit: int = 2,
        add_bits: int = 1,
        block_list_class=torch.nn.ModuleList,
    ):
        super().__init__()
        self.device = "cuda"

        self.batch_size = cali_data.batch_size
        self.input_prob = input_prob
        self.num_samples = num_samples
        self.block_list_class = block_list_class

        self.cached_fp_input = []
        self.cached_fp_output = []
        self.precision = base_bit

        print(f"Initial Activation Caching")
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                module.to(self.device)
            if isinstance(module, block_list_class):
                module[0].to(self.device)
            if isinstance(module, torch.nn.Linear) and "project_in" in name:
                module.to(self.device)

        get_inp_out_fp = GetLayerInpOut_fp(
            fp_block=model,
            device=self.device,
            input_prob=self.input_prob,
            block_list_class=block_list_class,
        )
        for step, batch in enumerate(tqdm(cali_data)):
            if step >= self.num_samples / cali_data.batch_size:
                break

            if self.input_prob > 0.0:
                cur_out, cur_input_fp, cur_other = get_inp_out_fp(
                    batch, is_first_block=True
                )
                self.cached_fp_input.append(cur_input_fp)
                self.cached_fp_output.append(cur_out)
                self.cached_fp_other = cur_other
            else:
                cur_out = get_inp_out_fp(batch, is_first_block=True)
                self.cached_fp_output.append(cur_out)

        self.cached_q_input = {}
        # for i in range(add_bits+1):
        #     self.cached_q_input[base_bit+i] = self.cached_fp_input
        self.cached_q_input[base_bit] = self.cached_fp_input
        self.cached_q_input[base_bit + add_bits] = self.cached_fp_input

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                module.to("cpu")
                print(name, "to CPU")
            if isinstance(module, block_list_class):
                # module[0].to('cpu')
                module.to("cpu")
                print(name, "to CPU")
            if isinstance(module, torch.nn.Linear) and "project_in" in name:
                module.to("cpu")
                print(name, "to CPU")

        del get_inp_out_fp.fp_block
        del get_inp_out_fp
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def fp_data_caching(self, fp_block):
        self.cached_fp_input = self.cached_fp_output
        self.cached_fp_output = []

        fp_block = fp_block.to(self.device)
        get_inp_out_fp = GetLayerInpOut_fp(
            fp_block=fp_block,
            device=self.device,
            input_prob=self.input_prob,
            block_list_class=self.block_list_class,
        )

        for step, batch in enumerate(tqdm(self.cached_fp_input)):
            if step >= self.num_samples / self.batch_size:
                break

            cur_out, _, cur_other = get_inp_out_fp(batch, others=self.cached_fp_other)
            self.cached_fp_output.append(cur_out)
            self.cached_fp_other = cur_other

        if self.input_prob == 0.0:
            del self.cached_fp_input

        get_inp_out_fp.fp_block.to("cpu")
        fp_block.to("cpu")
        del get_inp_out_fp.fp_block
        del get_inp_out_fp
        del fp_block
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def q_data_caching(self, q_block):
        self.cached_q_output = []

        get_inp_out_q = GetLayerInpOut_q(
            q_block,
            device=self.device,
        )

        for step, batch in enumerate(tqdm(self.cached_q_input[self.precision])):
            if step >= self.num_samples / self.batch_size:
                break

            cur_output_q = get_inp_out_q(batch, self.cached_fp_other)
            self.cached_q_output.append(cur_output_q)

        self.cached_q_input[self.precision] = self.cached_q_output

        del self.cached_q_output
        del get_inp_out_q
        torch.cuda.empty_cache()

    def set_precision(self, i):
        self.precision = i

    def __getitem__(self, index):
        batch = {
            "q_input": self.cached_q_input[self.precision][index],
            "fp_input": self.cached_fp_input[index],
            "fp_output": self.cached_fp_output[index],
        }

        if self.cached_fp_other.get("attention_mask", None) is not None:
            batch["attention_mask"] = self.cached_fp_other["attention_mask"]
        if self.cached_fp_other.get("position_ids", None) is not None:
            batch["position_ids"] = self.cached_fp_other["position_ids"]
        if self.cached_fp_other.get("cache_position", None) is not None:
            batch["cache_position"] = self.cached_fp_other["cache_position"]

        return batch

    def __len__(self):
        return len(self.cached_q_input[self.precision])


class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """

    pass


class DataCacheWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.inp_data = None
        self.out_data = None
        self.other_data = None

    def forward(self, inp, **other):
        self.inp_data = inp

        output = self.module(inp, **other)

        self.out_data = output
        self.other_data = other
        raise StopForwardException


class GetLayerInpOut_fp:
    def __init__(
        self,
        fp_block,
        device,
        input_prob: float = 0.0,
        block_list_class=torch.nn.ModuleList,
    ):
        self.fp_block = fp_block
        # self.fp_block = fp_block.to(device = device)
        self.device = device
        self.input_prob = input_prob
        self.block_list_class = block_list_class

    @torch.inference_mode
    def __call__(self, model_input, is_first_block=False, others=None):
        try:
            if list == type(model_input):
                model_input[0] = model_input[0].to(self.device)

                # CNN, VisionTransformer block...
                _ = self.fp_block(model_input[0])

                # model_input[0] = model_input[0].to('cpu')
            else:
                if is_first_block:
                    # move input to GPU
                    model_input = {
                        k: (
                            v.to(device=self.device, non_blocking=True)
                            if hasattr(v, "to")
                            else v
                        )
                        for k, v in model_input.items()
                    }

                    # model inference
                    # self.fp_block = self.fp_block.to(self.device)
                    _ = self.fp_block(**model_input)
                else:
                    # model inference
                    _ = self.fp_block(model_input, **others)
        except StopForwardException:
            pass

        if is_first_block:
            for name, module in self.fp_block.named_modules():
                if isinstance(module, self.block_list_class):
                    out_data = module[0].out_data
                    inp_data = module[0].inp_data
                    other_data = module[0].other_data
                    break
        else:
            out_data = self.fp_block.out_data
            inp_data = self.fp_block.inp_data
            other_data = self.fp_block.other_data

        if "tuple" in str(type(out_data)):
            out_data = out_data[0]

        # del self.fp_block
        torch.cuda.empty_cache()
        return (
            out_data.detach(),
            inp_data.detach(),
            other_data,
        )


class GetLayerInpOut_q:
    def __init__(self, q_block, device):
        self.q_block = q_block
        self.device = device

    def __call__(self, model_input, others):

        with torch.no_grad():
            # Recalculate input with network quantized
            try:
                _ = self.q_block(model_input, **others)
            except StopForwardException:
                pass

        if "tuple" in str(type(self.q_block.out_data)):
            self.q_block.out_data = self.q_block.out_data[0]

        return self.q_block.out_data.detach()
