# Copied from: https://github.com/snu-mllab/GuidedQuant/blob/main/inference/plugin.py

import torch
import custom_kernel

"""
Any-Precision
"""
@torch.library.custom_op("plugin::anyprec_gemv", mutates_args={"output"})
def anyprec_gemv(
    x: torch.Tensor, 
    q_weight: torch.Tensor, 
    lut: torch.Tensor, 
    output:torch.Tensor, 
    bitwidth:int) -> None:
    custom_kernel.anyprec_gemv(x, output, q_weight, lut, bitwidth)

@anyprec_gemv.register_fake
def _(x, q_weight, lut, output, bitwidth):
    return None

#@torch.library.custom_op("plugin::anyprec_dequant", mutates_args=())
def anyprec_dequant(
    q_weight: torch.Tensor, 
    lut: torch.Tensor, bitwidth:int) -> torch.Tensor:
    weight = custom_kernel.anyprec_dequant(q_weight, lut, bitwidth)
    return weight

@torch.library.custom_op("plugin::anybcq_gemv", mutates_args={"output"})
def anybcq_gemv(
    x: torch.Tensor, 
    output: torch.Tensor, 
    q_weight: torch.Tensor, 
    alpha: torch.Tensor, 
    q_bias: torch.Tensor, 
    bitwidth: int,
    max_num_bits: int, group_size: int) -> None:
    custom_kernel.anybcq_gemv(
        x, output, q_weight, alpha, q_bias, bitwidth, max_num_bits, group_size)

def anybcq_dequant(
    q_weight: torch.Tensor, 
    alpha: torch.Tensor, 
    q_bias: torch.Tensor, 
    bitwidth: int, 
    max_num_bits: int, group_size: int) -> None:
    weight = custom_kernel.anybcq_dequant(
        q_weight, alpha, q_bias, bitwidth, max_num_bits, group_size)
    return weight

@anybcq_gemv.register_fake
def _(x, output, q_weight, alpha, q_bias, bitwidth, max_num_bits, group_size):
    return None
