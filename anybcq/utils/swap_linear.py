# AnyBCQ
# Copyright (c) 2025-present NAVER Cloud Corp.
# Referenced from https://github.com/microsoft/LoRA
# Apache-2.0

import torch
import torch.nn as nn
import sys, os

from anybcq.inference.AnyBCQLinear import AnyBCQLinear
from anybcq.quantization.bcq_asym import quantize, batch_cg_torch
from anybcq.quantization.bcq_linear import BCQLinear, BCQConv1D
from anybcq.quantization.packer import Packer32


def swap_quant_linear(
    layer, n_bits, n_rounds, group_size,
    in_ch_wise=False, packing=True, linear_type='bcq',
    asymmetric = False,
):
    # quantization method
    w_hat, binary, alpha, beta, _ = quantize(
        layer.weight, 
        qbits      = n_bits, 
        rounds     = n_rounds, 
        group_size = group_size, 
        transpose  = in_ch_wise,
        asymmetric = asymmetric,
        # asymmetric = False,
    )

    out_features, in_features = layer.weight.size()
    if layer.bias is not None:
        bias = layer.bias.clone().detach()
    else:
        bias = None

    # del layer.weight  # for sure

    if linear_type == 'bcq':
        new_layer = BCQLinear(
            in_features  = in_features, 
            out_features = out_features,
            weight     = layer.weight.clone().detach(),
            fpBinary     = binary.clone().detach(),
            alpha        = alpha.clone().detach(), 
            beta         = beta.clone().detach(),
            bias         = bias, 
            packing      = packing, 
            in_ch_wise   = in_ch_wise,
            qbits        = n_bits, 
            group_size   = group_size,
        )
    else:
        raise ValueError(f'Invalid linear_type({linear_type})')
    
    layer.to('cpu')
    del layer.weight
    del layer.bias  # for sure
    del layer
    del binary
    del alpha
    del beta
    del w_hat  # for sure
    torch.cuda.empty_cache()

    return new_layer


def swap_quant_model(
    model, 
    n_bits: int          = 32,
    n_rounds: int        = 0,
    group_size: int      = -1,
    in_ch_wise: bool     = False, 
    packing: bool        = True,
    asymmetric: bool        = True,
    swap_type: str      = 'bcq',
    linear_type: str      = 'bcq',
    wq_params: dict      = None,
    aq_params: dict      = None,
):
    # Some part of the code was from https://github.com/microsoft/LoRA
    def _get_submodules(model, key):
        parent = model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)

    with torch.no_grad():
        from transformers.pytorch_utils import Conv1D
        for name, layer in model.named_modules():
            if (isinstance(layer, nn.Linear) and 
                'lm_head' not in name and 
                'project' not in name
            ):
                print(f'Swap {swap_type} linear with {name}')
                parent, target, target_name = _get_submodules(model, name)
                if swap_type == 'bcq':
                    new_module = swap_quant_linear(
                        layer, 
                        n_bits     = n_bits, 
                        n_rounds   = n_rounds,
                        group_size = group_size, 
                        in_ch_wise = in_ch_wise, 
                        packing    = packing,
                        linear_type= linear_type,
                        asymmetric = asymmetric,
                    )
                else:
                    raise ValueError(f"invalid swap_type, {swap_type}")
                _replace_module(parent, target_name, new_module, target)
                layer.to('cpu')
                del layer
                torch.cuda.empty_cache()

def delete_original_weight(
    model,
):
    with torch.no_grad():
        for name, layer in model.named_modules():
            if (isinstance(layer, BCQLinear)):
                delattr(layer, 'weight')
                print(name, "original weight deleted")

                torch.cuda.empty_cache()

def add_onebit_model(
    model, 
    n_rounds=20,
    group_size: int      = -1,
    in_ch_wise: bool     = False, 
    packing: bool        = True,
    swap_type: str      = 'bcq',
):
    # Some part of the code was from https://github.com/microsoft/LoRA
    def _get_submodules(model, key):
        parent = model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)

    with torch.no_grad():
        for name, layer in model.named_modules():
            if (isinstance(layer, BCQLinear)):
                print(f'[Add extra one-bit to {name}]')
                parent, target, target_name = _get_submodules(model, name)
                if swap_type == 'bcq':
                    new_module = add_onebit_linear_v2(
                        layer,
                        n_rounds,
                    )
                else:
                    raise ValueError(f"invalid swap_type, {swap_type}")
                _replace_module(parent, target_name, new_module, target)

                torch.cuda.empty_cache()

def add_onebit_linear_v2(layer: nn.Module,
                         rounds: int = 5):
    """
    Add a 3‑rd bit‑plane to an existing 2‑bit BCQLinear layer.
    The first two bit‑planes (B_fixed) are frozen.
    """

    # ------------------------------------------------------------------
    # 0.  Shape conventions used inside BCQLinear
    # ------------------------------------------------------------------
    # layer.fpBinary         : [O, I_grp, G, 2]      (binary codes)
    # alpha{k}               : [O, I_grp]            (per‑group scales)
    # layer.weight           : [O, I_grp*G]          (full‑precision)
    # After this routine:
    #   layer.fpBinary       : [O, I_grp, G, 3]
    #   alpha{0,1,2} updated/added
    # ------------------------------------------------------------------

    device = layer.weight.device
    dtype   = layer.weight.dtype
    

    B_fixed = layer.fpBinary.to(dtype)                       # (O,I_grp,G,2)  **FIX**

    # (α0, α1)  ->  [O, I_grp, 2]
    Alpha_prev = torch.stack(
        [getattr(layer, f'alpha{idx}').to(dtype) for idx in range(layer.num_bit)],
        dim=-1
    )# (O,I_grp,2)

    Alpha = torch.cat([Alpha_prev,
                       torch.zeros_like(Alpha_prev[..., :1])], dim=-1)  # (O,I_grp,3)

    O, I_grp, G, _ = B_fixed.shape
    W_full = layer.weight.view(O, I_grp, G).to(dtype)
    layer.num_bit += 1 # Early update layer config
    
    if hasattr(layer, 'beta'):
        beta = layer.beta.to(dtype)  # [O, I_grp]
        W_residual = W_full - beta.unsqueeze(-1)  # [O, I_grp, G]
    else:
        W_residual = W_full
    

    def _matmul_bits(alpha, B):
        # einsum:  α(o,i,b) * B(o,i,g,b)  ->  W_q(o,i,g)
        return torch.einsum('oib,oiab->oia', alpha, B)

    # ------------------ 2)  Alternating refinement ------------------
    for _ in range(rounds):
        dummy_zero = torch.zeros_like(B_fixed[..., :1])      # (O,I_grp,G,1) **FIX**
        # if hasattr(layer, 'beta'):
        #     W_residual = W_full - layer.beta.unsqueeze(-1).to(dtype)
        # else:
        #     W_residual = W_full
        
        res = W_residual - _matmul_bits(Alpha,
                                          torch.cat([B_fixed, dummy_zero], dim=-1))
        B_new = res.sign()                              # (O,I_grp,G)

        B_concat = torch.cat([B_fixed, B_new.unsqueeze(-1)], dim=-1)  # (O,I_grp,G,3)

        H = O * I_grp                                        # total rows
        B2 = B_concat.reshape(H, G, layer.num_bit).transpose(1, 2)       # (H,3,G)  == Bt
        BtB = B2 @ B2.transpose(1, 2)                        # (H,3,3)
        Btw = B2 @ W_residual.reshape(H, G, 1)                   # (H,3,1)

        Alpha2 = batch_cg_torch(BtB.float(), Btw.float().squeeze(-1),
                                x=Alpha.float().reshape(H, layer.num_bit))       # (H,3)
        Alpha2 = torch.clamp(Alpha2, torch.finfo(torch.float16).min, 
                             torch.finfo(torch.float16).max).to(torch.float16)
        Alpha  = Alpha2.reshape(O, I_grp, layer.num_bit)

    layer.fpBinary = torch.cat([layer.fpBinary, B_new.unsqueeze(-1)], dim=-1)

    
    for idx in range(layer.num_bit):
        name = f'alpha{idx}'
        if hasattr(layer, name):
            getattr(layer, name).data.copy_(Alpha[..., idx])
        else:
            layer.register_parameter(name, nn.Parameter(Alpha[..., idx]))

    return layer




def save_alpha_and_beta_in_bcqlinear(module: nn.Module):
    """
    """
    for name, child in list(module.named_children()):
        if isinstance(child, BCQLinear):
            alpha_list = [
                getattr(child, f'alpha{idx}').unsqueeze(2)
                for idx in range(child.num_bit)
            ]
            alpha = torch.cat(alpha_list, dim=2)  # shape [OC, IC, q]
            alpha = alpha.permute(1, 2, 0).contiguous()
            beta = child.beta.transpose(0,1).to(torch.half).to(alpha.device)
            setattr(child, f'alpha_{child.num_bit}', alpha.to(torch.half))
            setattr(child, f'beta_{child.num_bit}', beta)
            print(child)
        
        else:
            save_alpha_and_beta_in_bcqlinear(child)

def replace_bcq_with_lutgemm(module: nn.Module):
    """
    """
    packer = Packer32()
    for name, child in list(module.named_children()):
        if isinstance(child, BCQLinear):
            binary_pack = packer.pack(child.fpBinary)

            has_bias = (child.bias is not None)
            new_linear = AnyBCQLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                supported_bits=range(2, child.num_bit+1),
                group_size=child.group_size,
                bias=has_bias,
            )
            with torch.no_grad():
                new_linear.qweight.copy_(binary_pack)
                # (TODO) fixed to 2 -> min precision
                for bw in getattr(new_linear, "supported_bits", []):
                    alpha_name = f"alpha_{bw}"
                    beta_name = f"beta_{bw}"
                    if hasattr(child, alpha_name) and hasattr(new_linear, alpha_name) and hasattr(child, beta_name) and hasattr(new_linear, beta_name):
                        dst_alpha = getattr(new_linear, alpha_name)
                        src_alpha = getattr(child, alpha_name)
                        dst_beta = getattr(new_linear, beta_name)
                        src_beta = getattr(child, beta_name)
                        dst_alpha.copy_(src_alpha.to(device=dst_alpha.device, dtype=dst_alpha.dtype))
                        dst_beta.copy_(src_beta.to(device=dst_beta.device, dtype=dst_beta.dtype))
                if has_bias:
                    new_linear.bias.copy_(child.bias.to(torch.half))
            
            setattr(module, name, new_linear)
            del child
        
        else:
            replace_bcq_with_lutgemm(child)

def replace_bcq_with_linear(module: nn.Module):
    """
    """
    for name, child in list(module.named_children()):
        if isinstance(child, BCQLinear):
            alpha_list = [
                getattr(child, f'alpha{idx}').unsqueeze(2)
                for idx in range(child.num_bit)
            ]
            alpha = torch.cat(alpha_list, dim=2)  # shape [O, I, B]

            weight = torch.einsum('oib,oiab->oia', alpha, child.fpBinary.to(alpha.dtype))
            weight = weight.reshape(child.out_features, child.in_features)
            
            has_bias = (child.bias is not None)
            bias = child.bias.detach() if has_bias else None
            
            new_linear = nn.Linear(child.in_features, child.out_features, bias=has_bias)
            new_linear.weight.data.copy_(weight)
            if has_bias:
                new_linear.bias.data.copy_(bias)
            
            setattr(module, name, new_linear)
        
        else:
            replace_bcq_with_linear(child)


def set_precision_model(
    model, 
    precision
):

    with torch.no_grad():
        for name, layer in model.named_modules():
            if isinstance(layer, AnyBCQLinear):
                layer.set_precision(precision)
