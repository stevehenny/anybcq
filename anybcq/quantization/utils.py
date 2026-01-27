# AnyBCQ
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import torch
import torch.nn as nn
from itertools import product
import numpy as np
import sys
from tqdm import tqdm 
from bcq_linear import BCQLinear


def make_symm(weight):
    pos_values = torch.unique(torch.abs(weight))
    return torch.cat((pos_values, -pos_values)).sort().values

def find_alpha(weight, n_bits):
    # weight: 1-d vector with length of group size 

    # when len(unique_values) < 2**n_bits
    unique_values = torch.unique(weight, sorted=True)
    if len(unique_values) < 2**n_bits:
        unique_values = make_symm(unique_values)
    
    # find alpha
    alpha = []
    for i in range(1, n_bits+1):
        a = (unique_values[i] + unique_values[-1]) / 2
        alpha.append(a)

    return alpha

def find_binary(weight, n_bits, alpha):
    # weight: 1-d vector with length of group size 

    # dtype change
    alpha = [al.detach() for al in alpha]

    signs = list(product([-1, 1], repeat=n_bits)) 
    product_vals = [np.array(alpha)*np.array(sign) for sign in signs]

    # make look-up table
    LUT = {np.sum(val): sign for sign, val in zip(signs, product_vals)}  

    # find binary 
    binary = [LUT[w.item()] for w in weight]

    return binary
    
def swap_bcq_from_dequanted(name, layer, n_bits, group_size, device='cuda:0',
                            in_ch_wise=False, packing=True, bf16=True):
    # input: dequanted layer 
    # output: BCQLinear layer

    weight = layer.weight.to(device)
    in_features, out_features = weight.shape[0], weight.shape[1]

    # reshape 
    if in_ch_wise:
        # TODO: in-ch-wise implementation
        pass
    else:
        weight = weight.reshape(in_features, out_features//group_size, group_size)
        # alpha: [in, out//g, bits] / binary: [in, out//g, g, bits]
        Alpha = torch.zeros(in_features, out_features // group_size, n_bits)
        B = torch.zeros(in_features, out_features // group_size, group_size, n_bits)
        
    # find alpha & binary
    # TODO: find a way to avoid double for loop
    for i in range(in_features):
        for j in range(out_features//group_size):
            Alpha[i][j] = torch.tensor(find_alpha(weight[i][j], n_bits))
            B[i][j] = torch.tensor(find_binary(weight[i][j], n_bits, Alpha[i][j]))

    # BCQLinear
    bcq_layer = BCQLinear(
        in_features, out_features, B, Alpha, 
        bias=None, packing=True, in_ch_wise=in_ch_wise, bf16=True,
        qbits=n_bits, group_size=group_size, save_binary=False
        )
    del weight, B, Alpha 

    return bcq_layer


# Changes every Linear layer to BCQLinear layer. 
# Usage: swap_bcq_model_from_dequanted (model, **kwargs)
def swap_bcq_model_from_dequanted(model, n_bits, group_size, device='cuda:0', 
                                 in_ch_wise=False, packing=True, bf16=False):

    # Some part of the code was from https://github.com/microsoft/LoRA
    def _get_submodules(model, key):
        parent = model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)

    for name, layer in tqdm(model.named_modules()):
        if isinstance(layer, nn.Linear):
            parent, target, target_name = _get_submodules(model,name)
            new_module = swap_bcq_from_dequanted(name, layer, device=device, 
                            n_bits=n_bits, group_size=group_size,
                            in_ch_wise=in_ch_wise, packing=packing, bf16=bf16)
            _replace_module(parent, target_name, new_module, target)

