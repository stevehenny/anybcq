# AnyBCQ
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import torch
import torch.nn as nn
from anybcq.inference.plugin import *

class AnyBCQLinear(nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features,
        supported_bits, 
        group_size,
        bias=False, 
        precision=None,
        dtype=torch.half):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.supported_bits = sorted(set(supported_bits))
        self.max_prec       = max(self.supported_bits)
        self.precision = precision or self.max_prec
        if group_size == -1:
            self.group_size = in_features
        else:
            self.group_size = group_size
        self.dtype = dtype

        self.register_buffer(
            'qweight',
            torch.empty(
                (in_features//32, self.max_prec, out_features), 
                dtype=torch.int32)
        )
        self.register_buffer(
            'q_bias',
            torch.empty(
                (in_features // self.group_size, out_features), 
                dtype=self.dtype)
        )

        self.alpha_names = {}
        self.beta_names = {}
        for bw in self.supported_bits:
            buf_name = f"alpha_{bw}"
            self.register_buffer(
                buf_name,
                torch.empty(
                    (in_features // self.group_size, bw, out_features), 
                    dtype=dtype)
            )
            self.alpha_names[bw] = buf_name
            buf_name = f"beta_{bw}"
            self.register_buffer(
                buf_name,
                torch.empty(
                    (in_features // self.group_size, out_features), 
                    dtype=dtype)
            )
            self.beta_names[bw] = buf_name
       
        if bias:
            self.register_buffer(
                "bias",
                torch.empty((out_features,), dtype=self.dtype)
            )
        else:
            self.bias = None

        self.output = torch.zeros((1, 1, out_features), dtype=self.dtype, device='cuda')

    def set_precision(self, precision: int):
        if precision not in self.supported_bits:
            raise ValueError(f"precision {precision} not in supported set {self.supported_bits}")
        self.precision = precision
    
    def _get_alpha(self, w_bits: int):
        return getattr(self, f"alpha_{w_bits}")
    
    def _get_beta(self, w_bits: int):
        return getattr(self, f"beta_{w_bits}")

    def _gemm(self, x, w_bits, alpha, beta):
        """
        x : (B, T, in_features)
        w_bits: precision
        alpha: (num_groups, w_bits, out_features)
        return -> (B, T, out_features)
        """
        B, T, _ = x.shape
        weight = anybcq_dequant(
            self.qweight, alpha, beta, w_bits, self.max_prec, self.group_size
        )

        x_flat = x.reshape(-1, self.in_features)
        y_flat = torch.matmul(x_flat, weight)

        return y_flat.reshape(B, T, self.out_features)

    def forward(self, x, **kwargs):
        """
        x : (B, T, in_features)
        precision : None → self.precision / int 
        """
        # assert(x.shape[0] == 1)
        
        if 'precision' in kwargs:
            w_bits = kwargs['precision']
        else:
            w_bits = self.precision
        
        if w_bits not in self.supported_bits:
            raise ValueError(f"Unsupported precision {w_bits}; supported: {self.supported_bits}")
        
        alpha = self._get_alpha(w_bits)    
        beta = self._get_beta(w_bits)    
        
        if x.numel() // self.in_features == 1:
            self.output.zero_()
            anybcq_gemv(
                x, self.output,
                self.qweight, alpha, beta,
                w_bits, self.max_prec, self.group_size
            )
            out = self.output
        else:
            out = self._gemm(x, w_bits, alpha, beta)

        if self.bias is not None:
            out += self.bias
        return out
