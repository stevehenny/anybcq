# Referenced from: https://github.com/snu-mllab/GuidedQuant/blob/main/any_precision/modules/AnyPrecisionLinear.py

import torch
import torch.nn as nn

from anybcq.inference.plugin import *

class AnyPrecisionLinear(nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features, 
        supported_bits, 
        precision=None,
        bias=True, 
        device=None,
        dtype=torch.half):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.supported_bits = supported_bits
        self.precision = precision or max(self.supported_bits)
        self.dtype = dtype

        self.register_buffer(
            'qweight',
            torch.empty((max(supported_bits), out_features, in_features // 32), dtype=torch.int32, device=device)
        )

        for bit in supported_bits:
            self.register_buffer(
                f'lut{bit}',
                torch.empty((out_features, 2 ** bit), dtype=dtype, device=device)
            )

        if bias:
            self.register_buffer(
                "bias",
                torch.empty((out_features,), dtype=dtype, device=device)
            )
        else:
            self.bias = None

        self.output = torch.zeros((1, 1, self.out_features), dtype=self.dtype, device='cuda')
    
    def set_precision(self, precision: int):
        if precision not in self.supported_bits:
            raise ValueError(f"precision {precision} not in supported set {self.supported_bits}")
        self.precision = precision
    
    def _gemm(self, x, w_bits, lut):
        B, T, _ = x.shape
        weight = anyprec_dequant(self.qweight, lut, w_bits)
        x_flat = x.reshape(-1, self.in_features)
        y_flat = torch.matmul(x_flat, weight.T)
        return y_flat.reshape(B, T, self.out_features)

    def forward(self, x, **kwargs):

        """
        x : (B, T, in_features)
        precision : None → self.bitwidth / int 
        """
        
        if 'precision' in kwargs:
            w_bits = kwargs['precision']
        else:
            w_bits = self.precision
        
        if w_bits not in self.supported_bits:
            raise ValueError(f"Unsupported bitwidth {w_bits}; supported: {self.supported_bits}")
        
        if x.numel() // self.in_features == 1:
            self.output.zero_()
            anyprec_gemv(
                x, self.qweight, 
                self._buffers[f'lut{w_bits}'], self.output, w_bits)
            out = self.output
        else:
            out = self._gemm(x, w_bits, self._buffers[f'lut{w_bits}'])

        if self.bias is not None:
            out += self.bias
        return out
