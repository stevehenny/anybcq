import torch
import torch.nn as nn

class GradientScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n_alpha):
        ctx.n_alpha = n_alpha
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output / ctx.n_alpha
        return grad_output, None

