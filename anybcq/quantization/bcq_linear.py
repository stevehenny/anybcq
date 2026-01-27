# AnyBCQ
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import torch
import torch.nn as nn
from .packer import Packer
from .grad_scale import GradientScale

PACKER_INST = Packer()


class BCQLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        pBinary,
        bShape,
        alpha,
        packer,
        bias=None,
        in_ch_wise=False,
        bf16=False,
    ):
        if bf16:
            input = input.type(torch.bfloat16)
            bias = bias.type(torch.bfloat16)

        ctx.save_for_backward(input, alpha, bias)
        ctx.pBinary = pBinary
        ctx.bShape = bShape
        ctx.packer = packer
        ctx.in_ch_wise = in_ch_wise
        ctx.bf16 = bf16

        binary = ctx.packer.unpack(pBinary, bShape, dtype=torch.float32)
        binary = binary.to(alpha.device)

        weight = torch.einsum("oib,oiab->oia", alpha, binary)
        weight = weight.reshape([weight.shape[0], -1])

        if ctx.in_ch_wise:
            output = torch.einsum("m...i,io->m...o", input, weight)
        else:
            output = torch.einsum("m...i,oi->m...o", input, weight)

        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, alpha, bias = ctx.saved_tensors
        grad_input = grad_alpha = grad_bias = None

        binary = ctx.packer.unpack(ctx.pBinary, ctx.bShape, dtype=torch.float32)
        binary = binary.to(alpha.device)
        num_alpha = binary.shape[2]

        if ctx.needs_input_grad[0]:
            weight = torch.einsum("oib,oiab->oia", alpha, binary)
            weight = weight.reshape([weight.shape[0], -1])

            if ctx.bf16:
                weight = weight.type(torch.bfloat16)

            if ctx.in_ch_wise:
                grad_input = torch.einsum("m...o,io->m...i", grad_output, weight)
            else:
                grad_input = torch.einsum("m...o,oi->m...i", grad_output, weight)

        if ctx.needs_input_grad[3]:
            if ctx.bf16:
                binary = binary.type(torch.bfloat16)

            if ctx.in_ch_wise:
                if len(input.shape) == 2:
                    input = input.reshape([input.shape[0], -1, binary.shape[0]])
                else:
                    input = input.reshape(
                        [input.shape[0], input.shape[1], -1, binary.shape[2]]
                    )

                grad_weight_t = torch.einsum(
                    "m...o, m...ia -> iao", grad_output, input
                ).reshape(binary.shape[:-1])

                grad_alpha = torch.einsum("oia, oiab -> oib", grad_weight_t, binary)
                grad_alpha = grad_alpha / num_alpha
            else:
                if len(input.shape) == 2:
                    input = input.reshape([input.shape[0], -1, binary.shape[2]])
                else:
                    input = input.reshape(
                        [input.shape[0], input.shape[1], -1, binary.shape[2]]
                    )

                grad_alpha = torch.einsum(
                    "m...o, m...ia, oiab -> oib", grad_output, input, binary
                )
                grad_alpha = grad_alpha / num_alpha

        if bias is not None and ctx.needs_input_grad[-1]:
            grad_bias = grad_output.sum(0)

        return grad_input, None, None, grad_alpha, None, grad_bias, None, None


class BCQLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        fpBinary,
        alpha,
        beta,
        bias=None,
        packing=True,
        in_ch_wise=False,
        bf16=False,
        qbits=-1,
        group_size=-1,
        save_binary=False,
        weight=None,
    ):
        super().__init__()
        global PACKER_INST

        # check shapes
        self.in_features = in_features
        self.out_features = out_features
        self.weight = weight
        self.group_size = group_size

        self.packing = packing
        if self.packing:
            self.packer = PACKER_INST
            pBinary, bShape = self.packer.pack(fpBinary)

            # create weights(binary & alpha)
            if save_binary:
                self.register_buffer("pBinary", pBinary)
            else:
                self.pBinary = pBinary
            self.bShape = bShape
        else:
            self.register_buffer("fpBinary", fpBinary.cuda().half())
            self.num_alpha = torch.tensor(self.fpBinary.shape[2])
            self.num_alpha = self.num_alpha.cuda()

        self.in_ch_wise = in_ch_wise
        self.bf16 = bf16

        self.num_bit = qbits
        for idx in range(self.num_bit):
            setattr(
                self,
                "alpha" + str(idx),
                nn.Parameter(alpha[:, :, idx], requires_grad=True),
            )
        self.beta = nn.Parameter(beta, requires_grad=True)

        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.packing:
            alphaList = [
                getattr(self, "alpha" + str(idx)).unsqueeze(dim=2)
                for idx in range(self.num_bit)
            ]
            alpha = torch.cat(alphaList, 2)
            return BCQLinearFunction.apply(
                input,
                self.pBinary,
                self.bShape,
                alpha,
                self.packer,
                self.bias,
                self.in_ch_wise,
                self.bf16,
            )
        else:
            grad_fn = GradientScale.apply

            alphaList = [
                getattr(self, "alpha" + str(idx)).unsqueeze(dim=2)
                for idx in range(self.num_bit)
            ]
            alpha = torch.cat(alphaList, 2)

            weight = torch.einsum(
                "oib,oiab->oia",
                grad_fn(alpha, self.num_alpha),
                self.fpBinary.to(alpha.dtype),
            )
            weight = weight + self.beta.unsqueeze(-1)
            weight = weight.reshape([weight.shape[0], -1])
            weight = weight.to(input.dtype)

            if self.in_ch_wise:
                output = torch.einsum("m...i,io->m...o", input, weight)
            else:
                output = torch.einsum("m...i,oi->m...o", input, weight)

            if self.bias is not None:
                output += self.bias.unsqueeze(0).expand_as(output)
            return output

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, "
            "binary weights are unavailable to train".format(
                self.in_features, self.out_features, self.bias is not None
            )
        )


class BCQConv1D(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        fpBinary,
        alpha,
        bias=None,
        packing: bool = True,
        in_ch_wise: bool = False,
        bf16: bool = False,
        qbits: int = -1,
        group_size: int = -1,
        save_binary: bool = False,
        is_attention: bool = False,
    ):
        super().__init__()
        global PACKER_INST

        # check shapes
        self.in_features = in_features
        self.out_features = out_features
        self.nx = in_features
        self.nf = out_features

        self.packing = packing
        if self.packing:
            self.packer = PACKER_INST
            pBinary, bShape = self.packer.pack(fpBinary)

            # create weights(binary & alpha)
            if save_binary:
                self.register_buffer("pBinary", pBinary)
            else:
                self.pBinary = pBinary
            self.bShape = bShape
        else:
            if save_binary:
                if is_attention:
                    b_shape = fpBinary.shape
                    self.query_fpBinary = nn.Parameter(
                        fpBinary[
                            :, int(0 * b_shape[1] / 3) : int(1 * b_shape[1] / 3), :, :
                        ]
                    )
                    self.key_fpBinary = nn.Parameter(
                        fpBinary[
                            :, int(1 * b_shape[1] / 3) : int(2 * b_shape[1] / 3), :, :
                        ]
                    )
                    self.value_fpBinary = nn.Parameter(
                        fpBinary[
                            :, int(2 * b_shape[1] / 3) : int(3 * b_shape[1] / 3), :, :
                        ]
                    )
                else:
                    self.register_buffer("fpBinary", fpBinary)
            else:
                self.fpBinary = fpBinary
            self.num_alpha = group_size

        self.in_ch_wise = in_ch_wise
        self.bf16 = bf16
        self.num_bit = qbits
        self.is_attention = is_attention

        a_shape = alpha.shape
        if is_attention:
            for qkv_idx, layer_name in enumerate(["query", "key", "value"]):
                for idx in range(self.num_bit):
                    setattr(
                        self,
                        f"{layer_name}_alpha" + str(idx),
                        nn.Parameter(
                            alpha[
                                :,
                                int(qkv_idx * a_shape[1] / 3) : int(
                                    (qkv_idx + 1) * a_shape[1] / 3
                                ),
                                idx,
                            ],
                            requires_grad=True,
                        ),
                    )
        else:
            for idx in range(self.num_bit):
                setattr(
                    self,
                    "alpha" + str(idx),
                    nn.Parameter(alpha[:, :, idx], requires_grad=True),
                )

        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.packing:
            alphaList = [
                getattr(self, "alpha" + str(idx)).unsqueeze(dim=2)
                for idx in range(self.num_bit)
            ]
            alpha = torch.cat(alphaList, 2)
            return BCQLinearFunction.apply(
                input,
                self.pBinary,
                self.bShape,
                alpha,
                self.packer,
                self.bias,
                self.in_ch_wise,
                self.bf16,
            )
        else:
            grad_fn = GradientScale.apply

            if self.is_attention:
                alphaList = [
                    getattr(self, "query_alpha" + str(idx)).unsqueeze(dim=2)
                    for idx in range(self.num_bit)
                ]
                self.query_alpha = torch.cat(alphaList, 2)
                alphaList = [
                    getattr(self, "key_alpha" + str(idx)).unsqueeze(dim=2)
                    for idx in range(self.num_bit)
                ]
                self.key_alpha = torch.cat(alphaList, 2)
                alphaList = [
                    getattr(self, "value_alpha" + str(idx)).unsqueeze(dim=2)
                    for idx in range(self.num_bit)
                ]
                self.value_alpha = torch.cat(alphaList, 2)

                alpha = torch.cat(
                    (self.query_alpha, self.key_alpha, self.value_alpha), dim=1
                )
                binary = torch.cat(
                    (self.query_fpBinary, self.key_fpBinary, self.value_fpBinary), dim=1
                )

                weight = torch.einsum(
                    "oib,oiab->oia", grad_fn(alpha, self.num_alpha), binary
                )
                weight = weight.reshape([weight.shape[0], -1])
                weight = weight.to(input.dtype)
            else:
                alphaList = [
                    getattr(self, "alpha" + str(idx)).unsqueeze(dim=2)
                    for idx in range(self.num_bit)
                ]
                alpha = torch.cat(alphaList, 2)

                weight = torch.einsum(
                    "oib,oiab->oia", grad_fn(alpha, self.num_alpha), self.fpBinary
                )
                weight = weight.reshape([weight.shape[0], -1])
                weight = weight.to(input.dtype)

            if self.in_ch_wise:
                weight = weight.transpose(0, 1).contiguous()

            size_out = input.size()[:-1] + (self.nf,)
            output = torch.addmm(self.bias, input.view(-1, input.size(-1)), weight)
            output = output.view(size_out)

            return output

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, "
            "binary weights are unavailable to train".format(
                self.in_features, self.out_features, self.bias is not None
            )
        )
