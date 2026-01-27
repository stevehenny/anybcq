# AnyBCQ
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import torch
import numpy as np

class Packer:
    def __init__(self):
        self.s = torch.from_numpy(np.array([1, 2, 4, 8, 16, 32, 64, 128])).view(
            [-1, 1])
        if torch.cuda.is_available():
            self.s = self.s.cuda()
        self.w_pool = {}

    def __get_weight(self, shape, dtype):
        key = np.prod(shape)
        if key not in self.w_pool.keys():
            self.w_pool[key] = torch.zeros(shape, dtype=dtype)
            if torch.cuda.is_available():
                self.w_pool[key] = self.w_pool[key].cuda()
        return self.w_pool[key].reshape(shape)

    def pack(self, b):
        shape = b.shape
        p_b = b
        if torch.cuda.is_available():
            p_b = p_b.cuda()
        p_b = (p_b + 1) / 2  # (-1., +1.) -> (0, 1)
        p_b = torch.reshape(p_b, [8, -1]).type(torch.uint8)
        p_b = p_b * self.s
        p_b = p_b.sum(0)
        p_b = p_b.type(torch.uint8)
        return p_b, shape

    def unpack(self, pb, shape, dtype=torch.float16):
        b = self.__get_weight(shape, dtype).view([8, -1])
        for i in range(8):
            b[i] = (pb & 1)  # (pB%2)
            pb = pb >> 1  # //2
        b = b * 2 - 1
        b = b.reshape(shape)
        return b

class Packer32:
    def __init__(self):
        s = (1 << torch.arange(32, dtype=torch.int64)).view(32, 1)  # [32,1]
        if torch.cuda.is_available():
            s = s.cuda()
        self.s = s
        self.w_pool = {}

    def __get_weight(self, shape, dtype):
        key = (shape, dtype)
        if key not in self.w_pool:
            t = torch.zeros(shape, dtype=dtype)
            if torch.cuda.is_available():
                t = t.cuda()
            self.w_pool[key] = t
        return self.w_pool[key].reshape(shape)

    @torch.no_grad()
    def pack(self, b: torch.Tensor):
        """
        Input:
          b: {-1, +1} float tensor, shape = [OC, IC/group, group, num_bit]
        Output:
          packed: int32 tensor, shape = [IC/32, num_bit, OC]
        Description:
            IC axis (= (IC/group)*group) to pack 32 consecutive uint32s
            (PyTorch does not have uint32, so it is stored as int32; same bit pattern)
        """
        assert b.dim() == 4, f"b must be [OC, IC/group, group, num_bit], got {tuple(b.shape)}"
        OC, ICg, G, B = b.shape
        IC = ICg * G
        assert IC % 32 == 0, f"IC must be divisible by 32, got IC={IC}"

        device = b.device
        # (-1, +1) -> (0, 1)
        bits = ((b + 1.0) / 2.0).to(torch.int64)  # [OC, ICg, G, B]

        # [OC, ICg, G, B] -> [IC, B, OC]
        bits = bits.permute(1, 2, 3, 0).contiguous().view(IC, B, OC)  # [IC, B, OC]

        # [IC, B, OC] -> [IC//32, 32, B, OC]
        bits32 = bits.view(IC // 32, 32, B, OC)  # [I32,32,B,O]

        shifts = self.s
        if shifts.device != device:
            shifts = shifts.to(device)
        shifts = shifts.view(32, 1, 1)  # [32,1,1]

        packed64 = (bits32 * shifts).sum(dim=1)  # [I32,B,O], int64

        out_shape = (IC // 32, B, OC)
        packed = self.__get_weight(out_shape, torch.int32)
        if packed.device != device:
            packed = packed.to(device)
        packed.copy_(packed64.to(torch.int32))
        return packed  # [IC/32, num_bit, OC]

    @torch.no_grad()
    def unpack(self, pb: torch.Tensor, target_shape, dtype=torch.float16):
        """
        Input:
          pb: int32 bit-packed tensor, shape = [IC/32, num_bit, OC]
          target_shape: (OC, IC/group, group, num_bit)
          dtype: return dtype (default fp16)
        Output:
          b: {-1, +1} float tensor, shape = [OC, IC/group, group, num_bit]
        """
        assert pb.dim() == 3, f"pb must be [IC/32, num_bit, OC], got {tuple(pb.shape)}"
        OC, ICg, G, B = target_shape
        I32, B2, O2 = pb.shape
        assert B2 == B and O2 == OC, \
            f"Inconsistent num_bit/OC: packed={pb.shape}, target={target_shape}"
        IC = I32 * 32
        assert ICg * G == IC, \
            f"IC derived from target_shape ({ICg}*{G}) must equal IC={IC} from packed"

        device = pb.device
        # [I32, B, O] -> [I32, 1, B, O]
        pb64 = pb.to(dtype=torch.int64).unsqueeze(1)  # [I32,1,B,O]

        masks = self.s
        if masks.device != device:
            masks = masks.to(device)
        masks = masks.view(1, 32, 1, 1)

        bits32 = ((pb64 & masks) != 0).to(torch.int64)

        # [I32,32,B,O] -> [IC,B,O]
        bits = bits32.permute(0, 1, 2, 3).contiguous().view(IC, B, OC)  # [IC,B,O]

        # [IC,B,O] -> [O,IC,B] -> [OC, ICg, G, B]
        bits = bits.permute(2, 0, 1).contiguous().view(OC, ICg, G, B)  # {0,1}

        b = (bits.to(dtype) * 2) - 1
        if b.device != device:
            b = b.to(device)
        return b
