# Referenced from: https://github.com/allenbai01/ProxQuant/blob/1a9240e2afd1f49257ab5527b0f61862481a0e9d/reg.py

import time
import torch
import numpy as np
from tqdm import tqdm


def quantize(
    w,
    qbits,
    rounds=15,
    group_size=-1,
    transpose=False,
    exponent=0.0,
    clipping=1.0,
    pruning=0.0,
    use_bst=True,
    asymmetric=True,
):
    """
    Quantize weights with toggle:
      - asymmetric=True  -> ABCQ (w ≈ beta + Σ alpha_i b_i)
      - asymmetric=False -> BCQ  (w ≈ Σ alpha_i b_i), beta=0

    Returns (ret, B, alpha, beta, mask)
      ret:   quantized weights (same shape as w)
      B:     [row, n_group, group_size, qbits]
      alpha: [row, n_group, qbits]
      beta:  [row, n_group]     (zeros in symmetric mode)
      mask:  (wf != 0.0)
    """
    w_ = w.clone().cuda()
    if transpose:
        assert len(w_.shape) == 2, f"Check your weight shape {w_.shape}"
        w_ = w_.transpose(1, 0).contiguous()

    orig_shape = w_.shape
    group_size = group_size if group_size > 0 else orig_shape[-1]
    w_ = w_.view([-1, group_size])  # [rows, cols_per_group]

    w_abs = w_.abs()
    ws, _ = w_abs.view(-1).sort()
    wf = torch.ones(w_.shape, dtype=torch.float32, device=w.device)
    if pruning > 0.0:
        wf = wf * (w_ != 0.0)
    if exponent > 0.0 or clipping < 1.0:
        wf = w_abs / (w_abs.max() + 1e-12)
    if clipping < 1.0:
        c_th = ws[int(ws.size(0) * clipping)].item()
        wf = wf * (w_abs.max() / (c_th + 1e-12))
        wf = torch.minimum(wf, torch.ones_like(wf))
    if exponent > 0.0:
        wf = wf**exponent
    if pruning > 0.0:
        p_th = ws[int(ws.shape[0] * pruning)].item()
        mask = w_abs > p_th
        wf = wf * mask
        w_ = w_ * mask

    wf = wf.to(w_.device)

    if asymmetric:
        ret_no_bias, B, alpha, beta = greedy_mean_torch_asym(w_, n_bits=qbits, wf=wf)

        if rounds > 0 and qbits > 1:
            for _ in tqdm(range(rounds)):
                ret_no_bias, B, alpha, beta = refine_mean_torch_asym(
                    w_, ret_no_bias, B, alpha, beta, wf=wf, use_bst=use_bst
                )

        ret = (ret_no_bias + beta.view(-1, 1)).view(orig_shape)
    else:
        ret, B, alpha = greedy_mean_torch(w_, n_bits=qbits, wf=wf)

        if rounds > 0 and qbits > 1:
            for _ in tqdm(range(rounds)):
                ret, B, alpha = refine_mean_torch(
                    w_, ret, B, alpha, wf=wf, use_bst=use_bst
                )

        ret = ret.view(orig_shape)
        beta = torch.zeros(w_.shape[0], device=w_.device, dtype=ret.dtype)

    if transpose:
        ret = ret.transpose(1, 0).contiguous()

    B = B.reshape([orig_shape[0], orig_shape[1] // group_size, group_size, qbits])
    alpha = alpha.reshape([orig_shape[0], orig_shape[1] // group_size, qbits])
    beta = beta.reshape([orig_shape[0], orig_shape[1] // group_size])

    torch.cuda.empty_cache()
    return ret, B, alpha, beta, (wf != 0.0)


def greedy_mean_torch_asym(w, n_bits=1, wf=None):
    device = w.device
    d1, d2 = w.shape
    B = torch.zeros((d1, d2, n_bits), device=device)
    Alpha = torch.zeros((d1, n_bits), device=device)

    if wf is not None:
        sum_wf = torch.clamp(wf.sum(dim=1, keepdim=True), min=1e-12)
        beta = (w * wf).sum(dim=1, keepdim=True) / sum_wf
    else:
        beta = w.mean(dim=1, keepdim=True)

    r = (w - beta).clone()
    w_hat_no_bias = torch.zeros_like(w)

    for i in range(n_bits):
        b = r.sign()
        if wf is not None:
            denom = torch.clamp(wf.sum(dim=1, keepdim=True), min=1e-12)
            alpha = (r.abs() * wf).sum(dim=1, keepdim=True) / denom
        else:
            alpha = r.abs().mean(dim=1, keepdim=True)

        r -= b * alpha
        w_hat_no_bias += b * alpha
        B[:, :, i] = b
        Alpha[:, i] = alpha.view(-1)

    torch.cuda.empty_cache()
    return w_hat_no_bias, B, Alpha, beta.view(-1)


def refine_mean_torch_asym(w, w_hat_no_bias, B, Alpha, beta, wf=None, use_bst=True):
    w = w.float()
    d1, d2 = w.shape
    n_bits = B.shape[-1]

    with torch.no_grad():
        Bt = B.transpose(1, 2)
        ones = torch.ones((d1, d2), device=w.device, dtype=w.dtype)

        if wf is not None:
            W_B = Bt * wf.unsqueeze(1)
            W_1 = wf * ones
        else:
            W_B = Bt
            W_1 = ones

        B_cov = W_B.bmm(B)  # [d1,k,k]
        BtW1 = W_B.bmm(ones.unsqueeze(-1)).view(d1, n_bits)  # [d1,k]
        sumW = W_1.sum(dim=1)  # [d1]
        Btw = W_B.bmm(w.unsqueeze(-1)).view(d1, n_bits)  # [d1,k]
        one_w = (W_1 * w).sum(dim=1)  # [d1]

        A_ext = torch.zeros(
            (d1, n_bits + 1, n_bits + 1), device=w.device, dtype=w.dtype
        )
        A_ext[:, :n_bits, :n_bits] = B_cov
        A_ext[:, :n_bits, -1] = BtW1
        A_ext[:, -1, :n_bits] = BtW1
        A_ext[:, -1, -1] = sumW

        b_ext = torch.zeros((d1, n_bits + 1), device=w.device, dtype=w.dtype)
        b_ext[:, :n_bits] = Btw
        b_ext[:, -1] = one_w

        x0 = torch.zeros_like(b_ext)
        x0[:, :n_bits] = Alpha
        x0[:, -1] = beta
        theta = batch_cg_torch(A_ext, b_ext, x=x0)  # [d1,k+1]
        Alpha_new = theta[:, :n_bits]
        beta_new = theta[:, -1]

        Alpha_new, _ = Alpha_new.abs().sort(descending=True)

        if use_bst is False:
            r = w - beta_new.view(-1, 1)
            B_new = torch.zeros_like(B)
            for i in range(n_bits):
                B_new[:, :, i] = r.sign()
                B_new_alpha = Alpha_new[:, i].view([-1, 1])
                r -= B_new[:, :, i] * B_new_alpha
            del r
        else:
            B_new = find_B_torch(w - beta_new.view(-1, 1), Alpha_new)
            if wf is not None:
                B_new = B_new * (wf != 0.0).unsqueeze(-1)

        w_hat_no_bias_new = torch.einsum("ijl,il->ij", (B_new, Alpha_new))

    return w_hat_no_bias_new, B_new, Alpha_new, beta_new


def greedy_mean_torch(w, n_bits=1, wf=None):
    B = torch.zeros(w.shape + (n_bits,), device=w.device)
    Alpha = torch.zeros(w.shape[0], n_bits, device=w.device)
    r, w_hat = w.clone(), 0.0
    for i in range(n_bits):
        b = r.sign()
        if wf is not None:
            denom = torch.sum(wf, dim=1)
            alpha = (r.abs() * wf).sum(dim=1) / torch.clamp(denom, min=1e-12)
            alpha[torch.isnan(alpha)] = 0.0
            alpha = alpha.view(alpha.shape[0], 1)
        else:
            alpha = r.abs().mean(dim=1, keepdim=True)
        r -= b * alpha
        w_hat += b * alpha
        B[:, :, i] = b
        Alpha[:, i] = alpha.view(-1)
    del r, b, alpha
    torch.cuda.empty_cache()
    return w_hat, B, Alpha


def refine_mean_torch(w, w_hat, B, Alpha, wf=None, use_bst=True):
    w = w.float()
    d1, d2 = w.shape
    with torch.no_grad():
        n_bits = B.shape[-1]
        Bt = B.transpose(1, 2)
        if wf is not None:
            Bt = Bt * wf.unsqueeze(1)
        B_cov = Bt.bmm(B)
        Btw = Bt.bmm(w.unsqueeze(-1)).view(d1, n_bits)

        Alpha_new = batch_cg_torch(B_cov, Btw, x=Alpha)
        Alpha_new, _ = Alpha_new.abs().sort(descending=True)

        if use_bst is False:
            r = w.clone()
            B_new = torch.zeros_like(B)
            for i in range(n_bits):
                B_new[:, :, i] = r.sign()
                r -= B_new[:, :, i] * Alpha_new[:, i].view([-1, 1])
            del r
        else:
            B_new = find_B_torch(w, Alpha_new)
            B_new = B_new * (wf != 0.0).unsqueeze(-1) if wf is not None else B_new
        w_hat_new = torch.einsum("ijl,il->ij", (B_new, Alpha_new))
    return w_hat_new, B_new, Alpha_new


def list_binary_vecs(n):
    ListBinaryVecs = {0: [[]]}
    for m in range(1, n + 1):
        ListBinaryVecs[m] = [[1.0] + l for l in ListBinaryVecs[m - 1]] + [
            [-1.0] + l for l in ListBinaryVecs[m - 1]
        ]
    return ListBinaryVecs


def find_B_torch(w, Alpha):
    n_bits = Alpha.shape[-1]
    ListBinaryVecs = list_binary_vecs(n_bits)
    bin_mat = torch.from_numpy(np.vstack(ListBinaryVecs[n_bits]).astype(np.float32)).to(
        w.device
    )

    d1, d2 = w.shape
    row_inds = (
        torch.arange(d1, dtype=torch.long, device=w.device)
        .view(d1, 1)
        .repeat([1, d2])
        .view(-1)
    )
    v = Alpha.mm(bin_mat.t())
    v_sorted, inds = torch.sort(v)
    w_flat = w.view([-1])
    Left = torch.zeros(d1 * d2, dtype=torch.long, device=w.device)
    Right = torch.ones(d1 * d2, dtype=torch.long, device=w.device) * (2**n_bits - 1)
    for _ in range(n_bits):
        Mid_Left = torch.div(Left + Right - 1, 2, rounding_mode="trunc")
        Mid_Right = Mid_Left + 1
        mid_vals = (v_sorted[row_inds, Mid_Left] + v_sorted[row_inds, Mid_Right]) / 2
        inds_left = w_flat < mid_vals
        Right[inds_left] = Mid_Left[inds_left]
        Left[~inds_left] = Mid_Right[~inds_left]
    assignment_inds = inds[row_inds, Left].view(d1, d2)
    return bin_mat[assignment_inds, :]


def batch_cg_torch(A, b, x=None):
    d1, k, _ = A.shape
    x = x.clone().view(d1, k, 1)
    b = b.view(d1, k, 1)
    r = b - A.bmm(x)
    rtr_new = r.transpose(1, 2).bmm(r)
    p = r.clone()
    for _ in range(k):
        rtr = rtr_new
        Ap = A.bmm(p)
        alpha = rtr / (p.transpose(1, 2).bmm(Ap) + 1e-6)
        x += alpha * p
        r -= alpha * Ap
        rtr_new = r.transpose(1, 2).bmm(r)
        beta = rtr_new / (rtr + 1e-6)
        p = r + beta * p
    return x.view(d1, k)
