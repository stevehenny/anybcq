# AnyBCQ
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import torch
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

from .loss import LossFunction
from .cached_loader import CachedDataset, DataCacheWrapper
from anybcq.utils.swap_linear import (
    swap_quant_model,
    add_onebit_model,
    delete_original_weight,
    replace_bcq_with_linear,
    replace_bcq_with_lutgemm,
    save_alpha_and_beta_in_bcqlinear,
)

from .bcq_linear import BCQLinear
import gc


class AnyBCQ:
    def __init__(
        self,
        model: torch.nn.Module,
        data_loader: Dataset,
        wq_params: dict = None,
        aq_params: dict = None,
        batch_size: int = 16,
        iters: int = 10000,
        add_bits: int = 0,
        input_prob: float = 0.5,
        num_samples: int = 1024,
        w_lr1: float = 4e-5,
        w_lr2: float = 4e-5,
        w_lr3: float = 4e-5,
        a_lr: float = 4e-5,
        torch_dtype=torch.float32,
        recon_dtype=torch.float32,
        asymmetric: bool = True,
        train_beta: bool = True,
    ):
        self.wq_params = wq_params
        self.aq_params = aq_params
        self.model = model.eval()
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.iters = iters
        self.add_bits = add_bits
        self.input_prob = input_prob
        self.num_samples = num_samples
        self.w_lr1 = w_lr1
        self.w_lr2 = w_lr2
        self.w_lr3 = w_lr3
        self.a_lr = a_lr
        self.torch_dtype = torch_dtype
        self.recon_dtype = recon_dtype
        self.asymmetric = asymmetric
        self.train_beta = train_beta
        self.use_weight_quant = True if wq_params["n_bits"] < 16 else False
        self.use_act_quant = True if aq_params["n_bits"] < 16 else False

        self.attention_mask = None
        self.position_ids = None
        self.qlayer_list = ["INTLinear"]
        self.base_bit = wq_params["n_bits"]

    def minimize(self, block_list_class=torch.nn.ModuleList):
        print(self.model.device)
        print(self.model)
        # Find decoder layers
        for _, module in self.model.named_modules():
            if isinstance(module, block_list_class):
                block_units = module
                break

        block_units[0] = DataCacheWrapper(block_units[0])
        cached_data = CachedDataset(
            self.model,
            self.data_loader,
            self.input_prob,
            self.num_samples,
            self.base_bit,
            self.add_bits,
            block_list_class,
        )
        block_units[0] = block_units[0].module

        for idx in tqdm(range(len(block_units))):
            print("=" * 60)
            print(f"    Layer {idx} Optimization Start")
            print("=" * 60)
            print(f"1. Full-precision model Activation Caching")

            if idx > 0:
                fp_block = block_units[idx]
                wrapped_block = DataCacheWrapper(fp_block)
                cached_data.fp_data_caching(wrapped_block)

            print(f"\n2. Make independent Block")
            recon_block = block_units[idx].to("cuda")
            print(recon_block)

            print(f"\n3. Quantize independent Block")
            swap_quant_model(
                recon_block,
                n_bits=self.wq_params["n_bits"],
                n_rounds=20,
                group_size=self.wq_params["group_size"],
                packing=False,
                asymmetric=self.asymmetric,
                swap_type="bcq",
            )

            for i in range(self.add_bits + 1):
                if i == 0:
                    cached_data.set_precision(self.wq_params["n_bits"])
                else:
                    cached_data.set_precision(self.wq_params["n_bits"] + self.add_bits)
                cached_dataloader = DataLoader(
                    cached_data,
                    shuffle=True,
                    batch_size=1,
                )
                if i > 0:
                    print(
                        f"\n4-{i}. Add One Bit \
                        (precision={self.wq_params['n_bits']+i})"
                    )
                    add_onebit_model(
                        recon_block,
                        n_rounds=20,
                        group_size=self.wq_params["group_size"],
                        packing=False,
                        swap_type="bcq",
                    )
                recon_block = self.type_cast(recon_block, self.recon_dtype)
                print(f"\n4-{i}. Again Block Minimization Reconstruction Error")
                recon_block = self.blockReconstruction(
                    recon_block,
                    cached_dataloader,
                    add_bit=i,
                    layer_idx=idx,
                    num_layers=len(block_units),
                )
                if i == 0 or i == self.add_bits:
                    print(f"\n4-{i}. Quantized model Activation caching")
                    wrapped_block = DataCacheWrapper(recon_block)
                    cached_data.q_data_caching(wrapped_block)

            print(f"\n5. Delete original weight")
            recon_block = self.type_cast(recon_block, self.torch_dtype)

            replace_bcq_with_lutgemm(recon_block)
            recon_block = self.type_cast(recon_block, self.torch_dtype)
            recon_block.to("cpu")

            block_units[idx] = recon_block
            print(block_units[idx])

            del recon_block

            torch.cuda.empty_cache()
            print("-" * 60)
            del cached_dataloader
            del wrapped_block
            torch.cuda.empty_cache()

        del cached_data
        torch.cuda.empty_cache()

    def type_cast(self, block, cast_type):
        if block is None:
            return None

        if cast_type == torch.float32:
            block = block.float()
        elif cast_type == torch.float16:
            block = block.half()
        elif cast_type == torch.bfloat16:
            block = block.bfloat16()
        else:
            print(cast_type)
            raise ValueError("Invalid Type cast")

        return block

    def blockReconstruction(
        self,
        recon_block: nn.Module,
        cached_dataloader,
        add_bit,
        layer_idx=None,
        num_layers=None,
    ):
        w_para = []
        a_para = []

        if add_bit == 0:
            lr = self.w_lr1
        elif add_bit == 1:
            lr = self.w_lr2
        else:
            lr = self.w_lr3

        all_params = []
        for name, module in recon_block.named_modules():
            if isinstance(module, BCQLinear):
                num_bit = module.num_bit
                for idx in range(module.num_bit):
                    alpha_param = getattr(module, "alpha" + str(idx))
                    if idx == module.num_bit - 1:
                        alpha_lr = lr * 1.0
                    else:
                        # alpha_lr = lr * 0.1
                        alpha_lr = lr * 1.0

                    all_params.append({"params": [alpha_param], "lr": alpha_lr})

                if self.train_beta:
                    all_params.append({"params": [module.beta], "lr": lr})

        optimizer = torch.optim.Adam(all_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.iters, eta_min=0.0
        )

        print("*****************************************")
        print(f"Weight_Quant({num_bit} bits, +{add_bit}-bit)")
        print(f"w_lr: ", lr)
        print(f"Training iters: {self.iters}")
        print("*****************************************")

        loss_func = LossFunction(
            recon_block,
            max_count=self.iters,
        )

        epochs = int(self.iters / len(cached_dataloader))
        remainder = self.iters - len(cached_dataloader) * epochs

        for name, param in recon_block.named_parameters():
            param.requires_grad = False
        for module in recon_block.modules():
            if isinstance(module, BCQLinear):
                for idx in range(module.num_bit):
                    getattr(module, f"alpha{idx}").requires_grad = True
                if self.train_beta:
                    module.beta.requires_grad = True
        for name, param in recon_block.named_parameters():
            if param.requires_grad:
                print(f"{name} is trainable!")
        optimizer.zero_grad()
        for epoch in tqdm(range(epochs + 1)):
            for step, batch in enumerate(cached_dataloader):
                if epoch == epochs and step == remainder:
                    break

                # Make input, output for reconstruction error
                input_q = batch["q_input"].squeeze(0)
                input_fp = batch["fp_input"].squeeze(0)
                output_fp = batch["fp_output"].squeeze(0)
                self.attention_mask = batch.get("attention_mask", None)
                self.position_ids = batch.get("position_ids", None)
                cache_position = batch.get("cache_position", None)
                if self.attention_mask is not None:
                    self.attention_mask = self.attention_mask.squeeze(0)
                if self.position_ids is not None:
                    self.position_ids = self.position_ids.squeeze(0)
                    cache_position = cache_position.squeeze(0)

                # Optional input mixing
                if self.input_prob < 1.0:
                    input_q = torch.where(
                        torch.rand_like(input_q, dtype=input_q.dtype) < self.input_prob,
                        input_q,
                        input_fp,
                    )

                # Cast for fp32 MSE
                input_q = self.type_cast(input_q, self.recon_dtype)
                self.attention_mask = self.type_cast(
                    self.attention_mask, self.recon_dtype
                )

                # ───── NEW: build rotary embeddings if a position_ids tensor is present ──

                # ❶ Choose the RoPE module.
                if hasattr(self.model.model, "rotary_emb"):  # Qwen3 / Qwen2
                    rope = self.model.model.rotary_emb
                elif hasattr(recon_block.self_attn, "rotary_emb"):  # Llama‑style layers
                    rope = recon_block.self_attn.rotary_emb
                else:
                    raise RuntimeError("No rotary_emb found; add a special‑case here.")

                position_embeddings = None
                if self.position_ids is not None:
                    # `rotary_emb` lives inside the *attention* sub‑module
                    cos, sin = rope(
                        input_q, self.position_ids
                    )  # same call Qwen3Model uses
                    position_embeddings = (cos, sin)

                # ───── Forward pass through the isolated block ──────────────────────────
                if self.attention_mask is not None:  # language task
                    output_q = recon_block(
                        hidden_states=input_q,
                        attention_mask=self.attention_mask,
                        position_ids=self.position_ids,
                        position_embeddings=position_embeddings,  # ← NEW
                        cache_position=cache_position,
                    )
                else:  # e.g. vision transformer
                    output_q = recon_block(input_q)

                loss = loss_func(output_q, output_fp)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(w_para, max_norm=0.1)
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()

        del optimizer
        del loss
        torch.cuda.empty_cache()

        save_alpha_and_beta_in_bcqlinear(recon_block)
        print("=" * 30)

        return recon_block
