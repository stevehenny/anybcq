# AnyBCQ: Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs (ICLR 2026)

This repository provides an official implementation of **AnyBCQ**.

## 🔧 Installation

- NGC image used: `nvcr.io/nvidia/pytorch:24.10-py3` ([details / release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-10.html))

Install the Python package in editable mode:

```bash
pip install -e .
```

To install the custom CUDA kernel for LUT-based GEMV used in AnyBCQ:

```bash
cd anybcq/inference/custom_kernel
source do_install.sh
```

This will compile and install the required kernel for fast matrix-vector operations with arbitrary-bit quantized weights.

## Quick Evaluation

Get started quickly by downloading a pre-quantized model and running evaluation.

### Download checkpoint

Available checkpoints:
- [Llama-3.1-8B-anybcq2to4-g128](https://huggingface.co/gunho1123/Llama-3.1-8B-anybcq2to4-g128)
- [gemma-2-9b-anybcq2to4-g128](https://huggingface.co/gunho1123/gemma-2-9b-anybcq2to4-g128)
- [phi-4-anybcq2to4-g128](https://huggingface.co/gunho1123/phi-4-anybcq2to4-g128)
- [Qwen3-32B-anybcq2to4-g128](https://huggingface.co/gunho1123/Qwen3-32B-anybcq2to4-g128)

```bash
REPO_ID=gunho1123/Llama-3.1-8B-anybcq2to4-g128

python download_model.py --repo_id $REPO_ID --local_dir $REPO_ID
```

### Evaluation
```bash
REPO_ID=gunho1123/Llama-3.1-8B-anybcq2to4-g128

# MMLU
python run_eval.py --model_path $REPO_ID

# CSR
python run_eval.py --model_path $REPO_ID --downstream
```


## 🚀 PTQ (post-training quantization)

Run the following from the repository root (copy–paste as is):

```bash
# (Optional) choose GPU
export GPU=0

MODEL_PATH=meta-llama/Llama-3.1-8B
SAVE_PATH=llama3

# PTQ on Llama-3.1-8B with AnyBCQ
CUDA_VISIBLE_DEVICES=${GPU} python run_clm.py \
  --model_name_or_path ${MODEL_PATH} \
  --dataset_name c4 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 8 \
  --do_eval \
  --w_lr1 1e-4 --w_lr2 1e-4 --w_lr3 1e-4 \
  --n_bits_w 2 \
  --add_bits 2 \
  --group_size 128 \
  --num_samples 512 \
  --iters_w 5000 \
  --input_prob 0.5 \
  --asymmetric False \
  --train_beta False \
  --output_dir llama3 \
  --cache_dir cache_dir
```

Tip: We found `--input_prob 0.5` works well for LLaMA- and Gemma-family models, while `--input_prob 1.0` tends to work better for Qwen- and Phi-family models.


## ✅ Evaluation

Run the following from the repository root (copy–paste as is):

```bash
# (Optional) choose GPU
export GPU=0

# Path to the PTQ output (from the PTQ step above)
SAVE_PATH=llama3

# 1) MMLU evaluation
CUDA_VISIBLE_DEVICES=${GPU} python run_eval.py \
  --model_path "${SAVE_PATH}"

# 2) Common-sense reasoning (CSR) suite
CUDA_VISIBLE_DEVICES=${GPU} python run_eval.py \
  --model_path "${SAVE_PATH}" \
  --downstream
```

## ⚡ Throughput Evaluation

Run the following from the repository root (copy–paste as is).  
This measures token throughput (`tokens/s`) using the inference script.

```bash
# Enter the inference folder
cd anybcq/inference

# (Optional) choose GPU
export GPU=0

model_name_or_path="meta-llama/Llama-3.1-8B"

# ---------------------------
# AnyBCQ backend @ 2-bit
# ---------------------------
backend="anybcq"
bitwidth=2

CUDA_VISIBLE_DEVICES=${GPU} python generate.py --compile 2 --num_samples 5 \
  --model_name "${model_name_or_path}" --bitwidth ${bitwidth} --dtype "float16" \
  --backend ${backend} --max_new_tokens 100 --random_init

# ---------------------------
# AP backend @ 2-bit
# ---------------------------
backend="ap"
bitwidth=2

CUDA_VISIBLE_DEVICES=${GPU} python generate.py --compile 2 --num_samples 5 \
  --model_name "${model_name_or_path}" --bitwidth ${bitwidth} --dtype "float16" \
  --backend ${backend} --max_new_tokens 100 --random_init

# ---------------------------
# FP16 baseline (no backend arg)
# ---------------------------
bitwidth=16

CUDA_VISIBLE_DEVICES=${GPU} python generate.py --compile 2 --num_samples 5 \
  --model_name "${model_name_or_path}" --bitwidth ${bitwidth} --dtype "float16" \
  --max_new_tokens 100 --random_init
```

## 📜 Citation

If you find **AnyBCQ** useful, please cite:

```bibtex
@article{park2025anybcq,
  title={AnyBCQ: Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs},
  author={Park, Gunho and Bae, Jeongin and Kwon, Beomseok and Kim, Byeongwook and Kwon, Se Jung and Lee, Dongsoo},
  journal={arXiv preprint arXiv:2510.10467},
  year={2025}
}
```

## License

```
AnyBCQ
Copyright (c) 2025-present NAVER Cloud Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
