import os
import json
import argparse
from pathlib import Path
import torch
import torch.distributed as dist

# ---------------- ENVIRONMENT SETUP ----------------
# Use cluster-safe dataset cache
os.environ.setdefault(
    "HF_DATASETS_CACHE", os.path.expandvars("$WORK/../.cache/datasets")
)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from datasets import load_from_disk
import os
from anybcq.evaluate.helpers import utils
from anybcq.evaluate import eval

print("""This script will evaluate all models in the cache directory by:
    1. Calculating perplexity on specified datasets, and
    2. Evaluating downstream tasks using lm_eval on specified tasks.
""")

# ---------------- ARGUMENTS ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--output_file", type=str, default="results.json")
parser.add_argument("--redo", action="store_true")
parser.add_argument("--cache_dir", type=str, default="./cache")
parser.add_argument("--downstream", action="store_true")
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--offline", action="store_true", help="Force offline mode")
parser.add_argument("--chunk_size", type=int, default=4)
parser.add_argument("--dist_infer", action="store_true", help="Enable distributed stage-partitioned inference for perplexity eval.")
parser.add_argument("--num_stages", type=int, default=1, help="Number of distributed stages (must match WORLD_SIZE).")
parser.add_argument("--stage_rank", type=int, default=-1, help="Distributed stage rank. -1 reads rank from env.")
parser.add_argument("--dist_backend", type=str, default="nccl", help="torch.distributed backend.")
parser.add_argument("--dist_init_method", type=str, default="env://", help="torch.distributed init method.")
parser.add_argument("--skip_lm_eval", action="store_true", help="Skip lm-eval task evaluation.")
args = parser.parse_args()

if args.offline:
    os.environ["HF_DATASETS_OFFLINE"] = "1"

dist_state = None
if args.dist_infer:
    if args.num_stages < 1:
        raise ValueError("--num_stages must be >= 1")
    dist_state = eval.init_distributed_inference(
        num_stages=args.num_stages,
        stage_rank=args.stage_rank,
        dist_backend=args.dist_backend,
        dist_init_method=args.dist_init_method,
    )


def is_rank0():
    return dist_state is None or dist_state.stage_rank == 0

# ---------------- DATASET HELPERS ----------------


from datasets import Dataset


def get_wikitext2():
    import pyarrow as pa
    import pyarrow.feather as feather
    import pyarrow.dataset as ds

    dataset_dir = os.path.join(
        os.environ["HF_DATASETS_CACHE"],
        "Salesforce___wikitext",
        "wikitext-2-raw-v1",
        "0.0.0",
        "b08601e04326c79dfdd32d625aee71d232d685c3",
    )

    test_file = os.path.join(dataset_dir, "wikitext-test.arrow")
    train_file = os.path.join(dataset_dir, "wikitext-train.arrow")
    val_file = os.path.join(dataset_dir, "wikitext-validation.arrow")

    # Load the test split as a HuggingFace Dataset
    ds_test = Dataset.from_file(test_file)
    return ds_test["text"]  # This is exactly what your script wants


def get_c4():
    dataset_dir = os.path.join(
        os.environ["HF_DATASETS_CACHE"],
        "allenai___c4",
        "default-b04fc8a0b8562884",
        "0.0.0",
        "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
    )

    val_file = os.path.join(dataset_dir, "c4-train-00000-of-00002.arrow")
    ds_val = Dataset.from_file(val_file)
    return ds_val["text"][:10000]


# Patch dataloader dynamically
import anybcq.evaluate.helpers.dataloader as dataloader

dataloader.get_wikitext2 = get_wikitext2
dataloader.get_c4_new = get_c4

# ---------------- TASK SETUP ----------------
datasets = ["wikitext2", "c4_new"]

if args.downstream:
    tasks = ["winogrande", "piqa", "arc_easy", "arc_challenge", "hellaswag"]
    num_fewshot = 0
else:
    tasks = ["mmlu"]
    num_fewshot = 5

# ---------------- LOAD PREVIOUS RESULTS ----------------
if is_rank0() and os.path.exists(args.output_file):
    with open(args.output_file) as f:
        all_results = json.load(f)
else:
    all_results = {}


def save_results(results_dict):
    with open(args.output_file, "w") as f:
        json.dump(results_dict, f, indent=2)


# ---------------- RUN ----------------
print("==================================================")
print(f"Model: {args.model_path}")
if dist_state is not None:
    print(
        f"Distributed mode: rank {dist_state.stage_rank}/{dist_state.world_size} "
        f"(local_rank={dist_state.local_rank})"
    )
print("==================================================")

tokenizer_type, tokenizer, model = eval.auto_model_load(
    args.model_path,
    args.fp16,
    dist_state=dist_state,
)

# ---- PERPLEXITY EVAL ----
ppl_results = {}
if is_rank0():
    print("\n[INFO] Running perplexity eval...")
if dist_state is None:
    ppl_results = eval.evaluate_ppl(
        model,
        tokenizer,
        datasets,
        verbose=True,
        chunk_size=args.chunk_size,
        tokenizer_type=tokenizer_type,
    )
else:
    ppl_results = eval.evaluate_ppl_distributed(
        model,
        tokenizer,
        datasets,
        dist_state=dist_state,
        verbose=True,
        chunk_size=args.chunk_size,
        tokenizer_type=tokenizer_type,
    )

if is_rank0():
    all_results.setdefault(args.model_path, {}).setdefault("ppl", {}).update(ppl_results)
    save_results(all_results)

# ---- LM EVAL ----
if args.dist_infer or args.skip_lm_eval:
    if is_rank0():
        reason = "--dist_infer enabled" if args.dist_infer else "--skip_lm_eval enabled"
        print(f"\n[INFO] Skipping lm-eval ({reason}).")
else:
    if is_rank0():
        print("\n[INFO] Running lm-eval...")
    lm_eval_results = eval.run_lm_eval(tokenizer, model, tasks, num_fewshot)
    if is_rank0():
        all_results.setdefault(args.model_path, {}).setdefault("lm-eval", {}).update(
            lm_eval_results
        )
        save_results(all_results)

if is_rank0():
    print("\n================ RESULTS ================")
    print(json.dumps(all_results, indent=2))

del model
if dist_state is not None and dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()
