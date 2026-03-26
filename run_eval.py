import os
import json
import argparse
from pathlib import Path

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
args = parser.parse_args()

if args.offline:
    os.environ["HF_DATASETS_OFFLINE"] = "1"

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
if os.path.exists(args.output_file):
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
print("==================================================")

tokenizer_type, tokenizer, model = eval.auto_model_load(args.model_path, args.fp16)

# ---- PERPLEXITY EVAL ----
ppl_results = {}
print("\n[INFO] Running perplexity eval...")
ppl_results = eval.evaluate_ppl(
    model,
    tokenizer,
    datasets,
    verbose=True,
    chunk_size=4,
    tokenizer_type=tokenizer_type,
)

all_results.setdefault(args.model_path, {}).setdefault("ppl", {}).update(ppl_results)
save_results(all_results)

# ---- LM EVAL ----
print("\n[INFO] Running lm-eval...")
lm_eval_results = eval.run_lm_eval(tokenizer, model, tasks, num_fewshot)

all_results.setdefault(args.model_path, {}).setdefault("lm-eval", {}).update(
    lm_eval_results
)
save_results(all_results)

print("\n================ RESULTS ================")
print(json.dumps(all_results, indent=2))

del model
