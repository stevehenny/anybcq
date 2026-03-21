#!/usr/bin/env python3
import argparse
import fnmatch
import gc
import json
import os
import shutil
from pathlib import Path

import torch

try:
    from safetensors.torch import load_file as load_safetensors_file
    from safetensors.torch import save_file as save_safetensors_file
except ImportError:
    load_safetensors_file = None
    save_safetensors_file = None

from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)

try:
    from transformers.modeling_utils import shard_checkpoint
except ImportError:
    shard_checkpoint = None


WEIGHT_FILE_PATTERNS = (
    SAFE_WEIGHTS_NAME,
    "model-*.safetensors",
    SAFE_WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    "pytorch_model-*.bin",
    WEIGHTS_INDEX_NAME,
)


def is_weight_file(filename: str) -> bool:
    return any(fnmatch.fnmatch(filename, pattern) for pattern in WEIGHT_FILE_PATTERNS)


def load_single_weight_file(path: Path):
    if path.suffix == ".safetensors":
        if load_safetensors_file is None:
            raise ImportError(
                "safetensors is required to load .safetensors checkpoints."
            )
        return load_safetensors_file(str(path), device="cpu")
    return torch.load(str(path), map_location="cpu")


def load_checkpoint_state_dict(checkpoint_dir: Path):
    single_safe = checkpoint_dir / SAFE_WEIGHTS_NAME
    single_pt = checkpoint_dir / WEIGHTS_NAME
    safe_index = checkpoint_dir / SAFE_WEIGHTS_INDEX_NAME
    pt_index = checkpoint_dir / WEIGHTS_INDEX_NAME

    if single_safe.exists():
        return load_single_weight_file(single_safe)
    if single_pt.exists():
        return load_single_weight_file(single_pt)

    if safe_index.exists():
        if load_safetensors_file is None:
            raise ImportError(
                "safetensors is required to load sharded .safetensors checkpoints."
            )
        with safe_index.open("r", encoding="utf-8") as f:
            index_data = json.load(f)
        shards = sorted(set(index_data["weight_map"].values()))
        state_dict = {}
        for shard_name in shards:
            shard_path = checkpoint_dir / shard_name
            state_dict.update(load_single_weight_file(shard_path))
        return state_dict

    if pt_index.exists():
        with pt_index.open("r", encoding="utf-8") as f:
            index_data = json.load(f)
        shards = sorted(set(index_data["weight_map"].values()))
        state_dict = {}
        for shard_name in shards:
            shard_path = checkpoint_dir / shard_name
            state_dict.update(load_single_weight_file(shard_path))
        return state_dict

    raise FileNotFoundError(
        f"No recognized weight files found in checkpoint dir: {checkpoint_dir}"
    )


def save_checkpoint_state_dict(
    state_dict,
    output_dir: Path,
    safe_serialization: bool,
    max_shard_size: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if safe_serialization and save_safetensors_file is None:
        raise ImportError("safetensors is required when safe_serialization is enabled.")

    weight_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
    index_name = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME

    if shard_checkpoint is not None:
        shards, index = shard_checkpoint(
            state_dict,
            max_shard_size=max_shard_size,
            weights_name=weight_name,
        )
    else:
        shards = {weight_name: state_dict}
        index = None

    for shard_filename, shard_state_dict in shards.items():
        shard_path = output_dir / shard_filename
        if safe_serialization:
            save_safetensors_file(shard_state_dict, str(shard_path), metadata={"format": "pt"})
        else:
            torch.save(shard_state_dict, str(shard_path))

    if index is not None:
        with (output_dir / index_name).open("w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, sort_keys=True)


def copy_non_weight_files(src_dir: Path, dst_dir: Path):
    for entry in src_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.name == "config.json":
            continue
        if is_weight_file(entry.name):
            continue
        shutil.copy2(entry, dst_dir / entry.name)


def load_shard_metadata(shard_dir: Path):
    config_path = shard_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in shard directory: {shard_dir}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "anybcq" not in cfg:
        raise ValueError(f"config.json missing anybcq section: {shard_dir}")
    anybcq_cfg = cfg["anybcq"]
    chunk = anybcq_cfg.get("chunk")
    if chunk is None:
        raise ValueError(
            f"config.anybcq.chunk missing in {shard_dir}. Re-run run_clm.py with chunk args."
        )
    return cfg, anybcq_cfg, chunk


def validate_and_collect_shards(shard_dirs, require_full_coverage: bool):
    shard_records = []
    base_cfg = None
    base_anybcq = None
    num_layers = None

    for shard_dir in shard_dirs:
        cfg, anybcq_cfg, chunk = load_shard_metadata(shard_dir)

        if base_cfg is None:
            base_cfg = cfg
            base_anybcq = anybcq_cfg
            num_layers = int(chunk.get("num_layers", cfg.get("num_hidden_layers", -1)))
            if num_layers <= 0:
                raise ValueError(
                    f"Unable to infer num_layers from config/chunk metadata in {shard_dir}"
                )
        else:
            if cfg.get("architectures") != base_cfg.get("architectures"):
                raise ValueError("Shard architecture mismatch.")
            for key in ("seed_precision", "parent_precision", "group_size", "arch_config"):
                if anybcq_cfg.get(key) != base_anybcq.get(key):
                    raise ValueError(f"Shard anybcq config mismatch for key '{key}'.")

        start = int(chunk.get("layer_start", chunk.get("local_layer_start", -1)))
        end = int(chunk.get("layer_end", chunk.get("local_layer_end", -1)))
        if start < 0 or end <= start or end > num_layers:
            raise ValueError(
                f"Invalid chunk range [{start}, {end}) in {shard_dir} for num_layers={num_layers}"
            )
        if chunk.get("dist_ptq", False):
            local_start = int(chunk.get("local_layer_start", -1))
            local_end = int(chunk.get("local_layer_end", -1))
            if local_start < 0 or local_end <= local_start:
                raise ValueError(
                    f"Invalid local chunk range [{local_start}, {local_end}) in distributed shard {shard_dir}."
                )
        shard_records.append(
            {
                "dir": shard_dir,
                "layer_start": start,
                "layer_end": end,
                "chunk_index": int(chunk.get("chunk_index", -1)),
                "num_chunks": int(chunk.get("num_chunks", 1)),
                "local_layer_start": int(chunk.get("local_layer_start", 0)),
                "local_layer_end": int(chunk.get("local_layer_end", end - start)),
                "dist_ptq": bool(chunk.get("dist_ptq", False)),
            }
        )

    shard_records.sort(key=lambda r: (r["layer_start"], r["layer_end"]))

    prev_end = -1
    for rec in shard_records:
        if rec["layer_start"] < prev_end:
            raise ValueError("Overlapping chunk ranges detected across shard inputs.")
        prev_end = rec["layer_end"]

    if require_full_coverage:
        expected = 0
        for rec in shard_records:
            if rec["layer_start"] != expected:
                raise ValueError(
                    f"Missing chunk coverage before layer {rec['layer_start']}."
                )
            expected = rec["layer_end"]
        if expected != num_layers:
            raise ValueError(
                f"Chunks do not fully cover layers [0, {num_layers}). Coverage ends at {expected}."
            )

    return shard_records, base_cfg, base_anybcq, num_layers


def build_layer_prefixes(arch_config, layer_start: int, layer_end: int):
    model_name = arch_config["model_name"]
    layers_name = arch_config["layers_name"]
    base = f"{model_name}.{layers_name}."
    return tuple(f"{base}{idx}." for idx in range(layer_start, layer_end))


def merge_chunks(
    shard_records,
    base_dir: Path,
    arch_config,
):
    merged_state_dict = load_checkpoint_state_dict(base_dir)

    for rec in shard_records:
        shard_dir = rec["dir"]
        if shard_dir == base_dir:
            continue

        if rec.get("dist_ptq", False):
            local_prefixes = build_layer_prefixes(
                arch_config, rec["local_layer_start"], rec["local_layer_end"]
            )
            global_prefixes = build_layer_prefixes(
                arch_config, rec["layer_start"], rec["layer_end"]
            )
            prefix_pairs = list(zip(local_prefixes, global_prefixes))
        else:
            prefixes = build_layer_prefixes(
                arch_config, rec["layer_start"], rec["layer_end"]
            )
            prefix_pairs = [(prefix, prefix) for prefix in prefixes]
        shard_state_dict = load_checkpoint_state_dict(shard_dir)

        replaced = 0
        for key, value in shard_state_dict.items():
            for src_prefix, dst_prefix in prefix_pairs:
                if key.startswith(src_prefix):
                    new_key = dst_prefix + key[len(src_prefix) :]
                    merged_state_dict[new_key] = value
                    replaced += 1
                    break
        print(
            f"Merged {replaced} tensors from {shard_dir} for layers [{rec['layer_start']}, {rec['layer_end']})."
        )

        del shard_state_dict
        gc.collect()

    return merged_state_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge AnyBCQ chunked PTQ shard checkpoints into one full checkpoint."
    )
    parser.add_argument(
        "--shard_dir",
        action="append",
        required=True,
        help="Shard checkpoint directory. Repeat this arg for each shard.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for the merged checkpoint.",
    )
    parser.add_argument(
        "--base_shard_dir",
        default=None,
        help="Optional base shard dir to initialize from. Defaults to first shard by layer_start.",
    )
    parser.add_argument(
        "--allow_partial_coverage",
        action="store_true",
        help="Allow merging when shard coverage does not span all layers.",
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        default=True,
        help="Save merged checkpoint in safetensors format (default: enabled).",
    )
    parser.add_argument(
        "--max_shard_size",
        default="10GB",
        help="Maximum shard size for saved checkpoint (e.g., 5GB, 10GB).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    shard_dirs = [Path(d).resolve() for d in args.shard_dir]
    for shard_dir in shard_dirs:
        if not shard_dir.is_dir():
            raise FileNotFoundError(f"Shard directory not found: {shard_dir}")

    shard_records, base_cfg, base_anybcq, num_layers = validate_and_collect_shards(
        shard_dirs, require_full_coverage=not args.allow_partial_coverage
    )

    if args.base_shard_dir is None:
        base_dir = shard_records[0]["dir"]
    else:
        base_dir = Path(args.base_shard_dir).resolve()
        if base_dir not in [r["dir"] for r in shard_records]:
            raise ValueError("--base_shard_dir must be one of the provided --shard_dir values.")

    print(f"Using base shard: {base_dir}")

    merged_state_dict = merge_chunks(
        shard_records=shard_records,
        base_dir=base_dir,
        arch_config=base_anybcq["arch_config"],
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint_state_dict(
        merged_state_dict,
        output_dir=output_dir,
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )

    merged_cfg = dict(base_cfg)
    merged_anybcq_cfg = dict(base_anybcq)
    merged_anybcq_cfg["chunk"] = {
        "layer_start": 0,
        "layer_end": num_layers,
        "num_layers": num_layers,
        "num_quantized_layers": num_layers,
        "chunk_index": -1,
        "num_chunks": len(shard_records),
        "merged": True,
    }
    merged_cfg["anybcq"] = merged_anybcq_cfg
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(merged_cfg, f, indent=2, sort_keys=False)

    copy_non_weight_files(base_dir, output_dir)
    print(f"Merged checkpoint written to: {output_dir}")


if __name__ == "__main__":
    main()
