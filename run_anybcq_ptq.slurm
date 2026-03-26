#!/bin/bash

#SBATCH -J anybcq_ptq
#SBATCH -N 8 
#SBATCH --ntasks-per-node=1
#SBATCH -p gh
#SBATCH -t 10:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err 

set -euo pipefail

set +u
module purge
module load tacc-apptainer
set -u


export MODEL_PATH=$WORK/Llama-3.1-70B


MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-29500}
export DIST_OUTPUT_ROOT=${DIST_OUTPUT_ROOT:-$WORK/anybcq_dist}
export DIST_LOG_ROOT=${DIST_LOG_ROOT:-$WORK/anybcq_dist_logs}
export MERGED_OUTPUT_DIR=${MERGED_OUTPUT_DIR:-$DIST_OUTPUT_ROOT/merged_2to8}

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "DIST_OUTPUT_ROOT=$DIST_OUTPUT_ROOT"
echo "MERGED_OUTPUT_DIR=$MERGED_OUTPUT_DIR"

# ===== Run =====
cd $WORK
srun --ntasks=$SLURM_NTASKS \
     --ntasks-per-node=1 \
     --export=ALL \
     apptainer exec --nv ./pytorch_25.04-py3.sif bash -lc '

set -euo pipefail

cd $WORK/multi-precision-spec-decode/anybcq

export WORLD_SIZE=${SLURM_NTASKS}
export RANK=${SLURM_PROCID}
export LOCAL_RANK=${SLURM_LOCALID}

# ===== Cache setup =====
export CACHE_ROOT=$WORK/hf_cache
mkdir -p $CACHE_ROOT/{datasets,hub,transformers}
mkdir -p $DIST_OUTPUT_ROOT $DIST_LOG_ROOT

export HF_HOME=$CACHE_ROOT
export HF_DATASETS_CACHE=$CACHE_ROOT/datasets
export HUGGINGFACE_HUB_CACHE=$CACHE_ROOT/hub
export TRANSFORMERS_CACHE=$CACHE_ROOT/transformers

# ===== Debugging =====
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1

echo "RANK=$RANK / WORLD_SIZE=$WORLD_SIZE on $(hostname)"

# ===== Run training =====
python run_clm.py \
  --model_name_or_path ${MODEL_PATH} \
  --dataset_name c4 \
  --do_eval \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --w_lr1 1e-4 --w_lr2 1e-4 --w_lr3 1e-4 \
  --n_bits_w 2 \
  --add_bits 6 \
  --group_size 128 \
  --num_samples 512 \
  --iters_w 5000 \
  --input_prob 0.5 \
  --asymmetric False \
  --train_beta False \
  --dist_ptq True \
  --num_stages ${WORLD_SIZE} \
  --stage_rank ${RANK} \
  --microbatch_size 1 \
  --skip_eval True \
  --output_dir $DIST_OUTPUT_ROOT/stage_${RANK} \
  --cache_dir $WORK/hf_cache \
  --dist_debug True \
  2>&1 | tee $DIST_LOG_ROOT/stage_${RANK}.log
'

echo "Distributed stage runs completed; merging shards into $MERGED_OUTPUT_DIR"
apptainer exec --nv ./pytorch_25.04-py3.sif bash -lc '
set -euo pipefail
cd $WORK/multi-precision-spec-decode/anybcq

export WORLD_SIZE=${SLURM_NTASKS:-1}

echo "Merging with WORLD_SIZE=$WORLD_SIZE"

merge_args=()
for ((r=0; r<WORLD_SIZE; r++)); do
  shard_dir="$DIST_OUTPUT_ROOT/stage_${r}"
  echo "Adding shard: $shard_dir"
  merge_args+=(--shard_dir "$shard_dir")
done

echo "Final args: ${merge_args[@]}"

python merge_anybcq_chunks.py \
  "${merge_args[@]}" \
  --output_dir "$MERGED_OUTPUT_DIR"
'

echo "Merge complete: $MERGED_OUTPUT_DIR"
