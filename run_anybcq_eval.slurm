#!/bin/bash

#SBATCH -J anybcq_eval
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=gpu:1
#SBATCH -t 08:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

set +u
module purge
module load tacc-apptainer
set -u

export MODEL_PATH=$WORK/Llama-3.1-8B-anybcq2to4-g128

cd $WORK

srun --ntasks=$SLURM_NTASKS \
     --ntasks-per-node=1 \
     --export=ALL \
     apptainer exec --nv ./pytorch_25.04-py3.sif bash -lc '

set -euo pipefail

cd $WORK/multi-precision-spec-decode/anybcq

export HF_DATASETS_CACHE=$WORK/../.cache/datasets

# run eval 
python run_eval.py --model_path $MODEL_PATH

'
