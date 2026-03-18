#!/bin/bash
#SBATCH --job-name=grlnet_a100_50e
#SBATCH -p gpu_A100
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH -t 7-00:00:00
#SBATCH -o /home/faenna/grl/slurm-%j.out
#SBATCH -D /home/faenna/grl/GRLNet

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/faenna/grl/GRLNet}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/home/faenna/grl/torch/bin/activate}"
TRAIN_ROOT="${TRAIN_ROOT:-/home/faenna/grl/image-net1000/layout/train}"
EVAL_ROOT="${EVAL_ROOT:-/home/faenna/grl/image-net1000/layout/val}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/faenna/grl/runs/grlnet_a100_single_50e_auxh}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/recipes/imagenet/configs/grl_a100_single_50e.yaml}"

cd "$REPO_DIR"
source "$VENV_ACTIVATE"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

nvidia-smi || true
nvidia-smi -L || true

python -u recipes/imagenet/train.py \
  --config "$CONFIG_PATH" \
  --train-root "$TRAIN_ROOT" \
  --eval-root "$EVAL_ROOT" \
  --output-dir "$OUTPUT_DIR"
