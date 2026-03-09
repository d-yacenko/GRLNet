#!/bin/bash
#SBATCH --job-name=grlnet_t4_ddp_smoke
#SBATCH -p gpu_T4
#SBATCH --nodelist=hydra-gpu1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH -t 2-00:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/grl/GRLNet}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/grl/torch/bin/activate}"
TRAIN_ROOT="${TRAIN_ROOT:-/data/imagenet/train}"
EVAL_ROOT="${EVAL_ROOT:-/data/imagenet/val}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/grl/runs/grlnet_t4_ddp_smoke}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/recipes/imagenet/configs/grl_t4_ddp_smoke.yaml}"

cd "$REPO_DIR"
source "$VENV_ACTIVATE"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

nvidia-smi

torchrun --standalone --nproc_per_node=2 recipes/imagenet/train.py \
  --config "$CONFIG_PATH" \
  --train-root "$TRAIN_ROOT" \
  --eval-root "$EVAL_ROOT" \
  --output-dir "$OUTPUT_DIR"
