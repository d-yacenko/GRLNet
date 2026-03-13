#!/bin/bash
#SBATCH --job-name=grlnet_t4_ddp_bs5
#SBATCH -p gpu_T4
#SBATCH --nodelist=hydra-gpu1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH -t 2-00:00:00
#SBATCH -o /home/faenna/grl/slurm-%j.out
#SBATCH -D /home/faenna/grl/GRLNet

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/grl/GRLNet}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/grl/torch/bin/activate}"
TRAIN_ROOT="${TRAIN_ROOT:-/home/faenna/grl/image-net1000/layout/train}"
EVAL_ROOT="${EVAL_ROOT:-/home/faenna/grl/image-net1000/layout/val}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/grl/runs/grlnet_t4_ddp_smoke_bs5}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/recipes/imagenet/configs/grl_t4_ddp_smoke.yaml}"
CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-grl_t4_ddp_smoke_bs5}"

cd "$REPO_DIR"
source "$VENV_ACTIVATE"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-/home/faenna/grl/.config}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/home/faenna/grl/.config/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/home/faenna/grl/.cache}"
export TORCH_HOME="${TORCH_HOME:-/home/faenna/grl/.cache/torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$XDG_CONFIG_HOME" "$MPLCONFIGDIR"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME"

nvidia-smi

torchrun --standalone --nproc_per_node=2 recipes/imagenet/train.py \
  --config "$CONFIG_PATH" \
  --train-root "$TRAIN_ROOT" \
  --eval-root "$EVAL_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint-prefix "$CHECKPOINT_PREFIX" \
  --per-gpu-batch-size 5
