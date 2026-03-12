#!/bin/bash
#SBATCH --job-name=grlnet_t4_single_smoke
#SBATCH -p gpu_T4
#SBATCH --nodelist=hydra-gpu1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH -t 2-00:00:00
#SBATCH -o /home/faenna/grl/slurm-%j.out
#SBATCH -D /home/faenna/grl/GRLNet

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/grl/GRLNet}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/grl/torch/bin/activate}"
TRAIN_ROOT="${TRAIN_ROOT:-/home/faenna/grl/image-net1000/layout/train}"
EVAL_ROOT="${EVAL_ROOT:-/home/faenna/grl/image-net1000/layout/val}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/grl/runs/grlnet_t4_single_smoke}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/recipes/imagenet/configs/grl_t4_single_smoke.yaml}"

cd "$REPO_DIR"
source "$VENV_ACTIVATE"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-/home/faenna/grl/.config}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/home/faenna/grl/.config/matplotlib}"
mkdir -p "$XDG_CONFIG_HOME" "$MPLCONFIGDIR"

nvidia-smi

python -u recipes/imagenet/train.py \
  --config "$CONFIG_PATH" \
  --train-root "$TRAIN_ROOT" \
  --eval-root "$EVAL_ROOT" \
  --output-dir "$OUTPUT_DIR"
