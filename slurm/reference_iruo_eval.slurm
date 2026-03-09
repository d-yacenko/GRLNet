#!/bin/bash
#SBATCH --job-name=grlnet_ref_iruo_eval
#SBATCH -p gpu_T4
#SBATCH --nodelist=hydra-gpu1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH -t 0-08:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/grl/GRLNet}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/grl/torch/bin/activate}"
TRAIN_ROOT="${TRAIN_ROOT:-/FAST/IRUO/IRUO-Base/train/level0}"
EVAL_ROOT="${EVAL_ROOT:-/FAST/IRUO/IRUO-Base/val/level0}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$HOME/grl/runs/grlnet_iruo_reference/grl_iruo_reference_best.pth}"
OUTPUT_JSON="${OUTPUT_JSON:-$HOME/grl/runs/grlnet_iruo_reference/eval_summary.json}"

cd "$REPO_DIR"
source "$VENV_ACTIVATE"

export PYTHONUNBUFFERED=1

nvidia-smi

python -u scripts/eval_reference.py \
  --train-root "$TRAIN_ROOT" \
  --eval-root "$EVAL_ROOT" \
  --checkpoint "$CHECKPOINT_PATH" \
  --model grl_base \
  --batch-size 8 \
  --workers 4 \
  --track-length 10 \
  --image-size 224 \
  --center-crop \
  --output-json "$OUTPUT_JSON"
