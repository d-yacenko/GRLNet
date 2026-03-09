#!/bin/bash
#SBATCH --job-name=grlnet_ref_iruo
#SBATCH -p gpu_T4
#SBATCH --nodelist=hydra-gpu1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH -t 1-00:00:00
#SBATCH -o /home/faenna/grl/slurm-%j.out
#SBATCH -D /home/faenna/grl/GRLNet

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/grl/GRLNet}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/grl/torch/bin/activate}"
TRAIN_ROOT="${TRAIN_ROOT:-/FAST/IRUO/IRUO-Base/train/level0}"
EVAL_ROOT="${EVAL_ROOT:-/FAST/IRUO/IRUO-Base/val/level0}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/grl/runs/grlnet_iruo_reference}"

cd "$REPO_DIR"
source "$VENV_ACTIVATE"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

nvidia-smi

python -u scripts/train_reference.py \
  --train-root "$TRAIN_ROOT" \
  --eval-root "$EVAL_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --model grl_base \
  --epochs 5 \
  --batch-size 4 \
  --workers 4 \
  --track-length 10 \
  --image-size 224 \
  --center-crop \
  --checkpoint-prefix grl_iruo_reference \
  --progress-every-batches 25
