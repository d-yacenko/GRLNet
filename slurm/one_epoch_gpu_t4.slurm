#!/bin/bash
#SBATCH --job-name=grl_imagenet_t4_1e
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

ROOT_DIR="$HOME/grl"
REPO_DIR="$ROOT_DIR/GRLNet"
VENV_ACTIVATE="$ROOT_DIR/torch/bin/activate"
DATA_ROOT="$ROOT_DIR/image-net1000/layout"
OUTPUT_ROOT="$ROOT_DIR/runs/grl_imagenet_1epoch_t4"

cd "$ROOT_DIR"
source "$VENV_ACTIVATE"

export HOME=/home/faenna
export XDG_CONFIG_HOME=/home/faenna/grl/.config
export MPLCONFIGDIR=/home/faenna/grl/.config/matplotlib
mkdir -p "$XDG_CONFIG_HOME" "$MPLCONFIGDIR"

nvidia-smi
python "$REPO_DIR/scripts/run_one_epoch_benchmark.py" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --model grl_base \
  --device cuda \
  --batch-size 6 \
  --workers 8 \
  --progress-every-samples 1000 \
  --track-length 10 \
  --image-size 224 \
  --center-crop
