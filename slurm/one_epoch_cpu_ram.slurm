#!/bin/bash
#SBATCH --job-name=grl_imagenet_cpu_1e
#SBATCH -p amd_1Tb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH -t 2-00:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail

ROOT_DIR="$HOME/grl"
REPO_DIR="$ROOT_DIR/GRLNet"
VENV_ACTIVATE="$ROOT_DIR/torch/bin/activate"
DATA_ROOT="$ROOT_DIR/image-net1000/layout"
OUTPUT_ROOT="$ROOT_DIR/runs/grl_imagenet_1epoch_cpu"

cd "$ROOT_DIR"
source "$VENV_ACTIVATE"

export HOME=/home/faenna
export XDG_CONFIG_HOME=/home/faenna/grl/.config
export MPLCONFIGDIR=/home/faenna/grl/.config/matplotlib
mkdir -p "$XDG_CONFIG_HOME" "$MPLCONFIGDIR"

python "$REPO_DIR/scripts/run_one_epoch_benchmark.py" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --model grl_base \
  --device cpu \
  --batch-size 32 \
  --workers 16 \
  --num-threads 32 \
  --progress-every-samples 1000 \
  --track-length 10 \
  --image-size 224 \
  --center-crop
