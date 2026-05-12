#!/bin/bash
# StabHRec40-Lite (depthwise-separable) full ImageNet-1K training on A100 80GB.
# Single-phase 200-epoch schedule from scratch.
#
# Lite has 2.2× fewer parameters and ~7.9× less compute than the dense baseline,
# so per-GPU batch is lifted from 96 (dense phase-1) to 176. We give the model
# 200 epochs (vs 120 for dense) so the operator-level capacity reduction has
# enough iterations to converge.
#
# Auto-resume: the slurm wall is 8 days; if the run hits the wall before
# completing 200 epochs, simply resubmit this script — RESUME=auto picks
# up from the latest checkpoint and continues to epoch 200.

#SBATCH --job-name=stabhrec40_lite_a100_200e
#SBATCH -p gpu_A100
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH -t 8-00:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail

# Under slurm, sbatch copies this script to /var/spool/slurmd/... so
# ${BASH_SOURCE[0]} dirname-detection breaks. Prefer SLURM_SUBMIT_DIR (the
# directory from which sbatch was invoked — should be the repo root) and
# fall back to BASH_SOURCE only for direct (non-slurm) invocation.
if [[ -n "${REPO_DIR:-}" ]]; then
  :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

# Use the 200-epoch Lite config and a dedicated output directory.
export CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/src/grlnet/recipes/imagenet/configs/stabhrec40_lite_a100_single_200e.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/runs/stabhrec40_lite_a100_single_200e}"

# Match YAML defaults explicitly so env-var overrides are intentional, not silent.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-176}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-448}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export EPOCHS="${EPOCHS:-200}"
export LR="${LR:-0.08}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
export MOMENTUM="${MOMENTUM:-0.9}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
export RESUME="${RESUME:-auto}"

exec bash "$REPO_DIR/recipes/imagenet/launch/slurm_a100_single_50e_stabhrec40.sh"
