#!/bin/bash
# Phase-2 continuation for stabhrec40 on A100:
# resumes from the 50-epoch checkpoint, slightly increases batch size,
# resets optimizer/scheduler/scaler, and extends training to epoch 70.

#SBATCH --job-name=stabhrec40_a100_70e
#SBATCH -p gpu_A100
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH -t 4-00:00:00
#SBATCH -o /home/faenna/grl/slurm-%j.out
#SBATCH -D /home/faenna/grl/GRLNet

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/faenna/grl/GRLNet}"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-176}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-448}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export EPOCHS="${EPOCHS:-70}"
export LR="${LR:-0.048}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
export MOMENTUM="${MOMENTUM:-0.9}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
export RESUME="${RESUME:-auto}"
export RESUME_RESET_OPTIMIZER="${RESUME_RESET_OPTIMIZER:-1}"
export RESUME_RESET_SCHEDULER="${RESUME_RESET_SCHEDULER:-1}"
export RESUME_RESET_SCALER="${RESUME_RESET_SCALER:-1}"

exec bash "$REPO_DIR/recipes/imagenet/launch/slurm_a100_single_50e_stabhrec40.sh"
