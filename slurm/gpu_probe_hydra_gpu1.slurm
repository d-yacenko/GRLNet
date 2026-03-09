#!/bin/bash
#SBATCH --job-name=gpu_probe_hydra_gpu1
#SBATCH -p gpu_T4
#SBATCH --nodelist=hydra-gpu1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH -t 0-00:10:00
#SBATCH -o /home/faenna/grl/slurm-%j.out
#SBATCH -D /home/faenna/grl/GRLNet

set -euo pipefail

echo "job_id=$SLURM_JOB_ID"
echo "node=$(hostname)"
echo "partition=$SLURM_JOB_PARTITION"
echo "visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "hostlist=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"

echo "=== nvidia-smi -L ==="
nvidia-smi -L || true

echo "=== nvidia-smi ==="
nvidia-smi || true
