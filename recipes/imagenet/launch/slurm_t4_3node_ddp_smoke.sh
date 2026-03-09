#!/bin/bash
#SBATCH --job-name=grlnet_t4_3node_smoke
#SBATCH -p gpunode
#SBATCH --nodes=3
#SBATCH --nodelist=hydra-gpu1,hydra-gpu2,hydra-gpu3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH -t 2-00:00:00
#SBATCH -o /home/faenna/grl/slurm-%j.out
#SBATCH -D /home/faenna/grl/GRLNet

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/faenna/grl/GRLNet}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/home/faenna/grl/torch/bin/activate}"
TRAIN_ROOT="${TRAIN_ROOT:-/data/imagenet/train}"
EVAL_ROOT="${EVAL_ROOT:-/data/imagenet/val}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/faenna/grl/runs/grlnet_t4_3node_smoke}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/recipes/imagenet/configs/grl_t4_3node_smoke.yaml}"
MASTER_PORT="${MASTER_PORT:-29500}"

cd "$REPO_DIR"
source "$VENV_ACTIVATE"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

HOSTNAMES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MASTER_ADDR="${HOSTNAMES[0]}"

echo "nodes=${HOSTNAMES[*]}"
echo "master_addr=$MASTER_ADDR"
echo "master_port=$MASTER_PORT"
echo "nnodes=$SLURM_NNODES"
echo "tasks=$SLURM_NTASKS"

srun --kill-on-bad-exit=1 bash -lc "
  set -euo pipefail
  cd '$REPO_DIR'
  source '$VENV_ACTIVATE'

  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS='${OMP_NUM_THREADS}'
  export MASTER_ADDR='$MASTER_ADDR'
  export MASTER_PORT='$MASTER_PORT'
  export WORLD_SIZE='\${SLURM_NTASKS}'
  export RANK='\${SLURM_PROCID}'
  export LOCAL_RANK='\${SLURM_LOCALID}'

  echo 'hostname='\"\$(hostname)\"' rank='\"\$RANK\"' local_rank='\"\$LOCAL_RANK\"' node_id='\"\${SLURM_NODEID}\"' cuda_visible_devices='\"\${CUDA_VISIBLE_DEVICES:-unset}\"
  nvidia-smi -L || true
  nvidia-smi || true

  python -u recipes/imagenet/train.py \
    --config '$CONFIG_PATH' \
    --train-root '$TRAIN_ROOT' \
    --eval-root '$EVAL_ROOT' \
    --output-dir '$OUTPUT_DIR'
"
