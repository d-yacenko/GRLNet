# GRLNet ImageNet Recipe

This directory is the production/cluster lane for GRLNet.

Principles:
- `src/grl_model/*` stays academic and reference-oriented
- `recipes/imagenet/*` is the public large-scale training stack
- training semantics remain the same:
  - `train` phase uses `train_root`
  - `val` phase uses `eval_root`
  - `gold` phase uses `eval_root`

## What is implemented now

- explicit config-driven training via `train.py`
- single-GPU and `torchrun`-based DDP bootstrap
- JSONL progress logging for long cluster jobs
- GPU memory logging for batch-size tuning
- `latest` and `best` checkpoints
- resume from checkpoint path or `resume_from: auto`

## Smoke test target

The initial production smoke test is:
- ImageNet-style `train/` and `val/`
- `2 x T4`
- `5 epochs`
- progress logging during training
- checkpoint/resume enabled

On the currently visible Hydra T4 pool, the directly confirmed unique nodes are:
- `hydra-gpu1`
- `hydra-gpu2`
- `hydra-gpu3`

There is a Slurm launcher for the current `3 x T4` setup with `1 GPU per node`:
- `recipes/imagenet/launch/slurm_t4_3node_ddp_smoke.sh`
- it pins `hydra-gpu1,hydra-gpu2,hydra-gpu3`
- it uses `recipes/imagenet/configs/grl_t4_3node_smoke.yaml`

There is also a Slurm launcher for `4 x T4` with `1 GPU per node`:
- `recipes/imagenet/launch/slurm_t4_4node_ddp_smoke.sh`
- it launches one DDP rank per node via `srun`
- progress and checkpoint logs still go to a single output directory on rank 0

There is a single-GPU A100 training launcher for a longer production run:
- `recipes/imagenet/launch/slurm_a100_single_20e.sh`
- it uses `recipes/imagenet/configs/grl_a100_single_20e.yaml`
- default dataset roots are `/home/faenna/grl/image-net1000/layout/train` and `/home/faenna/grl/image-net1000/layout/val`
- it saves checkpoints and `progress.jsonl` to `/home/faenna/grl/runs/grlnet_a100_single_20e`

## Example

```bash
torchrun --standalone --nproc_per_node=2 recipes/imagenet/train.py \
  --config recipes/imagenet/configs/grl_t4_ddp_smoke.yaml \
  --train-root /data/imagenet/train \
  --eval-root /data/imagenet/val \
  --output-dir /data/runs/grlnet_t4_smoke
```

Important notes:
- `eval_on_main_rank_only: true` is currently the default for DDP smoke runs
- this keeps validation/gold metrics exact and makes `best` checkpointing simpler
- the main throughput bottlenecks are still in the data pipeline and CPU-side gold path

## Files

- `train.py`: production entrypoint
- `engine.py`: train/eval loop
- `dist.py`: DDP bootstrap/helpers
- `checkpointing.py`: checkpoint/resume helpers
- `data_pipeline.py`: recipe-side dataloader builder
- `configs/*.yaml`: example configs
- `launch/*.sh`: Slurm launch examples
