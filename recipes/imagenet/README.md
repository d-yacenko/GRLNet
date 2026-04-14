# GRLNet ImageNet Recipe

This directory keeps source-tree launch files for the current StabHRec40 model.
The installable copy of the same recipe is under `src/grlnet/recipes/imagenet`.

Canonical entrypoint:

```bash
python recipes/imagenet/train_stabhrec40.py \
  --config recipes/imagenet/configs/stabhrec40_a100_single_50e.yaml \
  --train-root /path/to/imagenet/train \
  --eval-root /path/to/imagenet/val \
  --output-dir runs/stabhrec40_a100_single_50e
```

After pip installation, use the console script:

```bash
grlnet-train-imagenet \
  --config src/grlnet/recipes/imagenet/configs/stabhrec40_a100_single_50e.yaml \
  --train-root /path/to/imagenet/train \
  --eval-root /path/to/imagenet/val
```

Current launch files:

- `launch/slurm_a100_single_50e_stabhrec40.sh`
- `launch/slurm_a100_single_70e_resume20_stabhrec40.sh`
- `launch/slurm_a100_single_120e_resume50_stabhrec40.sh`

Recipe features:

- SGD/Nesterov optimizer;
- warmup + cosine schedule;
- mixup and label smoothing;
- late auxiliary supervision;
- EMA validation/checkpoints;
- JSONL progress logging;
- `latest` and `best` checkpoints;
- resume from explicit path or `resume_from: auto`.
