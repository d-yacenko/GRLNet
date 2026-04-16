# GRLNet ImageNet Recipe

This directory keeps optional source-tree launch files for GRLNet/StabHRec40.
The train implementation and the default config are packaged under
`src/grlnet/recipes/imagenet` and are installed with `pip install`.

Canonical entrypoint:

```bash
python -m grlnet.recipes.imagenet.train \
  --config src/grlnet/recipes/imagenet/configs/stabhrec40_a100_single_50e.yaml \
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

The Slurm scripts intentionally do not contain user-specific paths. Submit them
from the repository root or pass `REPO_DIR`, `TRAIN_ROOT`, `EVAL_ROOT`,
`OUTPUT_DIR`, and optionally `VENV_ACTIVATE` through the environment.

Recipe features:

- SGD/Nesterov optimizer;
- warmup + cosine schedule;
- mixup and label smoothing;
- late auxiliary supervision;
- EMA validation/checkpoints;
- JSONL progress logging;
- `latest` and `best` checkpoints;
- resume from explicit path or `resume_from: auto`.
