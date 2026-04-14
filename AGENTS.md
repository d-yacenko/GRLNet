# GRLNet Agent Context

This repository is now organized around the current GRLNet/StabHRec40 model.
The older lattice/track-classifier code is historical and must not be treated
as the public API.

## Canonical Package

```text
src/grlnet/
  models/stabhrec40.py
  models/weights.py
  inference.py
  cli/
  recipes/imagenet/
```

The installable project name is `grlnet`; public imports should use:

```python
from grlnet import GRLNet, GRLNetWeights, grlnet_stabhrec40
```

## Canonical Model

Model name: `GRLNet` / `StabHRec40`

Input:

```text
Tensor[B, 3, H, W]
```

Output:

```text
Tensor[B, num_classes]
```

Training with `return_aux=True` returns `(main_logits, aux_logits)`.

## Checkpoints

Current local experiment artifacts are in:

```text
stabhrec40_a100_single_50e/
```

Training checkpoints contain both `model` and `ema_model`. Inference should
prefer `ema_model`.

## Recipe

Packaged train entrypoint:

```bash
grlnet-train-imagenet --config src/grlnet/recipes/imagenet/configs/stabhrec40_a100_single_50e.yaml
```

Main recipe features:

- SGD/Nesterov;
- mixup;
- label smoothing;
- EMA;
- auxiliary supervision from late recurrent steps;
- AMP and channels-last on CUDA.

## Do Not Reintroduce

Do not re-expose the retired lattice/track API as part of `pyproject.toml`.
The package finder intentionally includes only `grlnet*`.
