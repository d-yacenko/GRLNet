# GRLNet Agent Context

This repository is organized as a public PyTorch package for
GRLNet/StabHRec40.

## Canonical Package

```text
src/grlnet/
  models/stabhrec40.py
  models/weights.py
  inference.py
  cli/
  recipes/imagenet/
```

Public imports should use:

```python
from grlnet import GRLNet, GRLNetWeights, grlnet_stabhrec40
```

## Public Model

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

## Weights

Default weights are loaded through the torchvision-style API:

```python
model = grlnet_stabhrec40(weights=GRLNetWeights.DEFAULT)
```

The default checkpoint is hosted as a GitHub Release asset and loaded via
`torch.hub` cache with SHA256 verification.

## Recipe

Packaged train entrypoint:

```bash
grlnet-train-imagenet --config src/grlnet/recipes/imagenet/configs/stabhrec40_a100_single_50e.yaml
```

Main recipe features:

- SGD/Nesterov.
- Mixup.
- Label smoothing.
- EMA.
- Auxiliary supervision from late recurrent steps.
- AMP and channels-last on CUDA.
