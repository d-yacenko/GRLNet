# GRLNet

GRLNet is a compact recurrent image classifier for PyTorch. The current public
model is **StabHRec40**, a single physical recurrent cell unrolled over 12
steps. The repository has been reorganized around this model only; the older
lattice/track-oriented experiments are no longer part of the installable
package.

## Current Model

```text
Input image [B, 3, 224, 224]
  -> ConvGNAct stem
  -> H seed projection
  -> shared recurrent StabHRec40 cell, repeated for 12 steps
  -> readout from pooled H and C streams
  -> ImageNet classifier head
```

Key properties:

- single-image classifier API: `Tensor[B, 3, H, W] -> Tensor[B, K]`;
- 3.25M parameters for the ImageNet-1K configuration;
- auxiliary supervision on the final recurrent steps during training;
- EMA weights, mixup, label smoothing and SGD/Nesterov ImageNet recipe;
- weights can be loaded from a local checkpoint or a GitHub Release URL.

## Install

Editable local install:

```bash
pip install -e .
```

Direct install from GitHub:

```bash
pip install "grlnet @ git+https://github.com/d-yacenko/GRLNet.git"
```

If this repository is consumed as a subdirectory of a larger repository:

```bash
pip install "grlnet @ git+https://github.com/<ORG>/<REPO>.git#subdirectory=GRLNet"
```

## Quickstart

Create the model:

```python
from grlnet import grlnet_stabhrec40

model = grlnet_stabhrec40(weights=None, num_classes=1000)
```

Load a local training checkpoint:

```python
from grlnet.inference import load_model

model = load_model(
    checkpoint="stabhrec40_a100_single_50e/stabhrec40_a100_single_50e_best.pth",
    num_classes=1000,
)
```

Load published weights after a GitHub Release asset is available:

```python
from grlnet import GRLNetWeights, grlnet_stabhrec40

model = grlnet_stabhrec40(weights=GRLNetWeights.DEFAULT)
```

Predict one image:

```bash
grlnet-predict sample.jpg \
  --checkpoint stabhrec40_a100_single_50e/stabhrec40_a100_single_50e_best.pth \
  --topk 5
```

Evaluate an ImageFolder validation split:

```bash
grlnet-eval-imagenet \
  --data-root /path/to/imagenet/val \
  --checkpoint stabhrec40_a100_single_50e/stabhrec40_a100_single_50e_best.pth \
  --batch-size 128
```

Train or resume with the packaged ImageNet recipe:

```bash
grlnet-train-imagenet \
  --config src/grlnet/recipes/imagenet/configs/stabhrec40_a100_single_50e.yaml \
  --train-root /path/to/imagenet/train \
  --eval-root /path/to/imagenet/val \
  --output-dir runs/stabhrec40_a100_single_50e
```

## Repository Layout

```text
src/grlnet/
  models/
    stabhrec40.py       # model definition
    weights.py          # local/URL checkpoint loading
  cli/
    predict.py
    eval_imagenet.py
    info.py
  recipes/imagenet/
    train.py            # A100-ready training recipe
    configs/
docs/
examples/
tests/
stabhrec40_a100_single_50e/
  *_best.pth            # local experiment checkpoint, not required for pip install
```

## Weights Release Workflow

The code already supports URL-based loading through
`torch.hub.load_state_dict_from_url`. Before publishing a tagged release:

1. Upload the final `.pth` checkpoint as a GitHub Release asset.
2. Update `GRLNetWeights.IMAGENET1K_STABHREC40_A100_V1.url`.
3. Update the final `acc@1`, `acc@5`, epoch count and recipe metadata.
4. Tag the repository and verify `pip install "grlnet @ git+https://..."`.

## Documentation

- [Architecture](docs/architecture.md)
- [Training](docs/training.md)
- [Inference](docs/inference.md)
- [API](docs/api.md)

## Citation

If you use this repository, see [CITATION.cff](CITATION.cff).
