# GRLNet

GRLNet is a compact recurrent image classifier for PyTorch. The public model
in this repository is **StabHRec40**: a stabilized H-state recurrent cell shared
across 12 recurrent steps. The default ImageNet-1K weights are loaded through
the same user-facing pattern as torchvision models.

```python
from grlnet import GRLNetWeights, grlnet_stabhrec40

model = grlnet_stabhrec40(weights=GRLNetWeights.DEFAULT)
model.eval()
```

## Model

```text
Input [B, 3, 224, 224]
  -> ConvGNAct stem
  -> H seed projection
  -> shared StabHRec40 recurrent cell, repeated for 12 steps
  -> H/C global-average readout
  -> classifier head
```

StabHRec40 has 3.25M parameters in the ImageNet-1K configuration. The suffix
`40` denotes the convolution-equivalent feature depth at the default full
unroll: 3 stem convolutions, 1 H-seed projection, and 12 recurrent steps with
3 convolutional transforms per step.

Default released weights:

| weights | dataset | acc@1 | acc@5 | params | release |
| --- | --- | ---: | ---: | ---: | --- |
| `GRLNetWeights.DEFAULT` | ImageNet-1K | 69.768% | 88.964% | 3.25M | `v0.3.0` |

The checkpoint is hosted as a GitHub Release asset and cached by
`torch.hub` under `~/.cache/torch/hub/checkpoints/`. The loader verifies the
published SHA256 checksum.

See [MODEL_CARD.md](MODEL_CARD.md) for model details, training recipe, intended
use, and limitations.

## Install

Install directly from GitHub:

```bash
pip install "grlnet @ git+https://github.com/d-yacenko/GRLNet.git"
```

Editable local install:

```bash
git clone https://github.com/d-yacenko/GRLNet.git
cd GRLNet
pip install -e .
```

## Inference

Predict one image from Python:

```python
from PIL import Image
from grlnet import GRLNetWeights, grlnet_stabhrec40
from grlnet.inference import decode_topk, load_categories, predict_image, topk

model = grlnet_stabhrec40(weights=GRLNetWeights.DEFAULT)
image = Image.open("sample.jpg").convert("RGB")
logits = predict_image(model, image)

for item in decode_topk(topk(logits, k=5), load_categories()):
    print(item)
```

The same through the CLI:

```bash
grlnet-predict sample.jpg --weights DEFAULT --topk 5
```

Evaluate an ImageFolder validation split:

```bash
grlnet-eval-imagenet \
  --data-root /path/to/imagenet/val \
  --weights DEFAULT \
  --batch-size 128
```

## Transfer Learning

Replace the classifier heads and fine-tune on a new ImageFolder dataset:

```python
from grlnet import GRLNetWeights, grlnet_stabhrec40

model = grlnet_stabhrec40(weights=GRLNetWeights.DEFAULT)
model.reset_classifier(num_classes=20)
```

Runnable example:

```bash
python examples/transfer_learning.py \
  --train-root /path/to/files20/train \
  --val-root /path/to/files20/val \
  --epochs 10
```

For linear probing, add `--freeze-backbone`.

## Training

The packaged ImageNet recipe is available as a console script:

```bash
grlnet-train-imagenet \
  --config src/grlnet/recipes/imagenet/configs/stabhrec40_a100_single_50e.yaml \
  --train-root /path/to/imagenet/train \
  --eval-root /path/to/imagenet/val \
  --output-dir runs/stabhrec40_a100_single_50e
```

Recipe features include SGD/Nesterov, warmup plus cosine schedule, mixup, label
smoothing, late auxiliary supervision, EMA evaluation/checkpoints, AMP,
channels-last CUDA execution, and JSONL progress logging.

Optional Slurm launch templates are in `recipes/imagenet/launch/`.

## Examples

- `examples/basic_inference.py`: minimal ImageNet-1K inference.
- `examples/replace_classifier.py`: replace heads for a new class count.
- `examples/transfer_learning.py`: end-to-end ImageFolder fine-tuning.
- `examples/basic_training.py`: wrapper around the packaged training CLI.

## Citation

If you use this repository, cite the software entry in [CITATION.cff](CITATION.cff).
