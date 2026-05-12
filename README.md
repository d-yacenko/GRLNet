# GRLNet

GRLNet is a compact recurrent image classifier for PyTorch. The repository
publishes two ImageNet-1K-trained variants of the same architectural skeleton:

| variant | torch.hub entry | params | GMAC (T=12) | acc@1 | release |
| --- | --- | ---: | ---: | ---: | --- |
| **StabHRec40** (dense baseline) | `grlnet_stabhrec40` | 3.25 M | ≈ 76.05 | 69.768% | `v0.3.0` |
| **StabHRec40-Lite** (depthwise-separable) | `grlnet_stabhrec40_lite` | 1.49 M | ≈ 9.66 | (pending) | `v0.4.0` |

Both share the same recurrent cell × 12 unroll skeleton, the same training
recipe (SGD+Nesterov, cosine LR with warmup, MixUp, label smoothing, EMA,
AMP), and the same torchvision-style factory API.

```python
from grlnet import GRLNetWeights, grlnet_stabhrec40

# Dense baseline (v0.3.0)
model = grlnet_stabhrec40(weights=GRLNetWeights.DEFAULT)
model.eval()
```

```python
from grlnet import GRLNetLiteWeights, grlnet_stabhrec40_lite

# Lite variant (v0.4.0; checkpoint URL populated once the full A100 run completes)
model = grlnet_stabhrec40_lite(weights=GRLNetLiteWeights.DEFAULT)
model.eval()
```

## Model

```text
Input [B, 3, 224, 224]
  -> ConvGNAct stem
  -> H seed projection
  -> shared StabHRec40 / StabHRec40-Lite recurrent cell, repeated for 12 steps
  -> H/C global-average readout
  -> classifier head
```

The dense StabHRec40 has 3.25 M parameters and ≈ 76 GMAC at T=12 in the
ImageNet-1K configuration. The suffix `40` denotes the convolution-equivalent
feature depth at the default full unroll: 3 stem convolutions, 1 H-seed
projection, and 12 recurrent steps with 3 convolutional transforms per step.

StabHRec40-Lite replaces the three k=3 dense convolutions inside the recurrent
cell with depthwise-separable pairs (DW3×3 + PW1×1), yielding ≈ 1.49 M
parameters (2.2× fewer) and ≈ 9.66 GMAC (7.9× less compute) on the same
skeleton.

Both checkpoints are hosted as GitHub Release assets and cached by
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

The packaged ImageNet recipe is available as a console script. Choose a config
to select the variant (`stabhrec40_a100_single_50e.yaml` for the dense baseline,
`stabhrec40_lite_a100_single_120e.yaml` for the Lite variant):

```bash
# Dense baseline (50 epochs phase 1; chain phase 2/3 launchers afterward)
grlnet-train-imagenet \
  --config src/grlnet/recipes/imagenet/configs/stabhrec40_a100_single_50e.yaml \
  --train-root /path/to/imagenet/train \
  --eval-root /path/to/imagenet/val \
  --output-dir runs/stabhrec40_a100_single_50e
```

```bash
# Lite variant (recommended: single 200-epoch run from scratch)
grlnet-train-imagenet \
  --config src/grlnet/recipes/imagenet/configs/stabhrec40_lite_a100_single_200e.yaml \
  --train-root /path/to/imagenet/train \
  --eval-root /path/to/imagenet/val \
  --output-dir runs/stabhrec40_lite_a100_single_200e
```

A shorter 120-epoch ablation recipe is also provided
(`stabhrec40_lite_a100_single_120e.yaml`). The 200-epoch run is the
recommended primary recipe because the operator-level capacity reduction
(DW+PW vs dense 3×3) benefits from additional iterations to converge.

Recipe features include SGD/Nesterov, warmup plus cosine schedule, mixup, label
smoothing, late auxiliary supervision, EMA evaluation/checkpoints, AMP,
channels-last CUDA execution, and JSONL progress logging.

Slurm launch templates in `recipes/imagenet/launch/`:

- `slurm_a100_single_50e_stabhrec40.sh` — dense phase 1 (50 epochs)
- `slurm_a100_single_70e_resume20_stabhrec40.sh` — dense phase 2 lift
- `slurm_a100_single_120e_resume50_stabhrec40.sh` — dense phase 3 flat tail
- `slurm_a100_single_120e_stabhrec40_lite.sh` — Lite single-phase 120-epoch run (ablation)
- `slurm_a100_single_200e_stabhrec40_lite.sh` — Lite single-phase 200-epoch run (recommended, auto-resume on slurm wall)

## Examples

- `examples/basic_inference.py`: minimal ImageNet-1K inference.
- `examples/replace_classifier.py`: replace heads for a new class count.
- `examples/transfer_learning.py`: end-to-end ImageFolder fine-tuning.
- `examples/basic_training.py`: wrapper around the packaged training CLI.

## Citation

If you use this repository, cite the software entry in [CITATION.cff](CITATION.cff).
