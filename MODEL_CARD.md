# Model Card: GRLNet/StabHRec40 (dense) and GRLNet/StabHRec40-Lite (depthwise-separable)

This card describes two ImageNet-1K-trained variants of the GRLNet/StabHRec40
architecture family:

1. **StabHRec40 (dense, baseline)** — released in `v0.3.0`.
2. **StabHRec40-Lite (depthwise-separable)** — released in `v0.4.0`.

Both share the same architectural skeleton (one stabilized recurrent cell ×
12 unroll steps, H/C two-stream cell, residual stabilizers, late readout) and
the same training recipe (SGD+Nesterov, cosine LR with warmup, MixUp, label
smoothing, EMA, AMP). They differ only in how the three k=3 convolutions
inside the recurrent cell are implemented (dense vs depthwise-separable),
trading ~2.2× parameters and ~7.9× compute for a small accuracy delta.

## Model Details — StabHRec40 (dense)

- Model family: GRLNet, a Gated Recurrent Latent Network for image
  classification.
- Public architecture: StabHRec40, a stabilized H-state recurrent image
  classifier.
- Implementation: PyTorch.
- Input: RGB tensor `[B, 3, H, W]`, evaluated and trained with 224x224 crops.
- Output: class logits `[B, num_classes]`.
- Default weights: `GRLNetWeights.DEFAULT`.
- Release: `v0.3.0` (carried forward in `v0.4.0`, no checkpoint change).
- Checkpoint: `grlnet_stabhrec40_imagenet1k_a100_v2.pth`.
- SHA256: `75d586bdd5031fa8fa009fde618b133d5ad429e504cac81636c8daead01be4f2`.

## Model Details — StabHRec40-Lite (depthwise-separable)

- Public architecture: StabHRec40-Lite. The dense k=3 convolutions inside the
  recurrent cell (one gate-conv channels→4·channels and two delta-branch
  channels→channels) are replaced by depthwise-separable pairs (DW3×3 + PW1×1).
- All other components are identical to the dense baseline (stem, h-seed,
  GroupNorms, SiLU, gates, residual scalars, late readout, auxiliary head).
- Default weights: `GRLNetLiteWeights.DEFAULT`.
- Release: `v0.4.0`.
- Checkpoint: `grlnet_stabhrec40_lite_imagenet1k_a100_v1.pth` (pending — URL and
  SHA256 are populated when the full ImageNet-1K training run completes).

## Architecture Summary

The model uses a convolutional stem, an H-seed projection, and one shared
recurrent cell unrolled for 12 steps. Each recurrent step updates a hidden
stream `H` and a memory stream `C`. The classifier reads out global-average
pooled `H` and `C` from the final recurrent step. During training, auxiliary
heads supervise the final recurrent steps.

The ImageNet-1K configuration has 3,249,298 parameters. `StabHRec40` denotes
the default full-unroll convolution-equivalent feature depth: 3 stem
convolutions, 1 H-seed projection, and 12 recurrent steps with 3 convolutional
transforms per step.

## Metrics

ImageNet-1K validation, single-crop evaluation:

| | StabHRec40 (dense) | StabHRec40-Lite |
|---|---:|---:|
| acc@1 | 0.69768 | (pending) |
| acc@5 | 0.88964 | (pending) |
| parameters | 3,249,298 | 1,485,010 |
| GMAC (T=12, 224×224) | ≈ 76.05 | ≈ 9.66 |
| training epochs | 120 | 200 (planned) |
| hardware | NVIDIA A100 80GB | NVIDIA A100 80GB |

## Training Recipe

The released checkpoint was trained on ImageNet-1K with the packaged recipe in
`src/grlnet/recipes/imagenet/configs/stabhrec40_a100_single_50e.yaml`, followed
by the A100 continuation phases represented by the Slurm templates in
`recipes/imagenet/launch/`.

Main recipe components:

- SGD with Nesterov momentum.
- Warmup followed by cosine learning-rate scheduling.
- Mixup and label smoothing.
- Late auxiliary supervision over the final recurrent steps.
- Exponential moving average of weights for validation and released checkpoint.
- AMP, TF32, channels-last memory format, and JSONL progress logging.

## Intended Use

GRLNet is intended for image-classification research, transfer learning
experiments, compact-model comparisons, and recurrent-computation ablation
studies. The public API follows torchvision-style factory functions:

```python
from grlnet import GRLNetWeights, grlnet_stabhrec40

# Dense baseline (v0.3.0 checkpoint)
model = grlnet_stabhrec40(weights=GRLNetWeights.DEFAULT)
```

```python
from grlnet import GRLNetLiteWeights, grlnet_stabhrec40_lite

# Depthwise-separable Lite variant (v0.4.0+, checkpoint pending)
model = grlnet_stabhrec40_lite(weights=GRLNetLiteWeights.DEFAULT)
```

Or via `torch.hub`:

```python
import torch

dense = torch.hub.load("d-yacenko/GRLNet", "grlnet_stabhrec40",
                       weights="DEFAULT", trust_repo=True).eval()
lite  = torch.hub.load("d-yacenko/GRLNet", "grlnet_stabhrec40_lite",
                       weights="DEFAULT", trust_repo=True).eval()
```

## Limitations

- The released weights are trained for ImageNet-1K classification and should be
  fine-tuned or re-trained for substantially different domains.
- The model is recurrent at inference time; latency depends on the configured
  number of steps and the target hardware.
- The current public checkpoint is not a robustness-certified model.

## Citation

Use the software citation in `CITATION.cff`.
