# Model Card: GRLNet/StabHRec40 (dense), GRLNet/StabHRec40-Lite (depthwise-separable), and GRLNet/StabHRec40 INT8 (deployment)

This card describes the ImageNet-1K-trained variants of the GRLNet/StabHRec40
architecture family:

1. **StabHRec40 (dense, baseline)** — released in `v0.3.0`.
2. **StabHRec40-Lite (depthwise-separable)** — released in `v0.4.0`.
3. **StabHRec40 INT8 (post-training quantization)** — deployment artifact released in `v0.4.0`.

Variants 1 and 2 share the same architectural skeleton (one stabilized
recurrent cell × 12 unroll steps, H/C two-stream cell, residual stabilizers,
late readout) and the same training recipe (SGD+Nesterov, cosine LR with
warmup, MixUp, label smoothing, EMA, AMP). They differ only in how the three
k=3 convolutions inside the recurrent cell are implemented (dense vs
depthwise-separable), trading ~2.2× parameters and ~7.9× compute for a small
accuracy delta. Variant 3 is the dense baseline post-training-quantized to
INT8 via ONNX Runtime QDQ for edge deployment, with no architectural change.

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

## Model Details — StabHRec40 INT8 (post-training quantization)

- Source model: dense StabHRec40 (v0.3.0 EMA checkpoint), bit-exact-untied to a
  12-stage feed-forward graph and exported to ONNX.
- Quantization: ONNX Runtime QDQ pipeline, **per-tensor weights and
  activations**, **Percentile-99.99** activation calibration over 16
  ImageNet-1K calibration images, raw graph (no model preprocessing).
- Default weights: `GRLNetINT8Weights.DEFAULT`.
- Loader: `load_grlnet_int8_session()` returns an `onnxruntime.InferenceSession`.
- Release: `v0.4.0`.
- Checkpoint: `stabhrec40_untied_dealiased_v3_pertensor_percentile.onnx`
  (display label `grlnet_stabhrec40_imagenet1k_a100_v3_int8.onnx`), 24.8 MB
  on disk vs 95 MB for the FP32 ONNX (~3.8× smaller).
- SHA256: `69d383f383fbe04d91105ba3343de250dc1faf24939d515455696975226bdc65`.
- Accuracy: ImageNet-1K Top-1 = 68.47%, Δ = −1.30 pp vs FP32 dense (69.77%),
  measured on the full 50000-image validation split.

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

| | StabHRec40 (dense) | StabHRec40-Lite | StabHRec40 INT8 |
|---|---:|---:|---:|
| acc@1 | 0.69768 | (pending) | 0.68468 |
| acc@5 | 0.88964 | (pending) | (not measured) |
| parameters | 3,249,298 | 1,485,010 | 3,249,298 (untied 12-stage) |
| GMAC (T=12, 224×224) | ≈ 76.05 | ≈ 9.66 | ≈ 76.05 |
| disk size | 39 MB (FP32 .pth) | (pending) | 24.8 MB (ONNX QDQ) |
| training epochs | 120 | 200 (planned) | n/a (PTQ from dense) |
| hardware | NVIDIA A100 80GB | NVIDIA A100 80GB | CPU/edge (onnxruntime) |

## Training Recipe

The released checkpoint was trained on ImageNet-1K with the packaged recipe in
`src/grlnet/recipes/imagenet/configs/stabhrec40_a100_single_50e.yaml`, followed
by two A100 continuation phases (epochs 50→70 and 70→120) using the same
config family with `--resume`. Slurm wrappers used on the authors' cluster
are site-specific and not shipped with the package.

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

```python
import numpy as np
from grlnet import load_grlnet_int8_session

# INT8 ONNX deployment artifact (v0.4.0; PTQ from dense, −1.30 pp Top-1)
sess = load_grlnet_int8_session()
x = np.random.randn(1, 3, 224, 224).astype("float32")
logits = sess.run(["output"], {"input": x})[0]
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
