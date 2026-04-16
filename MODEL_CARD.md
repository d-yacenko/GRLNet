# Model Card: GRLNet/StabHRec40

## Model Details

- Model family: GRLNet, a Gated Recurrent Latent Network for image
  classification.
- Public architecture: StabHRec40, a stabilized H-state recurrent image
  classifier.
- Implementation: PyTorch.
- Input: RGB tensor `[B, 3, H, W]`, evaluated and trained with 224x224 crops.
- Output: class logits `[B, num_classes]`.
- Default weights: `GRLNetWeights.DEFAULT`.
- Release: `v0.3.0`.
- Checkpoint: `grlnet_stabhrec40_imagenet1k_a100_v2.pth`.
- SHA256: `75d586bdd5031fa8fa009fde618b133d5ad429e504cac81636c8daead01be4f2`.

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

| metric | value |
| --- | ---: |
| acc@1 | 0.69768 |
| acc@5 | 0.88964 |
| parameters | 3.25M |
| training epochs | 120 |

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

GRLNet/StabHRec40 is intended for image-classification research, transfer
learning experiments, compact-model comparisons, and recurrent-computation
ablation studies. The public API follows torchvision-style factory functions:

```python
from grlnet import GRLNetWeights, grlnet_stabhrec40

model = grlnet_stabhrec40(weights=GRLNetWeights.DEFAULT)
```

## Limitations

- The released weights are trained for ImageNet-1K classification and should be
  fine-tuned or re-trained for substantially different domains.
- The model is recurrent at inference time; latency depends on the configured
  number of steps and the target hardware.
- The current public checkpoint is not a robustness-certified model.

## Citation

Use the software citation in `CITATION.cff`.
