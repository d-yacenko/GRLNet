# Inference

## Python API

```python
from PIL import Image

from grlnet.inference import decode_topk, load_categories, load_model, predict_image, topk

model = load_model(
    checkpoint="stabhrec40_a100_single_50e/stabhrec40_a100_single_50e_best.pth",
    num_classes=1000,
)

image = Image.open("sample.jpg").convert("RGB")
logits = predict_image(model, image)
print(decode_topk(topk(logits, k=5), load_categories()))
```

## CLI

```bash
grlnet-predict sample.jpg \
  --checkpoint stabhrec40_a100_single_50e/stabhrec40_a100_single_50e_best.pth \
  --topk 5
```

## Weights

Local checkpoints and published release weights use the same loader. Training
checkpoints may contain both `model` and `ema_model`; the inference helpers
prefer `ema_model` by default because validation during training is performed on
EMA weights.
