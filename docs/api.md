# API

## Model

```python
from grlnet import GRLNet, GRLNetConfig, grlnet_stabhrec40

model = grlnet_stabhrec40(weights=None, num_classes=1000)
```

Main classes and helpers:

- `GRLNet`: StabHRec40 classifier module.
- `GRLNetConfig`: dataclass with default ImageNet architecture parameters.
- `StabHRec40Cell`: recurrent cell.
- `grlnet_stabhrec40`: torchvision-style factory function.
- `GRLNetWeights`: local/URL weight descriptor registry.

## Inference Helpers

```python
from grlnet.inference import load_model, predict_image, predict_tensor, topk
```

## CLI

```bash
grlnet-info
grlnet-predict sample.jpg --checkpoint model.pth
grlnet-eval-imagenet --data-root /path/to/val --checkpoint model.pth
grlnet-train-imagenet --config src/grlnet/recipes/imagenet/configs/stabhrec40_a100_single_50e.yaml
```
