from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from .models import GRLNet, GRLNetWeights, grlnet_stabhrec40, load_checkpoint_state_dict
from .transforms import imagenet_eval_transform


def load_model(
    *,
    checkpoint: str | Path | None = None,
    weights: GRLNetWeights | str | None = None,
    num_classes: int = 1000,
    device: str | torch.device = "cpu",
) -> GRLNet:
    model = grlnet_stabhrec40(weights=None, num_classes=num_classes)
    if weights is not None:
        state_dict = load_checkpoint_state_dict(weights)
        model.load_state_dict(state_dict, strict=True)
    if checkpoint is not None:
        state_dict = load_checkpoint_state_dict(Path(checkpoint))
        model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def predict_tensor(model: GRLNet, batch: torch.Tensor, *, device: str | torch.device | None = None) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device
    batch = batch.to(device)
    with torch.inference_mode():
        return model(batch)


def predict_image(
    model: GRLNet,
    image: Image.Image,
    *,
    image_size: int = 224,
    resize_size: int = 224,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    transform = imagenet_eval_transform(image_size=image_size, resize_size=resize_size)
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    return predict_tensor(model, tensor, device=device)


def topk(logits: torch.Tensor, *, k: int = 5) -> list[tuple[int, float]]:
    probs = logits.softmax(dim=1)
    values, indices = probs.topk(min(k, probs.shape[1]), dim=1)
    return [(int(index), float(value)) for index, value in zip(indices[0].cpu(), values[0].cpu())]


def load_categories(path: str | Path | None = None) -> list[str] | None:
    if path is None:
        try:
            from torchvision.models import ResNet18_Weights

            return list(ResNet18_Weights.IMAGENET1K_V1.meta["categories"])
        except Exception:
            return None
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def decode_topk(predictions: Sequence[tuple[int, float]], categories: Sequence[str] | None) -> list[dict[str, object]]:
    decoded = []
    for class_id, score in predictions:
        label = categories[class_id] if categories is not None and class_id < len(categories) else str(class_id)
        decoded.append({"class_id": class_id, "label": label, "score": score})
    return decoded


__all__ = [
    "decode_topk",
    "load_categories",
    "load_model",
    "predict_image",
    "predict_tensor",
    "topk",
]
