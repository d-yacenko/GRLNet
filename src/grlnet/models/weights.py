from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch.hub import load_state_dict_from_url


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not any(str(key).startswith("module.") for key in state_dict):
        return state_dict
    return {
        (str(key)[7:] if str(key).startswith("module.") else str(key)): value
        for key, value in state_dict.items()
    }


def extract_model_state_dict(payload: object, *, prefer_ema: bool = True) -> dict[str, torch.Tensor]:
    """Extract a model state dict from a training or inference checkpoint."""

    if isinstance(payload, dict):
        if prefer_ema and isinstance(payload.get("ema_model"), dict):
            payload = payload["ema_model"]
        elif isinstance(payload.get("model"), dict):
            payload = payload["model"]
        elif isinstance(payload.get("state_dict"), dict):
            payload = payload["state_dict"]

    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(payload)!r}")
    return _strip_module_prefix(payload)


@dataclass(frozen=True)
class GRLNetWeights:
    """Torchvision-style descriptor for published GRLNet weights."""

    name: str
    url: str | None = None
    checkpoint: str | Path | None = None
    prefer_ema: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    _registry: ClassVar[dict[str, "GRLNetWeights"]] = {}

    def __post_init__(self) -> None:
        type(self)._registry[self.name] = self

    @property
    def model_kwargs(self) -> dict[str, Any]:
        return dict(self.meta.get("model_kwargs", {}))

    @classmethod
    def get(cls, name: str) -> "GRLNetWeights":
        try:
            return cls._registry[name]
        except KeyError as exc:
            available = ", ".join(sorted(cls._registry))
            raise KeyError(f"Unknown GRLNetWeights entry {name!r}. Available: {available}") from exc

    @classmethod
    def names(cls) -> set[str]:
        return set(cls._registry)

    def get_state_dict(
        self,
        *,
        map_location: str | torch.device = "cpu",
        progress: bool = True,
        check_hash: bool = False,
    ) -> dict[str, torch.Tensor]:
        if self.url:
            payload = load_state_dict_from_url(
                self.url,
                map_location=map_location,
                progress=progress,
                check_hash=check_hash,
            )
        elif self.checkpoint:
            payload = torch.load(Path(self.checkpoint), map_location=map_location, weights_only=False)
        else:
            raise RuntimeError(
                f"Weights entry {self.name!r} has no URL/checkpoint yet. "
                "Publish the checkpoint as a GitHub Release asset and set its URL here."
            )
        return extract_model_state_dict(payload, prefer_ema=self.prefer_ema)


def load_checkpoint_state_dict(
    weights: GRLNetWeights | str | Path,
    *,
    map_location: str | torch.device = "cpu",
    progress: bool = True,
    prefer_ema: bool = True,
) -> dict[str, torch.Tensor]:
    if isinstance(weights, GRLNetWeights):
        return weights.get_state_dict(map_location=map_location, progress=progress)
    if isinstance(weights, str) and weights in GRLNetWeights.names():
        return GRLNetWeights.get(weights).get_state_dict(map_location=map_location, progress=progress)
    if isinstance(weights, str) and weights.startswith(("http://", "https://")):
        payload = load_state_dict_from_url(weights, map_location=map_location, progress=progress)
        return extract_model_state_dict(payload, prefer_ema=prefer_ema)
    payload = torch.load(Path(weights), map_location=map_location, weights_only=False)
    return extract_model_state_dict(payload, prefer_ema=prefer_ema)


GRLNetWeights.IMAGENET1K_STABHREC40_A100_V1 = GRLNetWeights(
    name="IMAGENET1K_STABHREC40_A100_V1",
    # Replace this release asset with the final checkpoint URL before tagging.
    url="https://github.com/d-yacenko/GRLNet/releases/download/v0.2.0/grlnet_stabhrec40_imagenet1k_a100_v1.pth",
    prefer_ema=True,
    meta={
        "dataset": "ImageNet-1K",
        "architecture": "GRLNet/StabHRec40",
        "num_params": 3_249_298,
        "recipe": "src/grlnet/recipes/imagenet/configs/stabhrec40_a100_single_50e.yaml",
        "model_kwargs": {
            "num_classes": 1000,
            "stem_channels": 64,
            "hidden_channels": 192,
            "steps": 12,
            "kernel_size": 3,
            "forget_bias": 1.0,
            "hidden_scale_init": -1.75,
            "delta_scale_init": -2.75,
            "aux_steps": 3,
            "aux_hidden_dim": 256,
            "main_dropout": 0.25,
            "aux_dropout": 0.15,
            "readout_mode": "hc",
        },
        "metrics": {
            "ImageNet-1K": {
                "acc@1": 0.693,
                "note": "Current in-progress checkpoint; update after final release run.",
            }
        },
    },
)

GRLNetWeights.DEFAULT = GRLNetWeights.IMAGENET1K_STABHREC40_A100_V1


__all__ = [
    "GRLNetWeights",
    "extract_model_state_dict",
    "load_checkpoint_state_dict",
]
