from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional, Union

import torch
from torch.hub import load_state_dict_from_url


def _extract_model_state(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model" in payload:
        payload = payload["model"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(payload)!r}")
    if any(str(key).startswith("module.") for key in payload):
        return {(str(key)[7:] if str(key).startswith("module.") else str(key)): value for key, value in payload.items()}
    return payload


@dataclass(frozen=True)
class GRLWeights:
    """Published GRL weights descriptor.

    The descriptor can point either to a local checkpoint or to a public URL.
    Official published weights are expected to use ``url`` so consumers can do:

    ``model = GRLClassifier.from_weights(GRLWeights.IMAGENET1K_AUXH_A100_E50_V1)``
    """

    name: str
    url: Optional[str] = None
    checkpoint: Optional[Union[str, Path]] = None
    meta: dict[str, Any] = field(default_factory=dict)

    _registry: ClassVar[dict[str, "GRLWeights"]] = {}

    def __post_init__(self) -> None:
        type(self)._registry[self.name] = self

    @property
    def model_kwargs(self) -> dict[str, Any]:
        return dict(self.meta.get("model_kwargs", {}))

    def get_state_dict(
        self,
        *,
        map_location: Union[str, torch.device] = "cpu",
        progress: bool = True,
        check_hash: bool = False,
        file_name: Optional[str] = None,
    ) -> dict[str, torch.Tensor]:
        if self.url is not None:
            payload = load_state_dict_from_url(
                self.url,
                model_dir=None,
                map_location=map_location,
                progress=progress,
                check_hash=check_hash,
                file_name=file_name,
            )
        elif self.checkpoint is not None:
            payload = torch.load(Path(self.checkpoint), map_location=map_location)
        else:
            raise RuntimeError(
                f"GRLWeights entry {self.name!r} does not have a published url yet. "
                "Set the release asset url in grl_model.models.weights before using it."
            )
        return _extract_model_state(payload)

    @classmethod
    def get(cls, name: str) -> "GRLWeights":
        try:
            return cls._registry[name]
        except KeyError as exc:
            available = ", ".join(sorted(cls._registry))
            raise KeyError(f"Unknown GRLWeights entry {name!r}. Available: {available}") from exc


# Fill the final release URL before publishing the first official weights.
GRLWeights.IMAGENET1K_AUXH_A100_E50_V1 = GRLWeights(
    name="IMAGENET1K_AUXH_A100_E50_V1",
    url="https://github.com/d-yacenko/GRLNet/releases/download/v0.1.0-rc1/grl_imagenet1k_auxh_a100_e50_v1.pth",
    meta={
        "dataset": "ImageNet-1K",
        "recipe": "recipes/imagenet/configs/grl_a100_single_50e.yaml",
        "model_kwargs": {
            "num_classes": 1000,
            "track_length": 10,
            "hidden_channels": (24, 32, 32, 48, 48, 64, 64, 96, 160),
            "pool_after_layers": (0, 2, 4, 6, 7),
            "global_pool": 2,
            "aux_h_supervision": True,
        },
    },
)


__all__ = ["GRLWeights"]
