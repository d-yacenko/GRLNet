from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar
from urllib.error import URLError
from urllib.parse import urlparse

import torch
from torch.hub import get_dir, load_state_dict_from_url


def _default_checkpoint_path(url: str) -> Path:
    filename = Path(urlparse(url).path).name
    return Path(get_dir()) / "checkpoints" / filename


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_state_dict_from_url_with_retries(
    url: str,
    *,
    map_location: str | torch.device = "cpu",
    progress: bool = True,
    check_hash: bool = False,
    expected_sha256: str | None = None,
    retries: int = 5,
    retry_delay_sec: float = 2.0,
) -> object:
    """Load a checkpoint through torch hub cache with retry and optional sha256 verification."""

    cache_path = _default_checkpoint_path(url)
    last_error: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            payload = load_state_dict_from_url(
                url,
                map_location=map_location,
                progress=progress,
                check_hash=check_hash,
            )
            if expected_sha256 and cache_path.exists():
                actual_sha256 = _sha256(cache_path)
                if actual_sha256 != expected_sha256:
                    cache_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Downloaded checkpoint hash mismatch for {url}: "
                        f"expected {expected_sha256}, got {actual_sha256}. "
                        "The cached file was removed; retry the download."
                    )
            return payload
        except (OSError, RuntimeError, URLError) as exc:
            last_error = exc
            if isinstance(exc, RuntimeError) and cache_path.exists():
                cache_path.unlink(missing_ok=True)
            if attempt >= retries:
                break
            # GitHub release downloads can transiently reset through proxies/CDNs.
            time.sleep(retry_delay_sec * attempt)

    raise RuntimeError(f"Failed to download checkpoint from {url} after {retries} attempts.") from last_error


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
        if name.upper() == "DEFAULT" and hasattr(cls, "DEFAULT"):
            return cls.DEFAULT
        try:
            return cls._registry[name]
        except KeyError as exc:
            available = ", ".join(sorted(cls.names()))
            raise KeyError(f"Unknown GRLNetWeights entry {name!r}. Available: {available}") from exc

    @classmethod
    def names(cls) -> set[str]:
        names = set(cls._registry)
        if hasattr(cls, "DEFAULT"):
            names.add("DEFAULT")
        return names

    def get_state_dict(
        self,
        *,
        map_location: str | torch.device = "cpu",
        progress: bool = True,
        check_hash: bool = False,
    ) -> dict[str, torch.Tensor]:
        if self.url:
            payload = _load_state_dict_from_url_with_retries(
                self.url,
                map_location=map_location,
                progress=progress,
                check_hash=check_hash,
                expected_sha256=self.meta.get("metrics", {}).get("ImageNet-1K", {}).get("sha256"),
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
        payload = _load_state_dict_from_url_with_retries(weights, map_location=map_location, progress=progress)
        return extract_model_state_dict(payload, prefer_ema=prefer_ema)
    payload = torch.load(Path(weights), map_location=map_location, weights_only=False)
    return extract_model_state_dict(payload, prefer_ema=prefer_ema)


GRLNetWeights.IMAGENET1K_STABHREC40_A100_V1 = GRLNetWeights(
    name="IMAGENET1K_STABHREC40_A100_V1",
    url="https://github.com/d-yacenko/GRLNet/releases/download/v0.3.0/grlnet_stabhrec40_imagenet1k_a100_v2.pth",
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
                "acc@1": 0.69768,
                "acc@5": 0.88964,
                "epoch": 120,
                "sha256": "75d586bdd5031fa8fa009fde618b133d5ad429e504cac81636c8daead01be4f2",
                "note": "120 epoch A100 ImageNet-1K checkpoint.",
            }
        },
    },
)

GRLNetWeights.DEFAULT = GRLNetWeights.IMAGENET1K_STABHREC40_A100_V1


@dataclass(frozen=True)
class GRLNetLiteWeights(GRLNetWeights):
    """Registry of published checkpoints for the GRLNet/StabHRec40-Lite variant.

    Separate registry namespace so ``GRLNetWeights.names()`` and
    ``GRLNetLiteWeights.names()`` stay independent. Both share the same
    download-and-verify code path inherited from ``GRLNetWeights``.
    """

    _registry: ClassVar[dict[str, "GRLNetLiteWeights"]] = {}


GRLNetLiteWeights.IMAGENET1K_STABHREC40_LITE_A100_V1 = GRLNetLiteWeights(
    name="IMAGENET1K_STABHREC40_LITE_A100_V1",
    # URL will be populated after the first published v0.4.0 release.
    url=None,
    prefer_ema=True,
    meta={
        "dataset": "ImageNet-1K",
        "architecture": "GRLNet/StabHRec40-Lite",
        "num_params": 1_485_010,
        "recipe": "src/grlnet/recipes/imagenet/configs/stabhrec40_lite_a100_single_200e.yaml",
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
                "acc@1": None,  # TBD after full A100 120-epoch training
                "acc@5": None,
                "epoch": None,
                "sha256": None,
                "note": "Pending: full ImageNet-1K training run (v0.4.0).",
            }
        },
    },
)

GRLNetLiteWeights.DEFAULT = GRLNetLiteWeights.IMAGENET1K_STABHREC40_LITE_A100_V1


__all__ = [
    "GRLNetWeights",
    "GRLNetLiteWeights",
    "extract_model_state_dict",
    "load_checkpoint_state_dict",
]
