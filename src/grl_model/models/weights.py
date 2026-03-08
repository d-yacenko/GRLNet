from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class GRLWeights:
    name: str
    checkpoint: str | Path
    meta: dict[str, object] | None = None

    def get_state_dict(self, map_location: str | torch.device = "cpu"):
        checkpoint = torch.load(self.checkpoint, map_location=map_location)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            return checkpoint["model"]
        return checkpoint


__all__ = ["GRLWeights"]
