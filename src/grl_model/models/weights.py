from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch


@dataclass(frozen=True)
class GRLWeights:
    name: str
    checkpoint: Union[str, Path]
    meta: Optional[dict[str, object]] = None

    def get_state_dict(self, map_location: Union[str, torch.device] = "cpu"):
        checkpoint = torch.load(self.checkpoint, map_location=map_location)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            return checkpoint["model"]
        return checkpoint


__all__ = ["GRLWeights"]
