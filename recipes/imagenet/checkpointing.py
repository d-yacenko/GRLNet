from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .config import RecipeConfig
from .dist import DistributedContext, unwrap_model


def normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def try_get_git_sha(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def restore_scheduler_state(scheduler: Any, state_dict: Optional[dict[str, Any]]) -> None:
    if not state_dict:
        return
    for key, value in state_dict.items():
        setattr(scheduler, key, value)


def _latest_checkpoint_path(output_dir: Path, prefix: str) -> Path:
    return output_dir / f"{prefix}_latest.pth"


def _best_checkpoint_path(output_dir: Path, prefix: str) -> Path:
    return output_dir / f"{prefix}_best.pth"


def resolve_resume_path(output_dir: Path, prefix: str, resume_from: Optional[str]) -> Optional[Path]:
    if resume_from is None:
        return None
    if resume_from == "auto":
        path = _latest_checkpoint_path(output_dir, prefix)
        return path if path.exists() else None
    return Path(resume_from)


def build_checkpoint_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    history: dict[str, list[float]],
    best_val_loss: float,
    best_val_acc: float,
    best_val_acc_top5: float,
    config: RecipeConfig,
    ctx: DistributedContext,
    output_dir: Path,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "kind": "grlnet_recipe_checkpoint",
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_val_acc_top5": best_val_acc_top5,
        "config": config.to_dict(),
        "dist": {
            "enabled": ctx.enabled,
            "world_size": ctx.world_size,
        },
        "meta": {
            "git_sha": try_get_git_sha(repo_root),
        },
        "rng_state_rank0": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def save_checkpoint(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def save_checkpoints(
    *,
    state: dict[str, Any],
    output_dir: Path,
    prefix: str,
    save_best: bool,
    is_best: bool,
) -> dict[str, str]:
    saved: dict[str, str] = {}
    latest_path = _latest_checkpoint_path(output_dir, prefix)
    save_checkpoint(state, latest_path)
    saved["latest"] = str(latest_path)
    if save_best and is_best:
        best_path = _best_checkpoint_path(output_dir, prefix)
        save_checkpoint(state, best_path)
        saved["best"] = str(best_path)
    return saved


def load_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    unwrap_model(model).load_state_dict(normalize_state_dict_keys(state_dict))

    if isinstance(checkpoint, dict):
        if optimizer is not None and checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            restore_scheduler_state(scheduler, checkpoint["scheduler"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint if isinstance(checkpoint, dict) else {}


def save_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
