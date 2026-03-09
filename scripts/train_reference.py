from __future__ import annotations

"""Academic/reference training entrypoint.

This wrapper is intentionally thin:
- training always uses ``train_root``
- validation and gold always use ``eval_root``
- all model/dataset semantics stay in ``grl_model``
"""

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    for candidate in (repo_root, repo_root / "src"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

from grl_model.models import grl_base, grl_tiny
from grl_model.utils import ReferenceTrainConfig, fit_reference_imagefolders, set_reference_seed


def build_model(name: str, num_classes: int, track_length: int):
    if name == "grl_tiny":
        return grl_tiny(num_classes=num_classes, track_length=track_length)
    if name == "grl_base":
        return grl_base(num_classes=num_classes, track_length=track_length)
    raise ValueError(f"Unknown model: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run academic/reference training with explicit train/eval roots.",
    )
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["grl_tiny", "grl_base"], default="grl_base")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--track-length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--center-crop", action="store_true")
    parser.add_argument("--train-gold-prob", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--bias-weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler-factor", type=float, default=0.7)
    parser.add_argument("--scheduler-patience", type=int, default=9)
    parser.add_argument("--scheduler-start-epoch", type=int, default=15)
    parser.add_argument("--scheduler-window-size", type=int, default=70)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-4)
    parser.add_argument("--scheduler-mode", choices=["min", "max"], default="min")
    parser.add_argument("--checkpoint-prefix", default="grl_reference")
    parser.add_argument("--progress-every-batches", type=int, default=0)
    parser.add_argument("--progress-every-samples", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--no-save-every-epoch", action="store_true")
    parser.add_argument("--no-log-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.train_root.exists():
        raise FileNotFoundError(args.train_root)
    if not args.eval_root.exists():
        raise FileNotFoundError(args.eval_root)

    set_reference_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    from torchvision.datasets import ImageFolder

    train_classes = ImageFolder(args.train_root).classes
    eval_classes = ImageFolder(args.eval_root).classes
    if train_classes != eval_classes:
        raise ValueError("train_root and eval_root must expose the same class order")

    model = build_model(args.model, num_classes=len(train_classes), track_length=args.track_length)
    config = ReferenceTrainConfig(
        epochs=args.epochs,
        train_gold_prob=args.train_gold_prob,
        lr=args.lr,
        weight_decay=args.weight_decay,
        bias_weight_decay=args.bias_weight_decay,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        scheduler_start_epoch=args.scheduler_start_epoch,
        scheduler_window_size=args.scheduler_window_size,
        scheduler_min_lr=args.scheduler_min_lr,
        scheduler_mode=args.scheduler_mode,
        use_amp=not args.no_amp,
        benchmark=not args.no_benchmark,
        save_every_epoch=not args.no_save_every_epoch,
        checkpoint_prefix=args.checkpoint_prefix,
        log_json=not args.no_log_json,
        progress_log_every_batches=args.progress_every_batches,
        progress_log_every_samples=args.progress_every_samples,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "script": "scripts/train_reference.py",
        "train_root": str(args.train_root),
        "eval_root": str(args.eval_root),
        "model": args.model,
        "num_classes": len(train_classes),
        "track_length": args.track_length,
        "image_size": args.image_size,
        "center_crop": bool(args.center_crop),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "seed": args.seed,
        "device": str(device),
        "checkpoint_prefix": args.checkpoint_prefix,
        "semantics": {
            "train_phase_root": str(args.train_root),
            "val_phase_root": str(args.eval_root),
            "gold_phase_root": str(args.eval_root),
        },
        "reference_config": config.__dict__,
    }
    with (args.output_dir / f"{args.checkpoint_prefix}_run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    result = fit_reference_imagefolders(
        model,
        data_root=args.train_root,
        eval_root=args.eval_root,
        track_length=args.track_length,
        batch_size=args.batch_size,
        workers=args.workers,
        image_size=args.image_size,
        center_crop=args.center_crop,
        device=device,
        config=config,
        output_dir=args.output_dir,
    )

    summary = {
        "script": "scripts/train_reference.py",
        "train_root": str(args.train_root),
        "eval_root": str(args.eval_root),
        "output_dir": str(args.output_dir),
        "best_val_acc": result.best_val_acc,
        "best_val_loss": result.best_val_loss,
        "best_epoch": result.best_epoch,
        "elapsed_sec": result.elapsed_sec,
        "checkpoint_prefix": args.checkpoint_prefix,
    }
    with (args.output_dir / f"{args.checkpoint_prefix}_train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
