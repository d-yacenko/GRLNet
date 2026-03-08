from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Optional

import torch
from torchvision.datasets import ImageFolder

from grl_model.models import grl_base, grl_tiny
from grl_model.utils import ReferenceTrainConfig, fit_reference_imagefolders, set_reference_seed


def build_model(name: str, num_classes: int, track_length: int):
    if name == "grl_tiny":
        return grl_tiny(num_classes=num_classes, track_length=track_length)
    if name == "grl_base":
        return grl_base(num_classes=num_classes, track_length=track_length)
    raise ValueError("Unknown model: %s" % name)


def configure_runtime(device: torch.device, num_threads: Optional[int]) -> None:
    if num_threads is not None and num_threads > 0:
        torch.set_num_threads(num_threads)
        try:
            torch.set_num_interop_threads(max(1, min(4, num_threads)))
        except RuntimeError:
            pass
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 1-epoch GRL benchmark on ImageNet-style train/val roots.")
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing train/ and val/ directories")
    parser.add_argument("--train-subdir", default="train")
    parser.add_argument("--val-subdir", default="val")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", choices=["grl_tiny", "grl_base"], default="grl_base")
    parser.add_argument("--device", default="cuda", help="cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--center-crop", action="store_true")
    parser.add_argument("--track-length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--progress-every-batches", type=int, default=0)
    parser.add_argument("--progress-every-samples", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    train_root = args.data_root / args.train_subdir
    eval_root = args.data_root / args.val_subdir
    if not train_root.exists():
        raise FileNotFoundError(train_root)
    if not eval_root.exists():
        raise FileNotFoundError(eval_root)

    set_reference_seed(args.seed)
    configure_runtime(device, args.num_threads)

    num_classes = len(ImageFolder(train_root).classes)
    model = build_model(args.model, num_classes=num_classes, track_length=args.track_length)

    output_dir = args.output_root / device.type
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ReferenceTrainConfig(
        epochs=1,
        save_every_epoch=True,
        log_json=True,
        checkpoint_prefix="%s_%s_1epoch" % (args.model, device.type),
        progress_log_every_batches=args.progress_every_batches,
        progress_log_every_samples=args.progress_every_samples,
    )

    run_meta = {
        "device": str(device),
        "model": args.model,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "track_length": args.track_length,
        "image_size": args.image_size,
        "center_crop": bool(args.center_crop),
        "num_threads": args.num_threads,
        "progress_every_batches": args.progress_every_batches,
        "progress_every_samples": args.progress_every_samples,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hostname": platform.node(),
        "train_root": str(train_root),
        "eval_root": str(eval_root),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    with (output_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    started = time.time()
    result = fit_reference_imagefolders(
        model,
        data_root=train_root,
        eval_root=eval_root,
        track_length=args.track_length,
        batch_size=args.batch_size,
        workers=args.workers,
        image_size=args.image_size,
        center_crop=args.center_crop,
        device=device,
        config=config,
        output_dir=output_dir,
    )
    wall_sec = time.time() - started

    summary = {
        "device": str(device),
        "elapsed_sec": result.elapsed_sec,
        "wall_sec": wall_sec,
        "best_val_acc": result.best_val_acc,
        "best_val_loss": result.best_val_loss,
        "best_epoch": result.best_epoch,
        "output_dir": str(output_dir),
    }
    with (output_dir / "benchmark_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
