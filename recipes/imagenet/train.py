from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    for candidate in (repo_root, repo_root / "src"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

from grl_model.models import GRLClassifier

from recipes.imagenet.checkpointing import load_checkpoint, resolve_resume_path, save_summary
from recipes.imagenet.config import RecipeConfig, load_recipe_config, save_recipe_config
from recipes.imagenet.data_pipeline import build_imagenet_dataloaders
from recipes.imagenet.dist import barrier, destroy_distributed, init_distributed, wrap_model
from recipes.imagenet.engine import build_recipe_optimizer, build_recipe_scheduler, run_training

def build_model(config: RecipeConfig, num_classes: int) -> GRLClassifier:
    if config.model.name != "grl":
        raise ValueError(f"Unknown model: {config.model.name}")
    model = GRLClassifier(
        num_classes=num_classes,
        track_length=config.model.track_length,
        hidden_channels=tuple(config.model.hidden_channels),
        pool_after_layers=tuple(config.model.pool_after_layers),
        global_pool=config.model.global_pool,
        aux_h_supervision=config.train.aux_h_loss_weight > 0.0,
    )
    for cell in model.cells:
        cell.forget_bias.data.fill_(float(config.model.forget_bias_init))
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production/cluster ImageNet recipe for GRLNet.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, default=None)
    parser.add_argument("--eval-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--per-gpu-batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--checkpoint-prefix", default=None)
    parser.add_argument("--progress-every-batches", type=int, default=None)
    parser.add_argument("--progress-every-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def apply_overrides(config: RecipeConfig, args: argparse.Namespace) -> RecipeConfig:
    if args.train_root is not None:
        config.data.train_root = str(args.train_root)
    if args.eval_root is not None:
        config.data.eval_root = str(args.eval_root)
    if args.output_dir is not None:
        config.checkpointing.output_dir = str(args.output_dir)
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.per_gpu_batch_size is not None:
        config.data.per_gpu_batch_size = args.per_gpu_batch_size
    if args.workers is not None:
        config.data.workers = args.workers
    if args.resume is not None:
        config.checkpointing.resume_from = args.resume
    if args.checkpoint_prefix is not None:
        config.checkpointing.checkpoint_prefix = args.checkpoint_prefix
    if args.progress_every_batches is not None:
        config.logging.progress_every_batches = args.progress_every_batches
    if args.progress_every_samples is not None:
        config.logging.progress_every_samples = args.progress_every_samples
    if args.device is not None:
        config.runtime.device = args.device
    return config


def set_recipe_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def configure_runtime(config: RecipeConfig, device: torch.device) -> None:
    torch.backends.cudnn.benchmark = bool(config.train.benchmark)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(config.runtime.tf32)
        torch.backends.cudnn.allow_tf32 = bool(config.runtime.tf32)
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_recipe_config(args.config), args)

    ctx = init_distributed(
        config.runtime.device,
        config.runtime.ddp_backend,
        timeout_minutes=config.runtime.ddp_timeout_minutes,
    )
    try:
        set_recipe_seed(config.runtime.seed)
        configure_runtime(config, ctx.device)

        output_dir = Path(config.checkpointing.output_dir)
        if ctx.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_recipe_config(config, output_dir / "resolved_config.yaml")
            run_meta = {
                "script": "recipes/imagenet/train.py",
                "device": str(ctx.device),
                "rank": ctx.rank,
                "world_size": ctx.world_size,
                "config": config.to_dict(),
                "semantics": {
                    "train_phase_root": config.data.train_root,
                    "val_phase_root": config.data.eval_root,
                    "gold_phase_root": config.data.eval_root,
                },
            }
            with (output_dir / "run_meta.json").open("w", encoding="utf-8") as f:
                json.dump(run_meta, f, indent=2)
        barrier(ctx)

        data_bundle = build_imagenet_dataloaders(config, ctx)
        model = build_model(config, num_classes=len(data_bundle.class_names)).to(ctx.device)
        model = wrap_model(model, ctx, broadcast_buffers=False, find_unused_parameters=False)

        optimizer = build_recipe_optimizer(model, config)
        scheduler = build_recipe_scheduler(optimizer, config)
        scaler = torch.amp.GradScaler("cuda", enabled=config.train.use_amp and ctx.device.type == "cuda")

        start_epoch = 0
        global_step = 0
        history = None
        best_val_loss = float("inf")
        best_val_acc = 0.0

        resume_path = resolve_resume_path(
            output_dir=output_dir,
            prefix=config.checkpointing.checkpoint_prefix,
            resume_from=config.checkpointing.resume_from,
        )
        if resume_path is not None:
            checkpoint = load_checkpoint(
                path=resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
            start_epoch = int(checkpoint.get("epoch", 0))
            global_step = int(checkpoint.get("global_step", 0))
            history = checkpoint.get("history")
            best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
            best_val_acc = float(checkpoint.get("best_val_acc", best_val_acc))

        result = run_training(
            model=model,
            dataloaders=data_bundle.dataloaders,
            train_sampler=data_bundle.train_sampler,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            ctx=ctx,
            output_dir=output_dir,
            start_epoch=start_epoch,
            global_step=global_step,
            history=history,
            best_val_loss=best_val_loss,
            best_val_acc=best_val_acc,
        )

        if ctx.is_main_process:
            summary = {
                "script": "recipes/imagenet/train.py",
                "output_dir": str(output_dir),
                "world_size": ctx.world_size,
                "device": str(ctx.device),
                "best_val_loss": result["best_val_loss"],
                "best_val_acc": result["best_val_acc"],
                "elapsed_sec": result["elapsed_sec"],
                "global_step": result["global_step"],
                "checkpoint_prefix": config.checkpointing.checkpoint_prefix,
            }
            save_summary(summary, output_dir / "train_summary.json")
            print(json.dumps(summary))
        barrier(ctx)
    finally:
        destroy_distributed(ctx)


if __name__ == "__main__":
    main()
