from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    for candidate in (repo_root, repo_root / "src"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

from grl_model.models import StabilizedHOnlyRecurrentClassifier

from recipes.imagenet.checkpointing import (
    normalize_state_dict_keys,
    resolve_resume_path,
    save_checkpoint,
    save_summary,
    try_get_git_sha,
)
from recipes.imagenet.dist import (
    DistributedContext,
    barrier,
    destroy_distributed,
    gpu_memory_stats,
    init_distributed,
    unwrap_model,
    wrap_model,
)


@dataclass
class ModelConfig:
    name: str = "stabhrec40"
    stem_channels: int = 64
    hidden_channels: int = 192
    steps: int = 12
    kernel_size: int = 3
    forget_bias: float = 1.0
    hidden_scale_init: float = -1.75
    delta_scale_init: float = -2.75
    aux_steps: int = 3
    aux_hidden_dim: int = 256
    main_dropout: float = 0.25
    aux_dropout: float = 0.15
    readout_mode: str = "hc"


@dataclass
class DataConfig:
    train_root: str = ""
    eval_root: str = ""
    gold_root: str = ""
    image_size: int = 224
    eval_resize_size: int = 224
    per_gpu_batch_size: int = 128
    per_gpu_eval_batch_size: int = 256
    workers: int = 16
    persistent_workers: bool = True
    prefetch_factor: int = 4
    pin_memory: bool = True
    drop_last_train: bool = True
    train_limit: int | None = None
    eval_limit: int | None = None
    gold_limit: int | None = None


@dataclass
class TrainConfig:
    epochs: int = 50
    grad_accum_steps: int = 1
    gradient_clip_norm: float | None = 5.0
    use_amp: bool = True
    benchmark: bool = True
    channels_last: bool = True
    eval_on_main_rank_only: bool = False
    label_smoothing: float = 0.05
    mixup_alpha: float = 0.2
    mixup_prob: float = 0.5
    ema_decay: float = 0.999
    warmup_epochs: int = 5
    lr_min_ratio: float = 0.1
    aux_weight: float = 0.2
    aux_weight_final: float = 0.05


@dataclass
class OptimizerConfig:
    lr: float = 0.08
    weight_decay: float = 1e-4
    momentum: float = 0.9


@dataclass
class LoggingConfig:
    progress_every_batches: int = 200
    progress_every_samples: int = 0
    jsonl_filename: str = "progress.jsonl"


@dataclass
class CheckpointConfig:
    output_dir: str = "runs/stabhrec40_imagenet_recipe"
    checkpoint_prefix: str = "stabhrec40_imagenet_recipe"
    save_every_epoch: bool = True
    save_best: bool = True
    save_best_gold: bool = True
    resume_from: Optional[str] = None


@dataclass
class RuntimeConfig:
    seed: int = 42
    device: str = "cuda"
    ddp_backend: str = "nccl"
    ddp_timeout_minutes: int = 60
    tf32: bool = True


@dataclass
class RecipeConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecipeConfig":
        return cls(
            model=ModelConfig(**data.get("model", {})),
            data=DataConfig(**data.get("data", {})),
            train=TrainConfig(**data.get("train", {})),
            optimizer=OptimizerConfig(**data.get("optimizer", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            checkpointing=CheckpointConfig(**data.get("checkpointing", {})),
            runtime=RuntimeConfig(**data.get("runtime", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_recipe_config(path: Path) -> RecipeConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return RecipeConfig.from_dict(raw)


def save_recipe_config(config: RecipeConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A100-ready ImageNet recipe for the stabilized H-only recurrent model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, default=None)
    parser.add_argument("--eval-root", type=Path, default=None)
    parser.add_argument("--gold-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--per-gpu-batch-size", type=int, default=None)
    parser.add_argument("--per-gpu-eval-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-min-ratio", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume-reset-optimizer", action="store_true")
    parser.add_argument("--resume-reset-scheduler", action="store_true")
    parser.add_argument("--resume-reset-scaler", action="store_true")
    parser.add_argument("--checkpoint-prefix", default=None)
    parser.add_argument("--progress-every-batches", type=int, default=None)
    parser.add_argument("--progress-every-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--gold-limit", type=int, default=None)
    return parser.parse_args()


def apply_overrides(config: RecipeConfig, args: argparse.Namespace) -> RecipeConfig:
    if args.train_root is not None:
        config.data.train_root = str(args.train_root)
    if args.eval_root is not None:
        config.data.eval_root = str(args.eval_root)
    if args.gold_root is not None:
        config.data.gold_root = str(args.gold_root)
    if args.output_dir is not None:
        config.checkpointing.output_dir = str(args.output_dir)
    if args.epochs is not None:
        config.train.epochs = int(args.epochs)
    if args.per_gpu_batch_size is not None:
        config.data.per_gpu_batch_size = int(args.per_gpu_batch_size)
    if args.per_gpu_eval_batch_size is not None:
        config.data.per_gpu_eval_batch_size = int(args.per_gpu_eval_batch_size)
    if args.grad_accum_steps is not None:
        config.train.grad_accum_steps = int(args.grad_accum_steps)
    if args.lr is not None:
        config.optimizer.lr = float(args.lr)
    if args.lr_min_ratio is not None:
        config.train.lr_min_ratio = float(args.lr_min_ratio)
    if args.weight_decay is not None:
        config.optimizer.weight_decay = float(args.weight_decay)
    if args.momentum is not None:
        config.optimizer.momentum = float(args.momentum)
    if args.warmup_epochs is not None:
        config.train.warmup_epochs = int(args.warmup_epochs)
    if args.workers is not None:
        config.data.workers = int(args.workers)
    if args.resume is not None:
        config.checkpointing.resume_from = args.resume
    if args.checkpoint_prefix is not None:
        config.checkpointing.checkpoint_prefix = args.checkpoint_prefix
    if args.progress_every_batches is not None:
        config.logging.progress_every_batches = int(args.progress_every_batches)
    if args.progress_every_samples is not None:
        config.logging.progress_every_samples = int(args.progress_every_samples)
    if args.device is not None:
        config.runtime.device = args.device
    if args.train_limit is not None:
        config.data.train_limit = int(args.train_limit)
    if args.eval_limit is not None:
        config.data.eval_limit = int(args.eval_limit)
    if args.gold_limit is not None:
        config.data.gold_limit = int(args.gold_limit)
    return config


def set_recipe_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def configure_runtime(config: RecipeConfig, device: torch.device) -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = bool(config.train.benchmark)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(config.runtime.tf32)
        torch.backends.cudnn.allow_tf32 = bool(config.runtime.tf32)
        torch.cuda.empty_cache()


def maybe_to_channels_last(x: torch.Tensor, *, enabled: bool) -> torch.Tensor:
    if enabled and x.ndim == 4:
        return x.contiguous(memory_format=torch.channels_last)
    return x


def build_model(config: RecipeConfig, num_classes: int) -> StabilizedHOnlyRecurrentClassifier:
    if config.model.name != "stabhrec40":
        raise ValueError(f"Unknown model: {config.model.name}")
    return StabilizedHOnlyRecurrentClassifier(
        num_classes=num_classes,
        stem_channels=config.model.stem_channels,
        hidden_channels=config.model.hidden_channels,
        steps=config.model.steps,
        kernel_size=config.model.kernel_size,
        forget_bias=config.model.forget_bias,
        aux_steps=config.model.aux_steps,
        aux_hidden_dim=config.model.aux_hidden_dim,
        main_dropout=config.model.main_dropout,
        aux_dropout=config.model.aux_dropout,
        hidden_scale_init=config.model.hidden_scale_init,
        delta_scale_init=config.model.delta_scale_init,
        readout_mode=config.model.readout_mode,
    )


def build_transforms(config: RecipeConfig) -> tuple[transforms.Compose, transforms.Compose]:
    image_size = int(config.data.image_size)
    eval_resize_size = int(config.data.eval_resize_size or image_size)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(eval_resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def _loader_kwargs(config: RecipeConfig, use_cuda: bool) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "num_workers": config.data.workers,
        "pin_memory": bool(config.data.pin_memory and use_cuda),
    }
    if config.data.workers > 0:
        kwargs["persistent_workers"] = bool(config.data.persistent_workers)
        kwargs["prefetch_factor"] = int(config.data.prefetch_factor)
    return kwargs


def maybe_limit_dataset(dataset, limit: int | None):
    if limit is None or limit >= len(dataset):
        return dataset
    return torch.utils.data.Subset(dataset, list(range(int(limit))))


def build_dataloaders(config: RecipeConfig, ctx: DistributedContext):
    train_root = Path(config.data.train_root)
    eval_root = Path(config.data.eval_root)
    gold_root = Path(config.data.gold_root) if config.data.gold_root else None
    if not train_root.exists():
        raise FileNotFoundError(train_root)
    if not eval_root.exists():
        raise FileNotFoundError(eval_root)
    if gold_root is not None and not gold_root.exists():
        raise FileNotFoundError(gold_root)

    train_transform, eval_transform = build_transforms(config)
    train_dataset = ImageFolder(train_root, transform=train_transform)
    val_dataset = ImageFolder(eval_root, transform=eval_transform)
    gold_dataset = ImageFolder(gold_root, transform=eval_transform) if gold_root is not None else None

    if train_dataset.classes != val_dataset.classes:
        raise ValueError("train_root and eval_root must expose the same class order")
    if gold_dataset is not None and train_dataset.classes != gold_dataset.classes:
        raise ValueError("train_root and gold_root must expose the same class order")
    class_names = list(train_dataset.classes)

    train_dataset = maybe_limit_dataset(train_dataset, config.data.train_limit)
    val_dataset = maybe_limit_dataset(val_dataset, config.data.eval_limit)
    gold_dataset = maybe_limit_dataset(gold_dataset, config.data.gold_limit) if gold_dataset is not None else None

    train_sampler: Optional[DistributedSampler] = None
    if ctx.enabled:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=True,
            drop_last=config.data.drop_last_train,
        )

    loader_kwargs = _loader_kwargs(config, use_cuda=ctx.device.type == "cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.per_gpu_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=bool(config.data.drop_last_train),
        **loader_kwargs,
    )

    val_loader = None
    gold_loader = None
    if not ctx.enabled or not config.train.eval_on_main_rank_only or ctx.is_main_process:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.per_gpu_eval_batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
        if gold_dataset is not None:
            gold_loader = DataLoader(
                gold_dataset,
                batch_size=config.data.per_gpu_eval_batch_size,
                shuffle=False,
                drop_last=False,
                **loader_kwargs,
            )

    return {
        "train": train_loader,
        "val": val_loader,
        "gold": gold_loader,
    }, class_names, train_sampler


def build_optimizer(model: nn.Module, config: RecipeConfig) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.parameters(),
        lr=config.optimizer.lr,
        momentum=config.optimizer.momentum,
        weight_decay=config.optimizer.weight_decay,
        nesterov=True,
    )


def lr_multiplier_for_epoch(config: RecipeConfig, epoch: int) -> float:
    epochs = int(config.train.epochs)
    warmup_epochs = max(0, min(int(config.train.warmup_epochs), max(1, epochs - 1)))
    min_ratio = max(0.0, min(float(config.train.lr_min_ratio), 1.0))
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return 0.2 + 0.8 * ((epoch + 1) / warmup_epochs)
    if epochs <= warmup_epochs:
        return 1.0
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def build_scheduler(optimizer: torch.optim.Optimizer, config: RecipeConfig) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch: int) -> float:
        return lr_multiplier_for_epoch(config, epoch)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def set_optimizer_lrs(
    optimizer: torch.optim.Optimizer,
    *,
    current_lr: float,
    initial_lr: Optional[float] = None,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(current_lr)
        if initial_lr is not None or "initial_lr" in group:
            group["initial_lr"] = float(group["lr"] if initial_lr is None else initial_lr)


def align_scheduler_to_resume_epoch(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: RecipeConfig,
    start_epoch: int,
) -> None:
    base_lr = float(config.optimizer.lr)
    current_lr = base_lr * lr_multiplier_for_epoch(config, start_epoch)
    set_optimizer_lrs(optimizer, current_lr=current_lr, initial_lr=base_lr)
    scheduler.base_lrs = [base_lr for _ in optimizer.param_groups]
    scheduler.last_epoch = int(start_epoch)
    scheduler._step_count = int(start_epoch) + 1
    scheduler._last_lr = [current_lr for _ in optimizer.param_groups]


def create_ema_model(model: nn.Module) -> nn.Module:
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


def update_ema(ema_model: nn.Module, model: nn.Module, *, decay: float) -> None:
    with torch.no_grad():
        model_state = unwrap_model(model).state_dict()
        ema_state = ema_model.state_dict()
        for name, ema_tensor in ema_state.items():
            model_tensor = model_state[name]
            if not torch.is_floating_point(ema_tensor):
                ema_tensor.copy_(model_tensor)
                continue
            ema_tensor.mul_(decay).add_(model_tensor.detach(), alpha=1.0 - decay)


def mixup_batch(x: torch.Tensor, y: torch.Tensor, *, alpha: float, prob: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0.0 or prob <= 0.0 or torch.rand((), device=x.device).item() >= prob:
        return x, y, y, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixed_ce_loss(criterion: nn.Module, logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    if lam >= 1.0 - 1e-8:
        return criterion(logits, y_a)
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


def aux_weight_for_epoch(config: RecipeConfig, epoch_idx: int) -> float:
    start = float(config.train.aux_weight)
    end = float(config.train.aux_weight_final)
    if config.train.epochs <= 1:
        return end
    half = max(1, config.train.epochs // 2)
    if epoch_idx < half:
        return start
    progress = (epoch_idx - half) / max(1, config.train.epochs - half - 1)
    return start + (end - start) * progress


def _topk_hits(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    topk = logits.topk(min(k, logits.shape[1]), dim=1).indices
    return int(topk.eq(labels.view(-1, 1)).any(dim=1).sum().item())


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
        f.flush()


def save_history(path: Path, history: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def build_checkpoint_state(
    *,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    history: dict[str, list[float]],
    best_val_acc: float,
    best_val_acc_top5: float,
    best_gold_acc: float,
    best_gold_acc_top5: float,
    config: RecipeConfig,
    ctx: DistributedContext,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "kind": "stabhrec40_recipe_checkpoint",
        "model": unwrap_model(model).state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "history": history,
        "best_val_acc": float(best_val_acc),
        "best_val_acc_top5": float(best_val_acc_top5),
        "best_gold_acc": float(best_gold_acc),
        "best_gold_acc_top5": float(best_gold_acc_top5),
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


def load_checkpoint_state(
    *,
    path: Path,
    model: nn.Module,
    ema_model: Optional[nn.Module],
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    scaler: Optional[torch.amp.GradScaler],
    load_optimizer_state: bool = True,
    load_scheduler_state: bool = True,
    load_scaler_state: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    unwrap_model(model).load_state_dict(normalize_state_dict_keys(state), strict=True)
    if ema_model is not None and isinstance(checkpoint, dict) and checkpoint.get("ema_model") is not None:
        ema_model.load_state_dict(normalize_state_dict_keys(checkpoint["ema_model"]), strict=True)
    if isinstance(checkpoint, dict):
        if load_optimizer_state and optimizer is not None and checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if load_scheduler_state and scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if load_scaler_state and scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint if isinstance(checkpoint, dict) else {}


def evaluate_phase(
    *,
    model: nn.Module,
    loader: Optional[DataLoader],
    criterion: nn.Module,
    config: RecipeConfig,
    ctx: DistributedContext,
) -> dict[str, float]:
    if loader is None:
        return {
            "loss": float("nan"),
            "loss_main": float("nan"),
            "loss_aux": float("nan"),
            "acc": float("nan"),
            "acc_top5": float("nan"),
            "num_samples": 0.0,
            "elapsed_sec": 0.0,
        }
    model.eval()
    loss_sum = 0.0
    loss_main_sum = 0.0
    loss_aux_sum = 0.0
    correct_sum = 0.0
    top5_sum = 0.0
    sample_sum = 0
    started = time.time()
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(ctx.device, non_blocking=True)
            labels = labels.to(ctx.device, non_blocking=True)
            inputs = maybe_to_channels_last(inputs, enabled=config.train.channels_last)
            with torch.amp.autocast("cuda", enabled=config.train.use_amp and ctx.device.type == "cuda"):
                logits, aux_logits = model(inputs, return_aux=True)
                loss_main = criterion(logits, labels)
                if aux_logits:
                    weights = torch.linspace(1.0, float(len(aux_logits)), steps=len(aux_logits), device=logits.device)
                    aux_losses = torch.stack([criterion(aux_logit, labels) for aux_logit in aux_logits])
                    loss_aux = (weights * aux_losses).sum() / weights.sum()
                else:
                    loss_aux = logits.new_zeros(())
                loss = loss_main + float(config.train.aux_weight_final) * loss_aux
            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            loss_main_sum += loss_main.item() * batch_size
            loss_aux_sum += loss_aux.item() * batch_size
            correct_sum += (logits.argmax(dim=1) == labels).sum().item()
            top5_sum += _topk_hits(logits, labels, 5)
            sample_sum += batch_size
    return {
        "loss": loss_sum / max(sample_sum, 1),
        "loss_main": loss_main_sum / max(sample_sum, 1),
        "loss_aux": loss_aux_sum / max(sample_sum, 1),
        "acc": correct_sum / max(sample_sum, 1),
        "acc_top5": top5_sum / max(sample_sum, 1),
        "num_samples": float(sample_sum),
        "elapsed_sec": time.time() - started,
    }


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
        history_path = output_dir / f"{config.checkpointing.checkpoint_prefix}_history.json"
        progress_path = output_dir / config.logging.jsonl_filename
        latest_path = output_dir / f"{config.checkpointing.checkpoint_prefix}_latest.pth"
        best_path = output_dir / f"{config.checkpointing.checkpoint_prefix}_best.pth"
        best_gold_path = output_dir / f"{config.checkpointing.checkpoint_prefix}_best_gold.pth"

        if ctx.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_recipe_config(config, output_dir / "resolved_config.yaml")
            run_meta = {
                "script": "recipes/imagenet/train_stabhrec40.py",
                "device": str(ctx.device),
                "rank": ctx.rank,
                "world_size": ctx.world_size,
                "config": config.to_dict(),
                "semantics": {
                    "train_phase_root": config.data.train_root,
                    "val_phase_root": config.data.eval_root,
                    "gold_phase_root": config.data.gold_root or None,
                },
            }
            with (output_dir / "run_meta.json").open("w", encoding="utf-8") as f:
                json.dump(run_meta, f, indent=2)
        barrier(ctx)

        dataloaders, class_names, train_sampler = build_dataloaders(config, ctx)
        base_model = build_model(config, num_classes=len(class_names)).to(ctx.device)
        if config.train.channels_last:
            base_model = base_model.to(memory_format=torch.channels_last)
        ema_model = create_ema_model(base_model).to(ctx.device)
        if config.train.channels_last:
            ema_model = ema_model.to(memory_format=torch.channels_last)
        model = wrap_model(base_model, ctx, broadcast_buffers=False, find_unused_parameters=False)

        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)
        scaler = torch.amp.GradScaler("cuda", enabled=config.train.use_amp and ctx.device.type == "cuda")
        criterion = nn.CrossEntropyLoss(label_smoothing=float(config.train.label_smoothing))

        start_epoch = 0
        global_step = 0
        best_val_acc = -float("inf")
        best_val_acc_top5 = -float("inf")
        best_gold_acc = -float("inf")
        best_gold_acc_top5 = -float("inf")
        history = {
            "loss_train": [],
            "loss_main_train": [],
            "loss_aux_train": [],
            "acc_train": [],
            "acc_top5_train": [],
            "loss_val": [],
            "loss_main_val": [],
            "loss_aux_val": [],
            "acc_val": [],
            "acc_top5_val": [],
            "loss_gold": [],
            "loss_main_gold": [],
            "loss_aux_gold": [],
            "acc_gold": [],
            "acc_top5_gold": [],
            "lr": [],
            "aux_weight": [],
            "mixup_lambda": [],
        }

        resume_path = resolve_resume_path(
            output_dir=output_dir,
            prefix=config.checkpointing.checkpoint_prefix,
            resume_from=config.checkpointing.resume_from,
        )
        if resume_path is not None:
            checkpoint = load_checkpoint_state(
                path=resume_path,
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                load_optimizer_state=not args.resume_reset_optimizer,
                load_scheduler_state=not args.resume_reset_scheduler,
                load_scaler_state=not args.resume_reset_scaler,
            )
            start_epoch = int(checkpoint.get("epoch", 0))
            global_step = int(checkpoint.get("global_step", 0))
            history = checkpoint.get("history", history)
            best_val_acc = float(checkpoint.get("best_val_acc", best_val_acc))
            best_val_acc_top5 = float(checkpoint.get("best_val_acc_top5", best_val_acc_top5))
            best_gold_acc = float(checkpoint.get("best_gold_acc", best_gold_acc))
            best_gold_acc_top5 = float(checkpoint.get("best_gold_acc_top5", best_gold_acc_top5))
            if args.resume_reset_scheduler:
                align_scheduler_to_resume_epoch(optimizer, scheduler, config, start_epoch)
            elif args.resume_reset_optimizer and scheduler is not None and getattr(scheduler, "_last_lr", None):
                base_lrs = list(getattr(scheduler, "base_lrs", []))
                target_lrs = list(getattr(scheduler, "_last_lr", []))
                for idx, group in enumerate(optimizer.param_groups):
                    if idx < len(target_lrs):
                        group["lr"] = float(target_lrs[idx])
                    if idx < len(base_lrs):
                        group["initial_lr"] = float(base_lrs[idx])

        if start_epoch == 0 and ctx.is_main_process and dataloaders["val"] is not None:
            init_val = evaluate_phase(
                model=ema_model,
                loader=dataloaders["val"],
                criterion=criterion,
                config=config,
                ctx=ctx,
            )
            history["loss_val"].append(float(init_val["loss"]))
            history["loss_main_val"].append(float(init_val["loss_main"]))
            history["loss_aux_val"].append(float(init_val["loss_aux"]))
            history["acc_val"].append(float(init_val["acc"]))
            history["acc_top5_val"].append(float(init_val["acc_top5"]))
            if dataloaders["gold"] is not None:
                init_gold = evaluate_phase(
                    model=ema_model,
                    loader=dataloaders["gold"],
                    criterion=criterion,
                    config=config,
                    ctx=ctx,
                )
                history["loss_gold"].append(float(init_gold["loss"]))
                history["loss_main_gold"].append(float(init_gold["loss_main"]))
                history["loss_aux_gold"].append(float(init_gold["loss_aux"]))
                history["acc_gold"].append(float(init_gold["acc"]))
                history["acc_top5_gold"].append(float(init_gold["acc_top5"]))
            save_history(history_path, history)

        started = time.time()
        for epoch in range(start_epoch, config.train.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            total_main_loss = 0.0
            total_aux_loss = 0.0
            total_acc = 0.0
            total_top5 = 0.0
            total_mixup = 0.0
            total_samples = 0
            total_batches = len(dataloaders["train"])
            next_batch_threshold = config.logging.progress_every_batches or 0
            next_sample_threshold = config.logging.progress_every_samples or 0
            last_step_ended = time.time()
            epoch_aux_weight = aux_weight_for_epoch(config, epoch)

            for batch_idx, (inputs, labels) in enumerate(dataloaders["train"], start=1):
                data_time_sec = time.time() - last_step_ended
                inputs = inputs.to(ctx.device, non_blocking=True)
                labels = labels.to(ctx.device, non_blocking=True)
                inputs = maybe_to_channels_last(inputs, enabled=config.train.channels_last)
                inputs, y_a, y_b, lam = mixup_batch(
                    inputs,
                    labels,
                    alpha=config.train.mixup_alpha,
                    prob=config.train.mixup_prob,
                )

                with torch.amp.autocast("cuda", enabled=config.train.use_amp and ctx.device.type == "cuda"):
                    logits, aux_logits = model(inputs, return_aux=True)
                    main_loss = mixed_ce_loss(criterion, logits, y_a, y_b, lam)
                    if aux_logits:
                        weights = torch.linspace(1.0, float(len(aux_logits)), steps=len(aux_logits), device=logits.device)
                        aux_losses = torch.stack([mixed_ce_loss(criterion, aux_logit, y_a, y_b, lam) for aux_logit in aux_logits])
                        aux_loss = (weights * aux_losses).sum() / weights.sum()
                    else:
                        aux_loss = logits.new_zeros(())
                    raw_loss = main_loss + epoch_aux_weight * aux_loss
                    loss = raw_loss / max(config.train.grad_accum_steps, 1)

                scaler.scale(loss).backward()
                should_step = batch_idx % max(config.train.grad_accum_steps, 1) == 0 or batch_idx == total_batches
                if should_step:
                    if config.train.gradient_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    update_ema(ema_model, model, decay=float(config.train.ema_decay))

                batch_size = int(labels.size(0))
                total_samples += batch_size
                total_loss += float(raw_loss.item()) * batch_size
                total_main_loss += float(main_loss.item()) * batch_size
                total_aux_loss += float(aux_loss.item()) * batch_size
                total_acc += float((logits.argmax(dim=1) == labels).float().mean().item()) * batch_size
                total_top5 += float(_topk_hits(logits, labels, 5))
                total_mixup += float(lam) * batch_size
                global_step += 1

                if ctx.is_main_process:
                    processed_samples = total_samples
                    should_log = False
                    if config.logging.progress_every_batches > 0 and batch_idx >= next_batch_threshold:
                        should_log = True
                    if config.logging.progress_every_samples > 0 and processed_samples >= next_sample_threshold:
                        should_log = True
                    if should_log:
                        record = {
                            "event": "batch_progress",
                            "epoch": epoch + 1,
                            "phase": "train",
                            "batch": batch_idx,
                            "batches_total": total_batches,
                            "rank": ctx.rank,
                            "world_size": ctx.world_size,
                            "samples_done_rank": processed_samples,
                            "approx_global_samples_done": processed_samples * ctx.world_size,
                            "data_time_sec": round(data_time_sec, 4),
                            "avg_loss_so_far": total_loss / max(total_samples, 1),
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                        record.update(gpu_memory_stats(ctx.device))
                        append_jsonl(progress_path, record)
                        if config.logging.progress_every_batches > 0:
                            while batch_idx >= next_batch_threshold:
                                next_batch_threshold += config.logging.progress_every_batches
                        if config.logging.progress_every_samples > 0:
                            while processed_samples >= next_sample_threshold:
                                next_sample_threshold += config.logging.progress_every_samples
                last_step_ended = time.time()

            scheduler.step()

            train_metrics = {
                "loss": total_loss / max(total_samples, 1),
                "loss_main": total_main_loss / max(total_samples, 1),
                "loss_aux": total_aux_loss / max(total_samples, 1),
                "acc": total_acc / max(total_samples, 1),
                "acc_top5": total_top5 / max(total_samples, 1),
                "num_samples": float(total_samples),
            }

            if ctx.enabled and config.train.eval_on_main_rank_only:
                barrier(ctx)
                if ctx.is_main_process:
                    val_metrics = evaluate_phase(
                        model=ema_model,
                        loader=dataloaders["val"],
                        criterion=criterion,
                        config=config,
                        ctx=ctx,
                    )
                    gold_metrics = evaluate_phase(
                        model=ema_model,
                        loader=dataloaders["gold"],
                        criterion=criterion,
                        config=config,
                        ctx=ctx,
                    )
                else:
                    val_metrics = evaluate_phase(model=ema_model, loader=None, criterion=criterion, config=config, ctx=ctx)
                    gold_metrics = evaluate_phase(model=ema_model, loader=None, criterion=criterion, config=config, ctx=ctx)
                barrier(ctx)
            else:
                val_metrics = evaluate_phase(
                    model=ema_model,
                    loader=dataloaders["val"],
                    criterion=criterion,
                    config=config,
                    ctx=ctx,
                )
                gold_metrics = evaluate_phase(
                    model=ema_model,
                    loader=dataloaders["gold"],
                    criterion=criterion,
                    config=config,
                    ctx=ctx,
                )

            if ctx.is_main_process:
                history["loss_train"].append(float(train_metrics["loss"]))
                history["loss_main_train"].append(float(train_metrics["loss_main"]))
                history["loss_aux_train"].append(float(train_metrics["loss_aux"]))
                history["acc_train"].append(float(train_metrics["acc"]))
                history["acc_top5_train"].append(float(train_metrics["acc_top5"]))
                history["loss_val"].append(float(val_metrics["loss"]))
                history["loss_main_val"].append(float(val_metrics["loss_main"]))
                history["loss_aux_val"].append(float(val_metrics["loss_aux"]))
                history["acc_val"].append(float(val_metrics["acc"]))
                history["acc_top5_val"].append(float(val_metrics["acc_top5"]))
                history["loss_gold"].append(float(gold_metrics["loss"]))
                history["loss_main_gold"].append(float(gold_metrics["loss_main"]))
                history["loss_aux_gold"].append(float(gold_metrics["loss_aux"]))
                history["acc_gold"].append(float(gold_metrics["acc"]))
                history["acc_top5_gold"].append(float(gold_metrics["acc_top5"]))
                history["lr"].append(float(optimizer.param_groups[0]["lr"]))
                history["aux_weight"].append(float(epoch_aux_weight))
                history["mixup_lambda"].append(float(total_mixup / max(total_samples, 1)))
                save_history(history_path, history)

                append_jsonl(progress_path, {"event": "phase_summary", "epoch": epoch + 1, "phase": "train", **train_metrics, "lr": optimizer.param_groups[0]["lr"]})
                append_jsonl(progress_path, {"event": "phase_summary", "epoch": epoch + 1, "phase": "val", **val_metrics, "lr": optimizer.param_groups[0]["lr"]})
                if dataloaders["gold"] is not None:
                    append_jsonl(progress_path, {"event": "phase_summary", "epoch": epoch + 1, "phase": "gold", **gold_metrics, "lr": optimizer.param_groups[0]["lr"]})

                is_best_val = not math.isnan(val_metrics["acc"]) and float(val_metrics["acc"]) > best_val_acc
                is_best_gold = not math.isnan(gold_metrics["acc"]) and float(gold_metrics["acc"]) > best_gold_acc
                if is_best_val:
                    best_val_acc = float(val_metrics["acc"])
                    best_val_acc_top5 = float(val_metrics["acc_top5"])
                if is_best_gold:
                    best_gold_acc = float(gold_metrics["acc"])
                    best_gold_acc_top5 = float(gold_metrics["acc_top5"])

                checkpoint_state = build_checkpoint_state(
                    model=model,
                    ema_model=ema_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch + 1,
                    global_step=global_step,
                    history=history,
                    best_val_acc=best_val_acc,
                    best_val_acc_top5=best_val_acc_top5,
                    best_gold_acc=best_gold_acc,
                    best_gold_acc_top5=best_gold_acc_top5,
                    config=config,
                    ctx=ctx,
                )
                save_checkpoint(checkpoint_state, latest_path)
                saved_paths = {"latest": str(latest_path)}
                if config.checkpointing.save_best and is_best_val:
                    save_checkpoint(checkpoint_state, best_path)
                    saved_paths["best"] = str(best_path)
                if config.checkpointing.save_best_gold and is_best_gold:
                    save_checkpoint(checkpoint_state, best_gold_path)
                    saved_paths["best_gold"] = str(best_gold_path)
                append_jsonl(progress_path, {"event": "checkpoint_saved", "epoch": epoch + 1, "paths": saved_paths})

                epoch_summary = {
                    "event": "epoch_summary",
                    "epoch": epoch + 1,
                    "lr": optimizer.param_groups[0]["lr"],
                    "best_val_acc": best_val_acc,
                    "best_val_acc_top5": best_val_acc_top5,
                    "best_gold_acc": best_gold_acc,
                    "best_gold_acc_top5": best_gold_acc_top5,
                    "elapsed_avg_sec": (time.time() - started) / max(epoch + 1 - start_epoch, 1),
                    "rank": ctx.rank,
                    "world_size": ctx.world_size,
                    "device": str(ctx.device),
                }
                epoch_summary.update(gpu_memory_stats(ctx.device))
                append_jsonl(progress_path, epoch_summary)

            barrier(ctx)

        if ctx.is_main_process:
            summary = {
                "script": "recipes/imagenet/train_stabhrec40.py",
                "output_dir": str(output_dir),
                "world_size": ctx.world_size,
                "device": str(ctx.device),
                "best_val_acc": finite_or_none(best_val_acc),
                "best_val_acc_top5": finite_or_none(best_val_acc_top5),
                "best_gold_acc": finite_or_none(best_gold_acc),
                "best_gold_acc_top5": finite_or_none(best_gold_acc_top5),
                "elapsed_sec": time.time() - started,
                "global_step": global_step,
                "checkpoint_prefix": config.checkpointing.checkpoint_prefix,
                "latest_checkpoint": str(latest_path),
                "best_checkpoint": str(best_path) if best_path.exists() else None,
                "best_gold_checkpoint": str(best_gold_path) if best_gold_path.exists() else None,
                "history_file": str(history_path),
            }
            if history.get("acc_train"):
                summary["final_acc_train"] = finite_or_none(history["acc_train"][-1])
            if history.get("acc_top5_train"):
                summary["final_acc_top5_train"] = finite_or_none(history["acc_top5_train"][-1])
            if history.get("acc_val"):
                summary["final_acc_val"] = finite_or_none(history["acc_val"][-1])
            if history.get("acc_top5_val"):
                summary["final_acc_top5_val"] = finite_or_none(history["acc_top5_val"][-1])
            if history.get("acc_gold"):
                summary["final_acc_gold"] = finite_or_none(history["acc_gold"][-1])
            if history.get("acc_top5_gold"):
                summary["final_acc_top5_gold"] = finite_or_none(history["acc_top5_gold"][-1])
            save_summary(summary, output_dir / "train_summary.json")
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        barrier(ctx)
    finally:
        destroy_distributed(ctx)


if __name__ == "__main__":
    main()
