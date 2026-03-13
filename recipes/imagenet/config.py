from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "grl_base"
    track_length: int = 10


@dataclass
class DataConfig:
    train_root: str = ""
    eval_root: str = ""
    image_size: int = 224
    center_crop: bool = True
    per_gpu_batch_size: int = 6
    workers: int = 8
    persistent_workers: bool = True
    prefetch_factor: int = 2
    pin_memory: bool = True
    drop_last_train: bool = True


@dataclass
class TrainConfig:
    epochs: int = 5
    grad_accum_steps: int = 1
    train_gold_prob: float = 0.5
    use_amp: bool = True
    benchmark: bool = True
    eval_on_main_rank_only: bool = True


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-2
    bias_weight_decay: float = 0.0


@dataclass
class SchedulerConfig:
    factor: float = 0.7
    patience: int = 9
    start_epoch: int = 15
    window_size: int = 70
    min_lr: float = 1e-4
    mode: str = "min"


@dataclass
class LoggingConfig:
    progress_every_batches: int = 50
    progress_every_samples: int = 0
    jsonl_filename: str = "progress.jsonl"
    log_json: bool = True


@dataclass
class CheckpointConfig:
    output_dir: str = "runs/grl_imagenet_recipe"
    checkpoint_prefix: str = "grl_imagenet_recipe"
    save_every_epoch: bool = True
    save_best: bool = True
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
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
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
            scheduler=SchedulerConfig(**data.get("scheduler", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            checkpointing=CheckpointConfig(**data.get("checkpointing", {})),
            runtime=RuntimeConfig(**data.get("runtime", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_raw_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text) or {}


def load_recipe_config(path: Path) -> RecipeConfig:
    return RecipeConfig.from_dict(_load_raw_config(path))


def save_recipe_config(config: RecipeConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
