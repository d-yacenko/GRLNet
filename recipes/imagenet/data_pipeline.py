from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from grl_model.data.datasets import SequenceFolderDataset

from .config import RecipeConfig
from .dist import DistributedContext


@dataclass
class DataBundle:
    dataloaders: dict[str, Optional[DataLoader]]
    class_names: list[str]
    train_sampler: Optional[DistributedSampler]


def build_recipe_transforms(image_size: int, center_crop: bool):
    train_ops = [transforms.Resize(image_size)]
    eval_ops = [transforms.Resize(image_size)]
    if center_crop:
        train_ops.append(transforms.CenterCrop(image_size))
        eval_ops.append(transforms.CenterCrop(image_size))
    train_ops.extend([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(
            (-10, 10),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def _loader_kwargs(config: RecipeConfig, use_cuda: bool) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "num_workers": config.data.workers,
        "pin_memory": bool(config.data.pin_memory and use_cuda),
    }
    if config.data.workers > 0:
        kwargs["persistent_workers"] = bool(config.data.persistent_workers)
        kwargs["prefetch_factor"] = config.data.prefetch_factor
    return kwargs


def build_imagenet_dataloaders(config: RecipeConfig, ctx: DistributedContext) -> DataBundle:
    train_root = Path(config.data.train_root)
    eval_root = Path(config.data.eval_root)
    if not train_root.exists():
        raise FileNotFoundError(train_root)
    if not eval_root.exists():
        raise FileNotFoundError(eval_root)

    train_classes = ImageFolder(train_root).classes
    eval_classes = ImageFolder(eval_root).classes
    if train_classes != eval_classes:
        raise ValueError("train_root and eval_root must expose the same class order")

    train_transform, eval_transform = build_recipe_transforms(
        image_size=config.data.image_size,
        center_crop=config.data.center_crop,
    )

    # Keep grouped track construction deterministic across ranks.
    random.seed(config.runtime.seed)
    train_dataset = SequenceFolderDataset(train_root, config.model.track_length, transform=train_transform)

    val_dataset = None
    gold_dataset = None
    if not ctx.enabled or config.train.eval_on_main_rank_only is False or ctx.is_main_process:
        random.seed(config.runtime.seed + 1)
        val_dataset = SequenceFolderDataset(eval_root, config.model.track_length, transform=eval_transform)
        random.seed(config.runtime.seed + 2)
        gold_dataset = SequenceFolderDataset(eval_root, config.model.track_length, transform=eval_transform)

    # Important semantics:
    # - train always uses train_root
    # - val and gold both reuse eval_root without splitting it further
    # - DistributedSampler is applied only to train, i.e. after grouped track
    #   construction, and does not create an additional val/gold split layer
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
        drop_last=config.data.drop_last_train,
        **loader_kwargs,
    )

    val_loader = None
    gold_loader = None
    if val_dataset is not None and gold_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.per_gpu_batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
        gold_loader = DataLoader(
            gold_dataset,
            batch_size=config.data.per_gpu_batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

    return DataBundle(
        dataloaders={
            "train": train_loader,
            "val": val_loader,
            "gold": gold_loader,
        },
        class_names=train_classes,
        train_sampler=train_sampler,
    )
