from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from grl_model.data.datasets import SequenceFolderDataset


@dataclass
class ReferenceTrainConfig:
    """Reference training hyperparameters aligned with the notebook recipe. / Reference-гиперпараметры обучения, согласованные с рецептом ноутбука."""
    epochs: int = 100
    train_gold_prob: float = 0.5
    lr: float = 1e-3
    weight_decay: float = 1e-2
    bias_weight_decay: float = 0.0
    scheduler_factor: float = 0.7
    scheduler_patience: int = 9
    scheduler_start_epoch: int = 15
    scheduler_window_size: int = 70
    scheduler_min_lr: float = 1e-4
    scheduler_mode: str = "min"
    use_amp: bool = True
    benchmark: bool = True
    save_every_epoch: bool = True
    checkpoint_prefix: str = "grl"
    log_json: bool = True
    progress_log_every_batches: int = 0
    progress_log_every_samples: int = 0


class SmoothedReduceLROnPlateau:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        factor: float = 0.5,
        patience: int = 7,
        start_epoch: int = 10,
        window_size: int = 7,
        min_lr: float = 1e-5,
        verbose: bool = True,
        mode: str = "min",
    ) -> None:
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.start_epoch = start_epoch
        self.window_size = window_size
        self.min_lr = min_lr
        self.verbose = verbose
        self.mode = mode
        self.loss_history: list[float] = []
        self.best_smoothed: Optional[float] = None
        self.num_bad_epochs = 0

    def step(self, epoch: int, current_loss: float) -> None:
        self.loss_history.append(current_loss)
        if epoch + 1 < self.start_epoch:
            return
        if len(self.loss_history) < self.window_size:
            return

        window = self.loss_history[-self.window_size:]
        smoothed = sum(window) / self.window_size

        if self.best_smoothed is None:
            self.best_smoothed = smoothed
            self.num_bad_epochs = 0
            return

        improved = (
            self.mode == "min" and smoothed < self.best_smoothed
        ) or (
            self.mode == "max" and smoothed > self.best_smoothed
        )

        if improved:
            self.best_smoothed = smoothed
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            for idx, param_group in enumerate(self.optimizer.param_groups):
                old_lr = param_group["lr"]
                new_lr = max(old_lr * self.factor, self.min_lr)
                if new_lr < old_lr:
                    param_group["lr"] = new_lr
                    if self.verbose:
                        print(
                            f"[Epoch {epoch + 1}] SmoothedReduceLROnPlateau: "
                            f"lr[{idx}] {old_lr:.6f} -> {new_lr:.6f}"
                        )
            self.best_smoothed = smoothed
            self.num_bad_epochs = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "factor": self.factor,
            "patience": self.patience,
            "start_epoch": self.start_epoch,
            "window_size": self.window_size,
            "min_lr": self.min_lr,
            "verbose": self.verbose,
            "mode": self.mode,
            "loss_history": self.loss_history,
            "best_smoothed": self.best_smoothed,
            "num_bad_epochs": self.num_bad_epochs,
        }


@dataclass
class ReferenceTrainResult:
    """Structured return value of the reference training helpers. / Структурированный результат reference-утилит обучения."""
    model: nn.Module
    best_val_acc: float
    best_val_loss: float
    best_epoch: int
    history: dict[str, list[float]]
    elapsed_sec: float


def set_reference_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_reference_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-2, bias_weight_decay: float = 0.0):
    """Build the notebook-style AdamW optimizer with split bias decay. / Собрать AdamW в стиле ноутбука с раздельным weight decay для bias."""
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and "bias" not in n],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and "bias" in n],
            "weight_decay": bias_weight_decay,
        },
    ]
    return torch.optim.AdamW(param_groups, lr=lr)



def build_reference_scheduler(optimizer: torch.optim.Optimizer, config: ReferenceTrainConfig):
    """Build the notebook-style smoothed plateau scheduler. / Собрать сглаженный plateau-scheduler в стиле ноутбука."""
    return SmoothedReduceLROnPlateau(
        optimizer=optimizer,
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        start_epoch=config.scheduler_start_epoch,
        window_size=config.scheduler_window_size,
        min_lr=config.scheduler_min_lr,
        verbose=True,
        mode=config.scheduler_mode,
    )



def _phase_names(dataloaders: dict[str, Any]) -> list[str]:
    phases = [phase for phase in ("train", "val", "gold") if phase in dataloaders]
    if not phases:
        raise ValueError("dataloaders must contain at least one of: train, val, gold")
    return phases



def _history_key(metric: str, phase: str) -> str:
    return f"{metric}_{phase}"


def _should_log_progress(
    *,
    batch_idx: int,
    processed_samples: int,
    next_batch_threshold: int,
    next_sample_threshold: int,
    config: ReferenceTrainConfig,
) -> bool:
    if config.progress_log_every_batches > 0 and batch_idx >= next_batch_threshold:
        return True
    if config.progress_log_every_samples > 0 and processed_samples >= next_sample_threshold:
        return True
    return False


def _build_default_transforms(image_size: int = 224, center_crop: bool = False):
    from torchvision import transforms

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


def _stratified_split_indices(all_idxs: List[int], all_labels: List[int], test_size: float, random_state: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(random_state)
    by_label: Dict[int, List[int]] = {}
    for idx, label in zip(all_idxs, all_labels):
        by_label.setdefault(label, []).append(idx)

    train_idxs: List[int] = []
    test_idxs: List[int] = []
    for label in sorted(by_label):
        idxs = list(by_label[label])
        rng.shuffle(idxs)
        if len(idxs) < 2:
            train_idxs.extend(idxs)
            continue
        n_test = int(round(len(idxs) * test_size))
        n_test = min(max(1, n_test), len(idxs) - 1)
        test_idxs.extend(idxs[:n_test])
        train_idxs.extend(idxs[n_test:])
    return train_idxs, test_idxs


def _random_split_indices(indices: List[int], test_size: float, random_state: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(random_state)
    idxs = list(indices)
    rng.shuffle(idxs)
    if len(idxs) < 2:
        return idxs, []
    n_test = int(round(len(idxs) * test_size))
    n_test = min(max(1, n_test), len(idxs) - 1)
    return idxs[n_test:], idxs[:n_test]



def plot_history(history: dict[str, list[float]]) -> None:
    """Plot train/val/gold loss and accuracy curves if present in history. / Построить кривые loss и accuracy для train/val/gold, если они присутствуют в истории."""
    epochs = range(1, len(next(iter(history.values()), [])) + 1)

    plt.figure()
    for phase in ("train", "val", "gold"):
        key = _history_key("loss", phase)
        if key in history:
            plt.plot(epochs, history[key], label=key)
    plt.legend(framealpha=1, frameon=True)
    plt.title("Loss")
    plt.show()

    plt.figure()
    for phase in ("train", "val", "gold"):
        key = _history_key("acc", phase)
        if key in history:
            plt.plot(epochs, history[key], label=key)
    plt.legend(framealpha=1, frameon=True)
    plt.title("Accuracy")
    plt.show()



def fit_reference(
    model: nn.Module,
    dataloaders: dict[str, Any],
    *,
    dataset_sizes: Optional[Dict[str, int]] = None,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[SmoothedReduceLROnPlateau] = None,
    device: Optional[torch.device] = None,
    config: Optional[ReferenceTrainConfig] = None,
    output_dir: Optional[Union[str, Path]] = None,
    class_names: Optional[List[str]] = None,
) -> ReferenceTrainResult:
    """Run the reference training loop on caller-provided dataloaders. / Запустить reference training loop на переданных dataloader'ах.

    This is the main low-level training API for users who already control their own
    dataset and dataloader construction.
    Это основной низкоуровневый training API для пользователей, которые сами управляют
    построением датасетов и dataloader'ов.
    """
    del class_names  # reserved for future richer reporting / зарезервировано для более подробного отчёта в будущем

    config = config or ReferenceTrainConfig()
    phases = _phase_names(dataloaders)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = build_reference_optimizer(
            model,
            lr=config.lr,
            weight_decay=config.weight_decay,
            bias_weight_decay=config.bias_weight_decay,
        )
    if scheduler is None:
        scheduler = build_reference_scheduler(optimizer, config)

    if dataset_sizes is None:
        dataset_sizes = {phase: len(dataloaders[phase].dataset) for phase in phases}

    use_amp = bool(config.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    torch.backends.cudnn.benchmark = bool(config.benchmark)

    history: Dict[str, List[float]] = {}
    for phase in phases:
        history[_history_key("loss", phase)] = []
        history[_history_key("acc", phase)] = []

    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_acc = 0.0
    best_val_loss = float("inf")
    since = time.time()

    for epoch in range(config.epochs):
        for phase in phases:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            processed_samples = 0
            phase_started = time.time()
            total_samples = dataset_sizes[phase]
            total_batches = len(dataloaders[phase])
            next_batch_threshold = config.progress_log_every_batches or 0
            next_sample_threshold = config.progress_log_every_samples or 0

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase], start=1):
                if phase == "train" and config.train_gold_prob > 0.0 and random.random() < config.train_gold_prob:
                    inputs = inputs.clone()
                    model.prep_batch(inputs)
                if phase == "gold":
                    inputs = inputs.clone()
                    model.prep_batch(inputs)

                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.set_grad_enabled(phase == "train"):
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                processed_samples += inputs.size(0)

                if _should_log_progress(
                    batch_idx=batch_idx,
                    processed_samples=processed_samples,
                    next_batch_threshold=next_batch_threshold,
                    next_sample_threshold=next_sample_threshold,
                    config=config,
                ):
                    elapsed_phase_sec = time.time() - phase_started
                    avg_sec_per_sample = elapsed_phase_sec / max(processed_samples, 1)
                    eta_sec = avg_sec_per_sample * max(total_samples - processed_samples, 0)
                    progress_record = {
                        "epoch": epoch + 1,
                        "phase": phase,
                        "batch": batch_idx,
                        "batches_total": total_batches,
                        "samples_done": processed_samples,
                        "samples_total": total_samples,
                        "samples_pct": round(100.0 * processed_samples / max(total_samples, 1), 3),
                        "elapsed_phase_sec": round(elapsed_phase_sec, 3),
                        "eta_phase_sec": round(eta_sec, 3),
                    }
                    if config.log_json:
                        print(json.dumps(progress_record))
                    else:
                        print(progress_record)
                    if config.progress_log_every_batches > 0:
                        while batch_idx >= next_batch_threshold:
                            next_batch_threshold += config.progress_log_every_batches
                    if config.progress_log_every_samples > 0:
                        while processed_samples >= next_sample_threshold:
                            next_sample_threshold += config.progress_log_every_samples

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            history[_history_key("loss", phase)].append(float(epoch_loss))
            history[_history_key("acc", phase)].append(float(epoch_acc))

            if phase == "val":
                scheduler.step(epoch, epoch_loss)
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_val_acc = float(epoch_acc)
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

            dataset = dataloaders[phase].dataset
            if phase == "train" and hasattr(dataset, "on_epoch_end"):
                dataset.on_epoch_end()

        record = {"epoch": epoch + 1}
        for phase in phases:
            record[_history_key("loss", phase)] = history[_history_key("loss", phase)][-1]
            record[_history_key("acc", phase)] = history[_history_key("acc", phase)][-1]
        record["elapsed_avg_sec"] = (time.time() - since) / (epoch + 1)
        record["lr"] = optimizer.param_groups[0]["lr"]
        if config.log_json:
            print(json.dumps(record))
        else:
            print(record)

        if output_path is not None and config.save_every_epoch:
            latest = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "history": history,
                "config": asdict(config),
            }
            torch.save(latest, output_path / f"{config.checkpoint_prefix}_latest.pth")
            with (output_path / f"{config.checkpoint_prefix}_history.json").open("w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

    model.load_state_dict(best_model_wts)
    model.eval()
    elapsed_sec = time.time() - since

    if output_path is not None:
        best = {
            "model": model.state_dict(),
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch + 1,
            "history": history,
            "config": asdict(config),
        }
        torch.save(best, output_path / f"{config.checkpoint_prefix}_best.pth")

    return ReferenceTrainResult(
        model=model,
        best_val_acc=best_val_acc,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch + 1,
        history=history,
        elapsed_sec=elapsed_sec,
    )


def fit_reference_imagefolders(
    model: nn.Module,
    *,
    data_root: Union[str, Path],
    eval_root: Optional[Union[str, Path]] = None,
    track_length: int,
    batch_size: int = 64,
    workers: int = 8,
    image_size: int = 224,
    center_crop: bool = False,
    val_split: float = 0.2,
    gold_split: float = 0.1,
    random_state: int = 42,
    train_transform=None,
    eval_transform=None,
    device: Optional[torch.device] = None,
    config: Optional[ReferenceTrainConfig] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> ReferenceTrainResult:
    """Run the canonical GRL training recipe on ImageFolder-style data. / Запустить канонический GRL-рецепт обучения на данных в стиле ImageFolder.

    The public high-level trainer follows the grouped-observation concept of the model:
    it builds `SequenceFolderDataset` splits and trains on grouped samples with notebook-style
    gold mixing. If `eval_root` is omitted, `train/val/gold` are split from `data_root`
    exactly in the notebook spirit. If `eval_root` is provided, the whole `data_root`
    is used for training and `eval_root` is used for both `val` and `gold` protocols.
    Публичный high-level trainer следует grouped-observation концепции модели:
    он строит split'ы на базе `SequenceFolderDataset` и обучает на групповых sample'ах
    с notebook-style gold mixing. Если `eval_root` не задан, `train/val/gold` делятся
    из `data_root` в духе ноутбука. Если `eval_root` задан, весь `data_root` идёт в train,
    а `eval_root` используется и для `val`, и для `gold` как два eval-протокола.
    """
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    if train_transform is None or eval_transform is None:
        default_train, default_eval = _build_default_transforms(
            image_size=image_size,
            center_crop=center_crop,
        )
        train_transform = train_transform or default_train
        eval_transform = eval_transform or default_eval

    data_root = Path(data_root)
    eval_root = Path(eval_root) if eval_root is not None else None

    if eval_root is None:
        full_folder = ImageFolder(data_root)
        all_idxs = list(range(len(full_folder.samples)))
        all_labels = [label for _, label in full_folder.samples]
        train_idxs, val_idxs = _stratified_split_indices(
            all_idxs,
            all_labels,
            test_size=val_split,
            random_state=random_state,
        )
        train_idxs, gold_idxs = _random_split_indices(
            train_idxs,
            test_size=gold_split,
            random_state=random_state,
        )
        train_dataset = SequenceFolderDataset(data_root, track_length, allowed_idxs=train_idxs, transform=train_transform)
        val_dataset = SequenceFolderDataset(data_root, track_length, allowed_idxs=val_idxs, transform=eval_transform)
        gold_dataset = SequenceFolderDataset(data_root, track_length, allowed_idxs=gold_idxs, transform=eval_transform)
    else:
        train_dataset = SequenceFolderDataset(data_root, track_length, transform=train_transform)
        val_dataset = SequenceFolderDataset(eval_root, track_length, transform=eval_transform)
        gold_dataset = SequenceFolderDataset(eval_root, track_length, transform=eval_transform)
        if train_dataset.folder.classes != val_dataset.folder.classes:
            raise ValueError("data_root and eval_root must expose the same class order")

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True),
        "gold": DataLoader(gold_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True),
    }
    dataset_sizes = {phase: len(loader.dataset) for phase, loader in dataloaders.items()}

    return fit_reference(
        model,
        dataloaders,
        dataset_sizes=dataset_sizes,
        device=device,
        config=config,
        output_dir=output_dir,
        class_names=train_dataset.folder.classes,
    )
