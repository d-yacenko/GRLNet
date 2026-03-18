from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from grl_model.data.adapters import apply_gold_protocol
from grl_model.utils import ReferenceTrainConfig, SmoothedReduceLROnPlateau, build_reference_optimizer

from .checkpointing import build_checkpoint_state, save_checkpoints
from .config import RecipeConfig
from .dist import DistributedContext, barrier, broadcast_float, gpu_memory_stats, reduce_sum, unwrap_model


def build_recipe_optimizer(model: nn.Module, config: RecipeConfig) -> torch.optim.Optimizer:
    return build_reference_optimizer(
        unwrap_model(model),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        bias_weight_decay=config.optimizer.bias_weight_decay,
    )


def build_recipe_scheduler(optimizer: torch.optim.Optimizer, config: RecipeConfig) -> SmoothedReduceLROnPlateau:
    scheduler_config = ReferenceTrainConfig(
        scheduler_factor=config.scheduler.factor,
        scheduler_patience=config.scheduler.patience,
        scheduler_start_epoch=config.scheduler.start_epoch,
        scheduler_window_size=config.scheduler.window_size,
        scheduler_min_lr=config.scheduler.min_lr,
        scheduler_mode=config.scheduler.mode,
    )
    return SmoothedReduceLROnPlateau(
        optimizer=optimizer,
        factor=scheduler_config.scheduler_factor,
        patience=scheduler_config.scheduler_patience,
        start_epoch=scheduler_config.scheduler_start_epoch,
        window_size=scheduler_config.scheduler_window_size,
        min_lr=scheduler_config.scheduler_min_lr,
        verbose=True,
        mode=scheduler_config.scheduler_mode,
    )


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
        f.flush()


def _should_log_progress(
    *,
    batch_idx: int,
    processed_samples: int,
    next_batch_threshold: int,
    next_sample_threshold: int,
    config: RecipeConfig,
) -> bool:
    if config.logging.progress_every_batches > 0 and batch_idx >= next_batch_threshold:
        return True
    if config.logging.progress_every_samples > 0 and processed_samples >= next_sample_threshold:
        return True
    return False


def _phase_summary_record(epoch: int, phase: str, metrics: dict[str, float], lr: float) -> dict[str, Any]:
    record = {
        "event": "phase_summary",
        "epoch": epoch + 1,
        "phase": phase,
        "loss": metrics["loss"],
        "acc": metrics["acc"],
        "num_samples": metrics["num_samples"],
        "lr": lr,
    }
    for key in ("loss_main", "loss_aux"):
        if key in metrics:
            record[key] = metrics[key]
    for key in (
        "grad_norm_preclip_mean",
        "grad_norm_preclip_max",
        "grad_norm_postclip_mean",
        "grad_clip_frac",
        "grad_nonfinite_batches",
    ):
        if key in metrics:
            record[key] = metrics[key]
    return record


def train_one_epoch(
    *,
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    config: RecipeConfig,
    ctx: DistributedContext,
    epoch: int,
    global_step: int,
    progress_path: Path,
) -> tuple[dict[str, float], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    running_loss_main = 0.0
    running_loss_aux = 0.0
    running_corrects = 0.0
    running_samples = 0
    phase_started = time.time()
    last_step_ended = phase_started
    total_batches = len(loader)
    next_batch_threshold = config.logging.progress_every_batches or 0
    next_sample_threshold = config.logging.progress_every_samples or 0
    grad_norm_preclip_sum = 0.0
    grad_norm_postclip_sum = 0.0
    grad_norm_preclip_max = 0.0
    grad_clip_steps = 0
    grad_nonfinite_steps = 0

    for batch_idx, (inputs, labels) in enumerate(loader, start=1):
        data_time_sec = time.time() - last_step_ended

        apply_gold = config.train.train_gold_prob > 0.0 and random.random() < config.train.train_gold_prob
        if apply_gold:
            inputs = apply_gold_protocol(inputs)

        inputs = inputs.to(ctx.device, non_blocking=True)
        labels = labels.to(ctx.device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=config.train.use_amp and ctx.device.type == "cuda"):
            if config.train.aux_h_loss_weight > 0.0:
                logits, aux_logits = model(inputs, return_aux=True)
                raw_loss_main = criterion(logits, labels)
                raw_loss_aux = criterion(aux_logits, labels)
                raw_loss = raw_loss_main + config.train.aux_h_loss_weight * raw_loss_aux
            else:
                logits = model(inputs)
                raw_loss_main = criterion(logits, labels)
                raw_loss_aux = None
                raw_loss = raw_loss_main
            loss = raw_loss / max(config.train.grad_accum_steps, 1)

        preds = logits.argmax(dim=1)
        scaler.scale(loss).backward()

        if batch_idx % max(config.train.grad_accum_steps, 1) == 0 or batch_idx == total_batches:
            if config.train.gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                total_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config.train.gradient_clip_norm,
                    error_if_nonfinite=False,
                )
                total_norm = float(total_norm_tensor.item())
                if math.isfinite(total_norm):
                    grad_norm_preclip_sum += total_norm
                    grad_norm_preclip_max = max(grad_norm_preclip_max, total_norm)
                    grad_norm_postclip_sum += min(total_norm, config.train.gradient_clip_norm)
                    if total_norm > config.train.gradient_clip_norm:
                        grad_clip_steps += 1
                else:
                    grad_nonfinite_steps += 1
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = labels.size(0)
        running_loss += raw_loss.detach().item() * batch_size
        running_loss_main += raw_loss_main.detach().item() * batch_size
        if raw_loss_aux is not None:
            running_loss_aux += raw_loss_aux.detach().item() * batch_size
        running_corrects += (preds == labels).sum().item()
        running_samples += batch_size
        global_step += 1

        step_time_sec = time.time() - last_step_ended
        if ctx.is_main_process and _should_log_progress(
            batch_idx=batch_idx,
            processed_samples=running_samples,
            next_batch_threshold=next_batch_threshold,
            next_sample_threshold=next_sample_threshold,
            config=config,
        ):
            record = {
                "event": "batch_progress",
                "epoch": epoch + 1,
                "phase": "train",
                "batch": batch_idx,
                "batches_total": total_batches,
                "rank": ctx.rank,
                "world_size": ctx.world_size,
                "samples_done_rank": running_samples,
                "approx_global_samples_done": running_samples * ctx.world_size,
                "data_time_sec": round(data_time_sec, 4),
                "step_time_sec": round(step_time_sec, 4),
                "avg_loss_so_far": running_loss / max(running_samples, 1),
                "lr": optimizer.param_groups[0]["lr"],
            }
            record.update(gpu_memory_stats(ctx.device))
            append_jsonl(progress_path, record)
            if config.logging.progress_every_batches > 0:
                while batch_idx >= next_batch_threshold:
                    next_batch_threshold += config.logging.progress_every_batches
            if config.logging.progress_every_samples > 0:
                while running_samples >= next_sample_threshold:
                    next_sample_threshold += config.logging.progress_every_samples
        last_step_ended = time.time()

    loss_sum = reduce_sum(running_loss, ctx)
    correct_sum = reduce_sum(running_corrects, ctx)
    sample_sum = reduce_sum(float(running_samples), ctx)
    metrics = {
        "loss": loss_sum / max(sample_sum, 1.0),
        "loss_main": running_loss_main / max(running_samples, 1),
        "acc": correct_sum / max(sample_sum, 1.0),
        "num_samples": sample_sum,
        "elapsed_sec": time.time() - phase_started,
    }
    if config.train.aux_h_loss_weight > 0.0:
        metrics["loss_aux"] = running_loss_aux / max(running_samples, 1)
    if config.train.gradient_clip_norm is not None:
        step_den = max(math.ceil(total_batches / max(config.train.grad_accum_steps, 1)), 1)
        metrics["grad_norm_preclip_mean"] = grad_norm_preclip_sum / step_den
        metrics["grad_norm_preclip_max"] = grad_norm_preclip_max
        metrics["grad_norm_postclip_mean"] = grad_norm_postclip_sum / step_den
        metrics["grad_clip_frac"] = grad_clip_steps / step_den
        metrics["grad_nonfinite_batches"] = float(grad_nonfinite_steps)
    return metrics, global_step


def evaluate_phase(
    *,
    model: nn.Module,
    loader,
    phase: str,
    criterion: nn.Module,
    config: RecipeConfig,
    ctx: DistributedContext,
    epoch: int,
) -> dict[str, float]:
    eval_model = unwrap_model(model)
    eval_model.eval()

    if phase == "gold":
        random.seed(config.runtime.seed + 20000 + epoch)

    loss_sum = 0.0
    loss_main_sum = 0.0
    loss_aux_sum = 0.0
    correct_sum = 0.0
    sample_sum = 0
    started = time.time()
    with torch.inference_mode():
        for inputs, labels in loader:
            if phase == "gold":
                inputs = apply_gold_protocol(inputs)

            inputs = inputs.to(ctx.device, non_blocking=True)
            labels = labels.to(ctx.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=config.train.use_amp and ctx.device.type == "cuda"):
                if config.train.aux_h_loss_weight > 0.0:
                    logits, aux_logits = eval_model(inputs, return_aux=True)
                    loss_main = criterion(logits, labels)
                    loss_aux = criterion(aux_logits, labels)
                    loss = loss_main + config.train.aux_h_loss_weight * loss_aux
                else:
                    logits = eval_model(inputs)
                    loss_main = criterion(logits, labels)
                    loss_aux = None
                    loss = loss_main
            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            loss_main_sum += loss_main.item() * batch_size
            if loss_aux is not None:
                loss_aux_sum += loss_aux.item() * batch_size
            correct_sum += (logits.argmax(dim=1) == labels).sum().item()
            sample_sum += batch_size
    metrics = {
        "loss": loss_sum / max(sample_sum, 1),
        "loss_main": loss_main_sum / max(sample_sum, 1),
        "acc": correct_sum / max(sample_sum, 1),
        "num_samples": sample_sum,
        "elapsed_sec": time.time() - started,
    }
    if config.train.aux_h_loss_weight > 0.0:
        metrics["loss_aux"] = loss_aux_sum / max(sample_sum, 1)
    return metrics


def run_training(
    *,
    model: nn.Module,
    dataloaders: dict[str, Any],
    train_sampler,
    optimizer: torch.optim.Optimizer,
    scheduler: SmoothedReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    config: RecipeConfig,
    ctx: DistributedContext,
    output_dir: Path,
    start_epoch: int = 0,
    global_step: int = 0,
    history: Optional[dict[str, list[float]]] = None,
    best_val_loss: float = float("inf"),
    best_val_acc: float = 0.0,
) -> dict[str, Any]:
    criterion = nn.CrossEntropyLoss()
    progress_path = output_dir / config.logging.jsonl_filename

    if history is None:
        history = {
            "loss_train": [],
            "loss_main_train": [],
            "acc_train": [],
            "loss_val": [],
            "loss_main_val": [],
            "acc_val": [],
            "loss_gold": [],
            "loss_main_gold": [],
            "acc_gold": [],
        }
        if config.train.aux_h_loss_weight > 0.0:
            history["loss_aux_train"] = []
            history["loss_aux_val"] = []
            history["loss_aux_gold"] = []
        if config.train.gradient_clip_norm is not None:
            history["grad_norm_preclip_mean_train"] = []
            history["grad_norm_preclip_max_train"] = []
            history["grad_norm_postclip_mean_train"] = []
            history["grad_clip_frac_train"] = []
            history["grad_nonfinite_batches_train"] = []
    else:
        history.setdefault("loss_main_train", [])
        history.setdefault("loss_main_val", [])
        history.setdefault("loss_main_gold", [])
        if config.train.aux_h_loss_weight > 0.0:
            history.setdefault("loss_aux_train", [])
            history.setdefault("loss_aux_val", [])
            history.setdefault("loss_aux_gold", [])
        if config.train.gradient_clip_norm is not None:
            history.setdefault("grad_norm_preclip_mean_train", [])
            history.setdefault("grad_norm_preclip_max_train", [])
            history.setdefault("grad_norm_postclip_mean_train", [])
            history.setdefault("grad_clip_frac_train", [])
            history.setdefault("grad_nonfinite_batches_train", [])

    started = time.time()
    for epoch in range(start_epoch, config.train.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_metrics, global_step = train_one_epoch(
            model=model,
            loader=dataloaders["train"],
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            config=config,
            ctx=ctx,
            epoch=epoch,
            global_step=global_step,
            progress_path=progress_path,
        )
        history["loss_train"].append(float(train_metrics["loss"]))
        history["loss_main_train"].append(float(train_metrics["loss_main"]))
        history["acc_train"].append(float(train_metrics["acc"]))
        if config.train.aux_h_loss_weight > 0.0:
            history["loss_aux_train"].append(float(train_metrics["loss_aux"]))
        if config.train.gradient_clip_norm is not None:
            history["grad_norm_preclip_mean_train"].append(float(train_metrics["grad_norm_preclip_mean"]))
            history["grad_norm_preclip_max_train"].append(float(train_metrics["grad_norm_preclip_max"]))
            history["grad_norm_postclip_mean_train"].append(float(train_metrics["grad_norm_postclip_mean"]))
            history["grad_clip_frac_train"].append(float(train_metrics["grad_clip_frac"]))
            history["grad_nonfinite_batches_train"].append(float(train_metrics["grad_nonfinite_batches"]))

        if hasattr(dataloaders["train"].dataset, "on_epoch_end"):
            random.seed(config.runtime.seed + 10000 + epoch)
            dataloaders["train"].dataset.on_epoch_end()

        val_metrics = {"loss": float("nan"), "loss_main": float("nan"), "acc": float("nan"), "num_samples": 0.0}
        gold_metrics = {"loss": float("nan"), "loss_main": float("nan"), "acc": float("nan"), "num_samples": 0.0}

        if ctx.enabled and config.train.eval_on_main_rank_only:
            barrier(ctx)
            if ctx.is_main_process:
                val_metrics = evaluate_phase(
                    model=model,
                    loader=dataloaders["val"],
                    phase="val",
                    criterion=criterion,
                    config=config,
                    ctx=ctx,
                    epoch=epoch,
                )
                gold_metrics = evaluate_phase(
                    model=model,
                    loader=dataloaders["gold"],
                    phase="gold",
                    criterion=criterion,
                    config=config,
                    ctx=ctx,
                    epoch=epoch,
                )
            val_loss_for_scheduler = broadcast_float(val_metrics["loss_main"] if ctx.is_main_process else 0.0, ctx)
            barrier(ctx)
        else:
            val_metrics = evaluate_phase(
                model=model,
                loader=dataloaders["val"],
                phase="val",
                criterion=criterion,
                config=config,
                ctx=ctx,
                epoch=epoch,
            )
            gold_metrics = evaluate_phase(
                model=model,
                loader=dataloaders["gold"],
                phase="gold",
                criterion=criterion,
                config=config,
                ctx=ctx,
                epoch=epoch,
            )
            val_loss_for_scheduler = float(val_metrics["loss_main"])

        scheduler.step(epoch, val_loss_for_scheduler)

        if ctx.is_main_process:
            history["loss_val"].append(float(val_metrics["loss"]))
            history["loss_main_val"].append(float(val_metrics["loss_main"]))
            history["acc_val"].append(float(val_metrics["acc"]))
            history["loss_gold"].append(float(gold_metrics["loss"]))
            history["loss_main_gold"].append(float(gold_metrics["loss_main"]))
            history["acc_gold"].append(float(gold_metrics["acc"]))
            if config.train.aux_h_loss_weight > 0.0:
                history["loss_aux_val"].append(float(val_metrics["loss_aux"]))
                history["loss_aux_gold"].append(float(gold_metrics["loss_aux"]))

            append_jsonl(progress_path, _phase_summary_record(epoch, "train", train_metrics, optimizer.param_groups[0]["lr"]))
            append_jsonl(progress_path, _phase_summary_record(epoch, "val", val_metrics, optimizer.param_groups[0]["lr"]))
            append_jsonl(progress_path, _phase_summary_record(epoch, "gold", gold_metrics, optimizer.param_groups[0]["lr"]))

            is_best = val_metrics["loss_main"] < best_val_loss
            if is_best:
                best_val_loss = float(val_metrics["loss_main"])
                best_val_acc = float(val_metrics["acc"])

            checkpoint_state = build_checkpoint_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                global_step=global_step,
                history=history,
                best_val_loss=best_val_loss,
                best_val_acc=best_val_acc,
                config=config,
                ctx=ctx,
                output_dir=output_dir,
            )
            if config.checkpointing.save_every_epoch or (config.checkpointing.save_best and is_best):
                saved_paths = save_checkpoints(
                    state=checkpoint_state,
                    output_dir=output_dir,
                    prefix=config.checkpointing.checkpoint_prefix,
                    save_best=config.checkpointing.save_best,
                    is_best=is_best,
                )
                append_jsonl(
                    progress_path,
                    {
                        "event": "checkpoint_saved",
                        "epoch": epoch + 1,
                        "paths": saved_paths,
                    },
                )

            epoch_summary = {
                "event": "epoch_summary",
                "epoch": epoch + 1,
                "lr": optimizer.param_groups[0]["lr"],
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "elapsed_avg_sec": (time.time() - started) / max(epoch + 1 - start_epoch, 1),
                "rank": ctx.rank,
                "world_size": ctx.world_size,
                "device": str(ctx.device),
            }
            epoch_summary.update(gpu_memory_stats(ctx.device))
            append_jsonl(progress_path, epoch_summary)

        barrier(ctx)

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "global_step": global_step,
        "elapsed_sec": time.time() - started,
    }
