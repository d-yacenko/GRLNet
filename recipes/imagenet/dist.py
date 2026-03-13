from __future__ import annotations

import os
from datetime import timedelta
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def _resolve_single_process_device(device_name: Optional[str]) -> torch.device:
    if device_name is None:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda:0")
    return device


def init_distributed(device_name: Optional[str], backend: str, timeout_minutes: int = 60) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if device_name is not None and device_name != "cuda":
            raise ValueError("DDP recipe currently expects CUDA device mode")
        if not torch.cuda.is_available():
            raise RuntimeError("DDP recipe requested but CUDA is not available")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=timeout_minutes),
        )
        return DistributedContext(
            enabled=True,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
        )

    device = _resolve_single_process_device(device_name)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(device.index if device.index is not None else 0)
    return DistributedContext(
        enabled=False,
        rank=0,
        local_rank=0,
        world_size=1,
        device=device,
    )


def wrap_model(
    model: torch.nn.Module,
    ctx: DistributedContext,
    *,
    broadcast_buffers: bool = False,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    if not ctx.enabled:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
        output_device=ctx.local_rank if ctx.device.type == "cuda" else None,
        broadcast_buffers=broadcast_buffers,
        find_unused_parameters=find_unused_parameters,
    )


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def barrier(ctx: DistributedContext) -> None:
    if ctx.enabled:
        if ctx.device.type == "cuda":
            dist.barrier(device_ids=[ctx.local_rank])
        else:
            dist.barrier()


def destroy_distributed(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


def broadcast_float(value: float, ctx: DistributedContext) -> float:
    if not ctx.enabled:
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=ctx.device)
    dist.broadcast(tensor, src=0)
    return float(tensor.item())


def reduce_sum(value: float, ctx: DistributedContext) -> float:
    if not ctx.enabled:
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=ctx.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def gpu_memory_stats(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "gpu_mem_total_mb": None,
            "gpu_mem_used_mb": None,
            "gpu_mem_free_mb": None,
            "gpu_mem_allocated_mb": None,
            "gpu_mem_reserved_mb": None,
            "gpu_mem_max_allocated_mb": None,
        }

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "gpu_mem_total_mb": round(total_bytes / (1024**2), 3),
        "gpu_mem_used_mb": round((total_bytes - free_bytes) / (1024**2), 3),
        "gpu_mem_free_mb": round(free_bytes / (1024**2), 3),
        "gpu_mem_allocated_mb": round(torch.cuda.memory_allocated(device) / (1024**2), 3),
        "gpu_mem_reserved_mb": round(torch.cuda.memory_reserved(device) / (1024**2), 3),
        "gpu_mem_max_allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 3),
    }
