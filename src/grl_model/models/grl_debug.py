from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _tensor_stats(tensor: Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    float_view = detached.float()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "mean": float(float_view.mean().item()),
        "std": float(float_view.std(unbiased=False).item()) if detached.numel() > 1 else 0.0,
        "min": float(float_view.min().item()),
        "max": float(float_view.max().item()),
        "mean_abs": float(float_view.abs().mean().item()),
        "max_abs": float(float_view.abs().max().item()),
    }


class DebugTraceStore:
    """Store forward tensors and their backward gradients for one debug run."""

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.forward_tensors: dict[str, Tensor] = {}
        self.forward_stats: dict[str, dict[str, Any]] = {}
        self.gradient_tensors: dict[str, Tensor] = {}
        self.gradient_stats: dict[str, dict[str, Any]] = {}

    def capture(self, name: str, tensor: Tensor) -> Tensor:
        detached = tensor.detach().cpu().clone()
        self.forward_tensors[name] = detached
        self.forward_stats[name] = _tensor_stats(detached)

        if tensor.requires_grad:
            def _hook(grad: Tensor, key: str = name) -> None:
                grad_cpu = grad.detach().cpu().clone()
                self.gradient_tensors[key] = grad_cpu
                self.gradient_stats[key] = _tensor_stats(grad_cpu)

            tensor.register_hook(_hook)
        return tensor

    def snapshot(self, *, extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return {
            "forward": self.forward_tensors,
            "forward_stats": self.forward_stats,
            "gradients": self.gradient_tensors,
            "gradient_stats": self.gradient_stats,
            "extra": extra or {},
        }

    def save(self, path: str | Path, *, extra: Optional[dict[str, Any]] = None) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.snapshot(extra=extra), output_path)
        return output_path


class DebugConvLSTMCell(nn.Module):
    """Debug copy of ConvLSTMCell with the same parameterization as the canonical GRL."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3, bias: bool = True) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        self.forget_bias = nn.Parameter(torch.ones(1, hidden_channels, 1, 1) * 0)

    def forward(self, x: Tensor, hidden: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        h_prev, c_prev = hidden
        combined = torch.cat([x, h_prev], dim=1)
        conv_out = self.conv(combined)

        cc_i, cc_f, cc_g, cc_o = torch.split(conv_out, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f + self.forget_bias)
        g = torch.tanh(cc_g)
        o = torch.sigmoid(cc_o)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


def _pool2_size(size: int) -> int:
    return (size - 2) // 2 + 1


def _build_activation(name: str) -> nn.Module:
    key = name.strip().lower()
    if key in {"identity", "none"}:
        return nn.Identity()
    if key == "tanh":
        return nn.Tanh()
    if key == "relu":
        return nn.ReLU()
    if key in {"leaky_relu", "lrelu"}:
        return nn.LeakyReLU(negative_slope=0.1)
    raise ValueError(f"Unsupported activation: {name!r}")


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _build_upward_norm(channels: int) -> nn.Module:
    return nn.GroupNorm(_group_count(channels), channels)


class AvgPoolProjectReadoutBranch(nn.Module):
    def __init__(self, *, in_channels: int, head_dim: int, pool_size: int) -> None:
        super().__init__()
        self.pool_size = int(pool_size)
        self.pool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        in_dim = int(in_channels) * self.pool_size * self.pool_size
        if in_dim == head_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(in_dim, int(head_dim), bias=False)

    def forward_with_intermediates(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        reduced = x
        pooled = self.pool(reduced).flatten(1)
        proj = self.proj(pooled)
        return reduced, pooled, proj

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_intermediates(x)[-1]


class ConvDownsampleProjectReadoutBranch(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        head_dim: int,
        squeeze_channels: int,
        target_map_size: int,
        pool_size: int,
        max_downsample_steps: int,
    ) -> None:
        super().__init__()
        squeeze_channels = int(squeeze_channels)
        if squeeze_channels <= 0:
            raise ValueError("squeeze_channels must be positive")
        self.target_map_size = int(target_map_size)
        self.pool_size = int(pool_size)
        self.squeeze = nn.Conv2d(int(in_channels), squeeze_channels, kernel_size=1, bias=False)
        self.squeeze_norm = _build_upward_norm(squeeze_channels)
        self.squeeze_activation = nn.LeakyReLU(negative_slope=0.1)
        self.down_blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(squeeze_channels, squeeze_channels, kernel_size=3, stride=2, padding=1, bias=False),
                _build_upward_norm(squeeze_channels),
                nn.LeakyReLU(negative_slope=0.1),
            )
            for _ in range(int(max_downsample_steps))
        )
        self.pool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        in_dim = squeeze_channels * self.pool_size * self.pool_size
        if in_dim == head_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(in_dim, int(head_dim), bias=False)

    def _reduce_spatial(self, x: Tensor) -> Tensor:
        x = self.squeeze_activation(self.squeeze_norm(self.squeeze(x)))
        block_idx = 0
        while min(x.shape[-2], x.shape[-1]) > self.target_map_size and block_idx < len(self.down_blocks):
            x = self.down_blocks[block_idx](x)
            block_idx += 1
        if min(x.shape[-2], x.shape[-1]) > self.target_map_size:
            x = F.adaptive_avg_pool2d(x, (self.target_map_size, self.target_map_size))
        return x

    def forward_with_intermediates(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        reduced = self._reduce_spatial(x)
        pooled = self.pool(reduced).flatten(1)
        proj = self.proj(pooled)
        return reduced, pooled, proj

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_intermediates(x)[-1]


class GRLDebugClassifier(nn.Module):
    """Debug copy of the canonical GRL model with tensor/gradient capture."""

    def __init__(
        self,
        *,
        num_classes: int = 1000,
        in_channels: int = 3,
        hidden_channels: Iterable[int] = (24, 32, 32, 48, 48, 64, 64, 96, 160),
        kernel_size: int = 3,
        global_pool: int = 2,
        track_length: int = 10,
        pool_after_layers: Optional[Iterable[int]] = None,
        aux_h_supervision: bool = False,
        c_head_activation: str = "tanh",
        c_readout_mode: str = "avgpool_proj",
        c_readout_channels: int = 16,
        c_readout_target_map: int = 7,
        c_readout_pool: int = 4,
        c_readout_max_downsamples: int = 8,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.seq_len_train = int(track_length)
        self.global_pool = int(global_pool)
        self.aux_h_supervision = bool(aux_h_supervision)
        self.hidden_channels = tuple(int(v) for v in hidden_channels)
        self.num_layers = len(self.hidden_channels)
        if pool_after_layers is None:
            self.pool_after_layers = (0, 2, 4, 6, 7)
        else:
            self.pool_after_layers = tuple(sorted(int(v) for v in pool_after_layers))
        self.upward_activation_name = "lrelu"
        self.upward_activation = _build_activation(self.upward_activation_name)
        self.upward_norm_name = "group"
        self.c_head_activation_name = c_head_activation.strip().lower()
        self.c_head_activation = _build_activation(self.c_head_activation_name)
        self.c_readout_mode_name = c_readout_mode.strip().lower()
        self.c_readout_channels = int(c_readout_channels)
        self.c_readout_target_map = int(c_readout_target_map)
        self.c_readout_pool = int(c_readout_pool)
        self.c_readout_max_downsamples = int(c_readout_max_downsamples)
        self.fusion_head_enabled = True

        prev_channels = in_channels
        self.cells = nn.ModuleList()
        for hidden in self.hidden_channels:
            self.cells.append(DebugConvLSTMCell(prev_channels, hidden, kernel_size))
            prev_channels = hidden
        self.upward_norms = nn.ModuleList(_build_upward_norm(hidden) for hidden in self.hidden_channels)

        self.maxpool = nn.MaxPool2d(2)
        self.head_pool = nn.AdaptiveAvgPool2d((global_pool, global_pool))
        self.head_dim = self.hidden_channels[-1] * global_pool * global_pool
        self.c_readout_projections = nn.ModuleList()
        self.c_readout_branches = nn.ModuleList()
        if self.c_readout_mode_name == "avgpool_proj":
            for hidden in self.hidden_channels:
                in_dim = hidden * global_pool * global_pool
                if in_dim == self.head_dim:
                    self.c_readout_projections.append(nn.Identity())
                else:
                    self.c_readout_projections.append(nn.Linear(in_dim, self.head_dim, bias=False))
        elif self.c_readout_mode_name == "conv_downsample":
            for hidden in self.hidden_channels:
                self.c_readout_branches.append(
                    ConvDownsampleProjectReadoutBranch(
                        in_channels=hidden,
                        head_dim=self.head_dim,
                        squeeze_channels=self.c_readout_channels,
                        target_map_size=self.c_readout_target_map,
                        pool_size=self.c_readout_pool,
                        max_downsample_steps=self.c_readout_max_downsamples,
                    )
                )
        else:
            raise ValueError(f"Unsupported c_readout_mode: {c_readout_mode!r}")
        self.fusion_input_dim = self.head_dim * 2
        self.fc = nn.Linear(self.fusion_input_dim, num_classes)
        self.aux_fc = nn.Linear(self.head_dim, num_classes) if self.aux_h_supervision else None

        self.trace = DebugTraceStore()
        self.debug_enabled = True
        self.capture_full_tensors = True
        self.last_debug_extra: dict[str, Any] = {}

    @staticmethod
    def _require_track_batch(x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected canonical input [B, T, C, H, W], got {tuple(x.shape)}")
        return x

    @classmethod
    def from_grl(cls, model: nn.Module) -> "GRLDebugClassifier":
        hidden_channels = tuple(getattr(model, "hidden_channels"))
        debug_model = cls(
            num_classes=int(getattr(model, "num_classes")),
            hidden_channels=hidden_channels,
            global_pool=int(getattr(model, "global_pool")),
            track_length=int(getattr(model, "seq_len_train")),
            pool_after_layers=tuple(getattr(model, "pool_after_layers", (0, 2, 4, 6, 7))),
            aux_h_supervision=bool(getattr(model, "aux_h_supervision", False)),
            c_head_activation=str(getattr(model, "c_head_activation_name", "tanh")),
            c_readout_mode=str(getattr(model, "c_readout_mode_name", "avgpool_proj")),
            c_readout_channels=int(getattr(model, "c_readout_channels", 16)),
            c_readout_target_map=int(getattr(model, "c_readout_target_map", 7)),
            c_readout_pool=int(getattr(model, "c_readout_pool", 4)),
            c_readout_max_downsamples=int(getattr(model, "c_readout_max_downsamples", 8)),
        )
        if not bool(getattr(model, "fusion_head_enabled", True)):
            debug_model.disable_fusion_head()
        debug_model.load_state_dict(model.state_dict(), strict=True)
        return debug_model

    def set_debug(self, enabled: bool = True, *, capture_full_tensors: bool = True) -> None:
        self.debug_enabled = enabled
        self.capture_full_tensors = capture_full_tensors

    def clear_debug_trace(self) -> None:
        self.trace.clear()
        self.last_debug_extra = {}

    def save_debug_trace(self, path: str | Path, *, extra: Optional[dict[str, Any]] = None) -> Path:
        payload_extra = dict(self.last_debug_extra)
        if extra:
            payload_extra.update(extra)
        return self.trace.save(path, extra=payload_extra)

    def _capture(self, name: str, tensor: Tensor) -> Tensor:
        if not self.debug_enabled:
            return tensor
        if self.capture_full_tensors:
            return self.trace.capture(name, tensor)

        detached = tensor.detach().cpu()
        self.trace.forward_stats[name] = _tensor_stats(detached)
        if tensor.requires_grad:
            def _hook(grad: Tensor, key: str = name) -> None:
                grad_cpu = grad.detach().cpu()
                self.trace.gradient_stats[key] = _tensor_stats(grad_cpu)

            tensor.register_hook(_hook)
        return tensor

    def _layer_hw(self, height: int, width: int) -> list[tuple[int, int]]:
        sizes: list[tuple[int, int]] = []
        h_size = height
        w_size = width
        for layer_idx in range(self.num_layers):
            sizes.append((h_size, w_size))
            if layer_idx in self.pool_after_layers:
                h_size = _pool2_size(h_size)
                w_size = _pool2_size(w_size)
        return sizes

    def _readout_c_layer(self, c_state: Tensor, layer_idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        c_head_input = self.c_head_activation(c_state)
        if self.c_readout_mode_name == "avgpool_proj":
            reduced = c_head_input
            pooled = self.head_pool(reduced).flatten(1)
            proj = self.c_readout_projections[layer_idx](pooled)
        else:
            reduced, pooled, proj = self.c_readout_branches[layer_idx].forward_with_intermediates(c_head_input)
        return c_head_input, reduced, pooled, proj

    def _compose_h_branch(self, h: list[Tensor], *, prefix: str) -> Tensor:
        pooled_h = self.head_pool(h[-1]).flatten(1)
        self._capture(f"{prefix}/pooled_h", pooled_h)
        self._capture(f"{prefix}/h_branch", pooled_h)
        return pooled_h

    def _compose_c_readout(
        self,
        c: list[Tensor],
        *,
        prefix: str,
        like: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        c_proj_per_layer: list[Tensor] = []
        c_head = torch.zeros_like(like)
        for layer_idx, c_state in enumerate(c):
            c_head_input, reduced_c, pooled_c, c_proj = self._readout_c_layer(c_state, layer_idx)
            self._capture(f"{prefix}/c_head_act/layer_{layer_idx}", c_head_input)
            self._capture(f"{prefix}/c_reduced/layer_{layer_idx}", reduced_c)
            self._capture(f"{prefix}/c_pooled/layer_{layer_idx}", pooled_c)
            self._capture(f"{prefix}/c_proj/layer_{layer_idx}", c_proj)
            c_proj_per_layer.append(c_proj)
            c_head = c_head + c_proj
        self._capture(f"{prefix}/c_residual_sum", c_head)
        self._capture(f"{prefix}/c_branch", c_head)
        return c_head, tuple(c_proj_per_layer)

    def _compose_c_proj_per_layer(
        self,
        c: list[Tensor],
        *,
        prefix: str,
    ) -> tuple[Tensor, ...]:
        c_proj_per_layer: list[Tensor] = []
        for layer_idx, c_state in enumerate(c):
            c_head_input, reduced_c, pooled_c, c_proj = self._readout_c_layer(c_state, layer_idx)
            self._capture(f"{prefix}/c_head_act/layer_{layer_idx}", c_head_input)
            self._capture(f"{prefix}/c_reduced/layer_{layer_idx}", reduced_c)
            self._capture(f"{prefix}/c_pooled/layer_{layer_idx}", pooled_c)
            self._capture(f"{prefix}/c_proj/layer_{layer_idx}", c_proj)
            c_proj_per_layer.append(c_proj)
        return tuple(c_proj_per_layer)

    def _compose_readout_state(
        self,
        h: list[Tensor],
        c: list[Tensor],
        *,
        prefix: str,
    ) -> dict[str, Tensor | tuple[Tensor, ...]]:
        pooled_h = self._compose_h_branch(h, prefix=prefix)
        c_head, c_proj_per_layer = self._compose_c_readout(c, prefix=prefix, like=pooled_h)
        readout_state: dict[str, Tensor | tuple[Tensor, ...]] = {
            "h_branch": pooled_h,
            "c_branch": c_head,
            "c_proj_per_layer": c_proj_per_layer,
        }
        fusion_input = self.fuse_readout_state(readout_state, prefix=prefix)
        self._capture(f"{prefix}/fusion_input", fusion_input)
        return readout_state

    def fuse_readout_state(
        self,
        readout_state: dict[str, Tensor | tuple[Tensor, ...]],
        *,
        prefix: str = "head",
    ) -> Tensor:
        del prefix  # reserved for future branch-specific capture / пока не используется
        h_branch = readout_state["h_branch"]
        c_branch = readout_state["c_branch"]
        if not isinstance(h_branch, Tensor) or not isinstance(c_branch, Tensor):
            raise TypeError("readout_state must contain tensor branches 'h_branch' and 'c_branch'")
        return torch.cat([h_branch, c_branch], dim=1)

    def _run_recurrence_states(
        self,
        x_seq: Tensor,
    ) -> tuple[list[Tensor], list[Tensor]]:
        x_seq = self._require_track_batch(x_seq)
        batch_size, _, _, height, width = x_seq.shape
        layer_sizes = self._layer_hw(height, width)
        self.last_debug_extra = {
            "track_shape": list(x_seq.shape),
            "active_track_length": int(self.seq_len_train),
            "full_track_length": int(x_seq.shape[1]),
            "num_layers": int(self.num_layers),
            "hidden_channels": list(self.hidden_channels),
            "pool_after_layers": list(self.pool_after_layers),
            "upward_activation": self.upward_activation_name,
            "upward_norm": self.upward_norm_name,
            "c_head_activation": self.c_head_activation_name,
            "c_readout_mode": self.c_readout_mode_name,
            "c_readout_channels": int(self.c_readout_channels),
            "c_readout_target_map": int(self.c_readout_target_map),
            "c_readout_pool": int(self.c_readout_pool),
            "fusion_input_dim": int(self.fusion_input_dim),
            "layer_hw": [list(hw) for hw in layer_sizes],
        }
        self._capture("inputs/track", x_seq)

        h: list[Tensor] = []
        c: list[Tensor] = []
        for layer_idx, (cell, (h_size, w_size)) in enumerate(zip(self.cells, layer_sizes)):
            h_state = x_seq.new_zeros((batch_size, cell.hidden_channels, h_size, w_size))
            c_state = torch.zeros_like(h_state)
            h.append(h_state)
            c.append(c_state)
            self._capture(f"layer_{layer_idx}/init_h", h_state)
            self._capture(f"layer_{layer_idx}/init_c", c_state)

        for t in range(x_seq.shape[1]):
            x = x_seq[:, t]
            self._capture(f"time_{t}/input_frame", x)
            for layer_idx, cell in enumerate(self.cells):
                self._capture(f"layer_{layer_idx}/time_{t}/x_in", x)
                self._capture(f"layer_{layer_idx}/time_{t}/h_prev", h[layer_idx])
                self._capture(f"layer_{layer_idx}/time_{t}/c_prev", c[layer_idx])
                h[layer_idx], c[layer_idx] = cell(x, (h[layer_idx], c[layer_idx]))
                self._capture(f"layer_{layer_idx}/time_{t}/h_out", h[layer_idx])
                self._capture(f"layer_{layer_idx}/time_{t}/c_out", c[layer_idx])
                x = self.upward_activation(h[layer_idx])
                self._capture(f"layer_{layer_idx}/time_{t}/upward_act", x)
                x = self.upward_norms[layer_idx](x)
                self._capture(f"layer_{layer_idx}/time_{t}/upward_out", x)
                if layer_idx in self.pool_after_layers:
                    x = self.maxpool(x)
                self._capture(f"layer_{layer_idx}/time_{t}/x_to_next", x)
        return h, c

    def _run_recurrence(
        self,
        x_seq: Tensor,
        *,
        return_readout_state: bool = False,
    ) -> Tensor | dict[str, Tensor | tuple[Tensor, ...]]:
        h, c = self._run_recurrence_states(x_seq)
        readout_state = self._compose_readout_state(h, c, prefix="head")
        if return_readout_state:
            return readout_state
        return self.fuse_readout_state(readout_state, prefix="head")

    def forward_features(self, x_seq: Tensor) -> Tensor:
        return self._run_recurrence(x_seq)

    def forward_readout_state(self, x_seq: Tensor) -> dict[str, Tensor | tuple[Tensor, ...]]:
        readout_state = self._run_recurrence(x_seq, return_readout_state=True)
        if not isinstance(readout_state, dict):
            raise TypeError("forward_readout_state expected a readout-state dict from _run_recurrence")
        return readout_state

    def forward_h_branch(self, x_seq: Tensor) -> Tensor:
        h, _ = self._run_recurrence_states(x_seq)
        return self._compose_h_branch(h, prefix="head")

    def forward_h_and_c_proj(self, x_seq: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        h, c = self._run_recurrence_states(x_seq)
        h_branch = self._compose_h_branch(h, prefix="head")
        c_proj_per_layer = self._compose_c_proj_per_layer(c, prefix="head")
        return h_branch, c_proj_per_layer

    def disable_fusion_head(self) -> None:
        self.fusion_head_enabled = False
        self.fc = nn.Identity()

    def classify_readout_state(self, readout_state: dict[str, Tensor | tuple[Tensor, ...]]) -> Tensor:
        logits = self.fc(self.fuse_readout_state(readout_state, prefix="head"))
        self._capture("head/logits", logits)
        return logits

    def forward(self, x: Tensor) -> Tensor:
        readout_state = self.forward_readout_state(x)
        return self.classify_readout_state(readout_state)

    def forward_with_aux(self, x: Tensor) -> tuple[Tensor, Tensor]:
        readout_state = self.forward_readout_state(x)
        logits = self.classify_readout_state(readout_state)
        if self.aux_fc is None:
            raise RuntimeError("Auxiliary h-head is disabled for this GRLDebugClassifier instance")
        h_branch = readout_state["h_branch"]
        if not isinstance(h_branch, Tensor):
            raise TypeError("readout_state['h_branch'] must be a tensor")
        aux_logits = self.aux_fc(h_branch)
        self._capture("head/aux_logits", aux_logits)
        return logits, aux_logits
