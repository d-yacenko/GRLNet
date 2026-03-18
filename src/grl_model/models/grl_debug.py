from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from torch import Tensor, nn


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
        self.c_head_activation_name = "tanh"
        self.c_head_activation = _build_activation(self.c_head_activation_name)

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
        for hidden in self.hidden_channels:
            in_dim = hidden * global_pool * global_pool
            if in_dim == self.head_dim:
                self.c_readout_projections.append(nn.Identity())
            else:
                self.c_readout_projections.append(nn.Linear(in_dim, self.head_dim, bias=False))
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
        )
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

    def forward_features(self, x_seq: Tensor) -> Tensor:
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

        pooled_h = self.head_pool(h[-1]).flatten(1)
        self._capture("head/pooled_h", pooled_h)

        c_head = torch.zeros_like(pooled_h)
        for layer_idx, (c_state, proj) in enumerate(zip(c, self.c_readout_projections)):
            c_head_input = self.c_head_activation(c_state)
            self._capture(f"head/c_head_act/layer_{layer_idx}", c_head_input)
            pooled_c = self.head_pool(c_head_input).flatten(1)
            self._capture(f"head/c_pooled/layer_{layer_idx}", pooled_c)
            c_proj = proj(pooled_c)
            self._capture(f"head/c_proj/layer_{layer_idx}", c_proj)
            c_head = c_head + c_proj
        self._capture("head/c_residual_sum", c_head)

        self._capture("head/h_branch", pooled_h)
        self._capture("head/c_branch", c_head)
        fusion_input = torch.cat([pooled_h, c_head], dim=1)
        self._capture("head/fusion_input", fusion_input)
        return fusion_input

    def forward(self, x: Tensor) -> Tensor:
        features = self.forward_features(x)
        logits = self.fc(features)
        self._capture("head/logits", logits)
        return logits

    def forward_with_aux(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.forward_features(x)
        logits = self.fc(features)
        self._capture("head/logits", logits)
        if self.aux_fc is None:
            raise RuntimeError("Auxiliary h-head is disabled for this GRLDebugClassifier instance")
        h_branch = features[:, : self.head_dim]
        aux_logits = self.aux_fc(h_branch)
        self._capture("head/aux_logits", aux_logits)
        return logits, aux_logits
