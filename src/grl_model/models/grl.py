from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .weights import GRLWeights


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
    """Legacy c-readout: direct adaptive average pooling followed by a linear projection."""

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
    """Gradient-friendly c-readout: local squeeze plus learned strided downsampling before pooling."""

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
        self.squeeze_channels = squeeze_channels
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


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell used by the published GRL family."""

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
        # Этот bias управляет только forget-gate и длиной полезной памяти по c.
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

        # Главная temporal memory-path проходит через c, а не через h.
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class GRLClassifier(nn.Module):
    """Canonical deep GRL model used for the final library surface.

    The current production-oriented GRL keeps the temporal memory in ``c``, propagates
    features upward through a separate ``h``-path, and reads all final ``c`` states
    directly into the classifier through a concat head.
    """

    def __init__(
        self,
        *,
        num_classes: int = 1000,
        in_channels: int = 3,
        hidden_channels: Iterable[int] = (24, 32, 32, 48, 48, 64, 64, 96, 160),
        pool_after_layers: Iterable[int] = (0, 2, 4, 6, 7),
        kernel_size: int = 3,
        global_pool: int = 2,
        track_length: int = 10,
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
        self.pool_after_layers = tuple(sorted(int(v) for v in pool_after_layers))
        # Канонический upward-path фиксирован на LeakyReLU.
        self.upward_activation_name = "lrelu"
        self.upward_activation = _build_activation(self.upward_activation_name)
        self.upward_norm_name = "group"
        # c-head activation is configurable so experiments can trade gradient flow vs. selectivity.
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
            self.cells.append(ConvLSTMCell(prev_channels, hidden, kernel_size))
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

    @classmethod
    def from_weights(
        cls,
        weights: GRLWeights | str,
        *,
        map_location: str | torch.device = "cpu",
        progress: bool = True,
        check_hash: bool = False,
        **override_kwargs,
    ) -> "GRLClassifier":
        resolved = GRLWeights.get(weights) if isinstance(weights, str) else weights
        model_kwargs = dict(resolved.model_kwargs)
        model_kwargs.update(override_kwargs)
        model = cls(**model_kwargs)
        state_dict = resolved.get_state_dict(
            map_location=map_location,
            progress=progress,
            check_hash=check_hash,
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    @staticmethod
    def _require_track_batch(x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected canonical input [B, T, C, H, W], got {tuple(x.shape)}")
        return x

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

    def _compose_h_branch(self, h: list[Tensor]) -> Tensor:
        return self.head_pool(h[-1]).flatten(1)

    def _compose_c_readout(self, c: list[Tensor], *, like: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        c_proj_per_layer: list[Tensor] = []
        c_head = torch.zeros_like(like)
        for layer_idx, c_state in enumerate(c):
            _, _, _, c_proj = self._readout_c_layer(c_state, layer_idx)
            c_proj_per_layer.append(c_proj)
            c_head = c_head + c_proj
        return c_head, tuple(c_proj_per_layer)

    def _compose_c_proj_per_layer(self, c: list[Tensor]) -> tuple[Tensor, ...]:
        c_proj_per_layer: list[Tensor] = []
        for layer_idx, c_state in enumerate(c):
            _, _, _, c_proj = self._readout_c_layer(c_state, layer_idx)
            c_proj_per_layer.append(c_proj)
        return tuple(c_proj_per_layer)

    def _compose_readout_state(self, h: list[Tensor], c: list[Tensor]) -> dict[str, Tensor | tuple[Tensor, ...]]:
        pooled_h = self._compose_h_branch(h)
        c_head, c_proj_per_layer = self._compose_c_readout(c, like=pooled_h)
        return {
            "h_branch": pooled_h,
            "c_branch": c_head,
            "c_proj_per_layer": c_proj_per_layer,
        }

    def fuse_readout_state(self, readout_state: dict[str, Tensor | tuple[Tensor, ...]]) -> Tensor:
        h_branch = readout_state["h_branch"]
        c_branch = readout_state["c_branch"]
        if not isinstance(h_branch, Tensor) or not isinstance(c_branch, Tensor):
            raise TypeError("readout_state must contain tensor branches 'h_branch' and 'c_branch'")
        # Конкатенация оставляет h- и c-ветки раздельными до classifier.
        return torch.cat([h_branch, c_branch], dim=1)

    def _compose_head_features(self, h: list[Tensor], c: list[Tensor]) -> Tensor:
        return self.fuse_readout_state(self._compose_readout_state(h, c))

    def _run_recurrence_states(self, x_seq: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        x_seq = self._require_track_batch(x_seq)
        batch_size, _, _, height, width = x_seq.shape
        layer_sizes = self._layer_hw(height, width)

        h: list[Tensor] = []
        c: list[Tensor] = []
        for cell, (h_size, w_size) in zip(self.cells, layer_sizes):
            h_state = x_seq.new_zeros((batch_size, cell.hidden_channels, h_size, w_size))
            c_state = torch.zeros_like(h_state)
            h.append(h_state)
            c.append(c_state)

        for t in range(x_seq.shape[1]):
            x = x_seq[:, t]
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx], c[layer_idx] = cell(x, (h[layer_idx], c[layer_idx]))
                # Вверх по глубине идёт отдельный feature-path; temporal memory остаётся в c.
                x = self.upward_activation(h[layer_idx])
                # Нормализация после upward-активации выравнивает масштаб x и recurrent h.
                x = self.upward_norms[layer_idx](x)
                if layer_idx in self.pool_after_layers:
                    x = self.maxpool(x)
        return h, c

    def _run_recurrence(self, x_seq: Tensor, *, return_readout_state: bool = False) -> Tensor | dict[str, Tensor | tuple[Tensor, ...]]:
        h, c = self._run_recurrence_states(x_seq)
        readout_state = self._compose_readout_state(h, c)
        if return_readout_state:
            return readout_state
        return self.fuse_readout_state(readout_state)

    def classify_features(self, features: Tensor) -> Tensor:
        return self.fc(features)

    def classify_readout_state(self, readout_state: dict[str, Tensor | tuple[Tensor, ...]]) -> Tensor:
        return self.classify_features(self.fuse_readout_state(readout_state))

    def main_readout_modules(self) -> tuple[nn.Module, ...]:
        if self.c_readout_mode_name == "avgpool_proj":
            return (*self.c_readout_projections, self.fc)
        return (*self.c_readout_branches, self.fc)

    def disable_fusion_head(self) -> None:
        self.fusion_head_enabled = False
        self.fc = nn.Identity()

    def reset_num_classes(self, num_classes: int) -> None:
        num_classes = int(num_classes)
        self.num_classes = num_classes
        if self.fusion_head_enabled:
            self.fc = nn.Linear(self.fusion_input_dim, num_classes)
        else:
            self.fc = nn.Identity()
        if self.aux_fc is not None:
            self.aux_fc = nn.Linear(self.head_dim, num_classes)

    def forward_features(self, x_seq: Tensor) -> Tensor:
        return self._run_recurrence(x_seq)

    def forward_readout_state(self, x_seq: Tensor) -> dict[str, Tensor | tuple[Tensor, ...]]:
        readout_state = self._run_recurrence(x_seq, return_readout_state=True)
        if not isinstance(readout_state, dict):
            raise TypeError("forward_readout_state expected a readout-state dict from _run_recurrence")
        return readout_state

    def forward_h_branch(self, x_seq: Tensor) -> Tensor:
        h, _ = self._run_recurrence_states(x_seq)
        return self._compose_h_branch(h)

    def forward_h_and_c_proj(self, x_seq: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        h, c = self._run_recurrence_states(x_seq)
        return self._compose_h_branch(h), self._compose_c_proj_per_layer(c)

    def split_features(self, features: Tensor) -> tuple[Tensor, Tensor]:
        return features[:, : self.head_dim], features[:, self.head_dim :]

    def forward(self, x: Tensor, *, return_aux: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        features = self.forward_features(x)
        logits = self.classify_features(features)
        if not return_aux:
            return logits
        if self.aux_fc is None:
            raise RuntimeError("Auxiliary h-head is disabled for this GRLClassifier instance")
        h_branch, _ = self.split_features(features)
        aux_logits = self.aux_fc(h_branch)
        return logits, aux_logits
