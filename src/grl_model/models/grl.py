from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn


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
        # Канонический c-head readout фиксирован на tanh.
        self.c_head_activation_name = "tanh"
        self.c_head_activation = _build_activation(self.c_head_activation_name)

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
        for hidden in self.hidden_channels:
            in_dim = hidden * global_pool * global_pool
            if in_dim == self.head_dim:
                self.c_readout_projections.append(nn.Identity())
            else:
                self.c_readout_projections.append(nn.Linear(in_dim, self.head_dim, bias=False))
        self.fusion_input_dim = self.head_dim * 2
        self.fc = nn.Linear(self.fusion_input_dim, num_classes)
        self.aux_fc = nn.Linear(self.head_dim, num_classes) if self.aux_h_supervision else None

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

    def forward_features(self, x_seq: Tensor) -> Tensor:
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

        pooled_h = self.head_pool(h[-1]).flatten(1)
        c_head = torch.zeros_like(pooled_h)
        for c_state, proj in zip(c, self.c_readout_projections):
            # Мягкая нелинейность на c-head делает shortcut в голову менее "бесплатным".
            pooled_c = self.head_pool(self.c_head_activation(c_state)).flatten(1)
            c_head = c_head + proj(pooled_c)

        # Конкатенация оставляет h- и c-ветки раздельными до последнего classifier.
        return torch.cat([pooled_h, c_head], dim=1)

    def split_features(self, features: Tensor) -> tuple[Tensor, Tensor]:
        return features[:, : self.head_dim], features[:, self.head_dim :]

    def forward(self, x: Tensor, *, return_aux: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        features = self.forward_features(x)
        logits = self.fc(features)
        if not return_aux:
            return logits
        if self.aux_fc is None:
            raise RuntimeError("Auxiliary h-head is disabled for this GRLClassifier instance")
        h_branch, _ = self.split_features(features)
        aux_logits = self.aux_fc(h_branch)
        return logits, aux_logits
