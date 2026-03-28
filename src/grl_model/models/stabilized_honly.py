from __future__ import annotations

from typing import Literal

import torch
from torch import nn


def choose_groups(channels: int, *, max_groups: int = 32) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = nn.GroupNorm(choose_groups(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class StabilizedHOnlyRecurrentCell(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 3,
        forget_bias: float = 1.0,
        hidden_scale_init: float = -1.75,
        delta_scale_init: float = -2.75,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        groups = choose_groups(channels)
        self.forget_bias = float(forget_bias)
        self.gate_norm = nn.GroupNorm(groups, channels)
        self.gate_act = nn.SiLU(inplace=True)
        self.gate_conv = nn.Conv2d(channels, channels * 4, kernel_size=kernel_size, padding=padding, bias=False)
        self.c_norm = nn.GroupNorm(groups, channels)
        self.delta = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
        )
        self.hidden_scale = nn.Parameter(torch.tensor(float(hidden_scale_init)))
        self.delta_scale = nn.Parameter(torch.tensor(float(delta_scale_init)))

    def forward(self, h_prev: torch.Tensor, c_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gates = self.gate_conv(self.gate_act(self.gate_norm(h_prev)))
        i_raw, f_raw, o_raw, g_raw = gates.chunk(4, dim=1)
        i = torch.sigmoid(i_raw)
        f = torch.sigmoid(f_raw + self.forget_bias)
        o = torch.sigmoid(o_raw)
        g = torch.tanh(g_raw)

        c = f * c_prev + i * g
        hidden = o * torch.tanh(self.c_norm(c))
        delta = self.delta(hidden)
        h = (
            h_prev
            + torch.sigmoid(self.hidden_scale) * hidden
            + torch.sigmoid(self.delta_scale) * delta
        )
        return h, c


class StabilizedHOnlyRecurrentClassifier(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        stem_channels: int = 64,
        hidden_channels: int = 192,
        steps: int = 12,
        kernel_size: int = 3,
        forget_bias: float = 1.0,
        aux_steps: int = 3,
        aux_hidden_dim: int = 256,
        main_dropout: float = 0.25,
        aux_dropout: float = 0.15,
        hidden_scale_init: float = -1.75,
        delta_scale_init: float = -2.75,
        readout_mode: Literal["hc"] = "hc",
    ) -> None:
        super().__init__()
        self.steps = int(steps)
        self.hidden_channels = int(hidden_channels)
        self.aux_steps = max(0, min(int(aux_steps), self.steps))
        self.readout_mode = str(readout_mode)
        self.num_classes = int(num_classes)

        self.stem = nn.Sequential(
            ConvGNAct(3, stem_channels, kernel_size=5, stride=2),
            ConvGNAct(stem_channels, stem_channels, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvGNAct(stem_channels, stem_channels, kernel_size=3, stride=1),
        )
        self.h_seed = nn.Sequential(
            nn.Conv2d(
                stem_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.GroupNorm(choose_groups(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.cell = StabilizedHOnlyRecurrentCell(
            hidden_channels,
            kernel_size=kernel_size,
            forget_bias=forget_bias,
            hidden_scale_init=hidden_scale_init,
            delta_scale_init=delta_scale_init,
        )

        readout_dim = hidden_channels * 2
        self.main_head = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, 512),
            nn.GELU(),
            nn.Dropout(float(main_dropout)),
            nn.Linear(512, num_classes),
        )
        self.aux_head = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, aux_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(aux_dropout)),
            nn.Linear(aux_hidden_dim, num_classes),
        )

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(torch.mean(x, dim=(-2, -1)), start_dim=1)

    def _step_readout(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return torch.cat([self._pool(h), self._pool(c)], dim=1)

    def reset_classifier(self, num_classes: int) -> None:
        readout_dim = self.hidden_channels * 2
        aux_hidden_dim = int(self.aux_head[1].out_features)
        main_dropout = float(self.main_head[3].p)
        aux_dropout = float(self.aux_head[3].p)
        self.main_head = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, 512),
            nn.GELU(),
            nn.Dropout(main_dropout),
            nn.Linear(512, num_classes),
        )
        self.aux_head = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, aux_hidden_dim),
            nn.GELU(),
            nn.Dropout(aux_dropout),
            nn.Linear(aux_hidden_dim, num_classes),
        )
        self.num_classes = int(num_classes)

    def forward(self, x: torch.Tensor, *, return_aux: bool = False):
        stem = self.stem(x)
        h = self.h_seed(stem)
        c = torch.zeros_like(h)

        late_readouts: list[torch.Tensor] = []
        late_start = self.steps - self.aux_steps
        for step_idx in range(self.steps):
            h, c = self.cell(h, c)
            if step_idx >= late_start:
                late_readouts.append(self._step_readout(h, c))

        main_logits = self.main_head(late_readouts[-1])
        if not return_aux:
            return main_logits

        aux_logits = [self.aux_head(readout) for readout in late_readouts[:-1]]
        return main_logits, aux_logits
