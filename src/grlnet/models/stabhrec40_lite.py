"""StabHRec40-Lite: depthwise-separable variant of StabHRec40.

Same architectural skeleton as StabHRec40 (one shared cell × 12 unroll steps,
H/C two-stream cell with residual stabilizers, late readout, untying-friendly
weight layout), but with dense 3×3 convolutions inside the recurrent cell
replaced by depthwise-separable pairs (DW3×3 + PW1×1).

Compute / parameter trade-off (cell only, channels=192, hidden×56×56):
  * gate_conv 192→768 (k=3, dense):   1.327 M params, 4.16 GMAC/step
    → DW3×3(192) + PW1×1(192→768):    0.149 M params, 0.47 GMAC/step (~8.9× ↓)
  * delta conv1 192→192 (k=3, dense): 0.332 M params, 1.04 GMAC/step
    → DW3×3 + PW1×1 192→192:          0.039 M params, 0.12 GMAC/step (~8.6× ↓)
  * delta conv2 192→192:              same as conv1

Cell total:        1.99 M → 0.23 M params (~8.7× ↓)
Full model est.:   3.25 M → ~1.5 M params (~2.2× ↓)
GMAC (T=12):        76 → ~8.5 GMAC (~9× ↓)

Outer structure (stem, h-seed, heads, h/c update rules, residual scalars,
auxiliary supervision, readout) is identical to the dense baseline, so
the gradient analysis, untying lemma, and anytime mechanism transfer
unchanged. Only ‖J_h‖, ‖J_δ‖ operator norms differ empirically.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn

from .stabhrec40 import ConvGNAct, choose_groups


class DSConv(nn.Module):
    """Depthwise-separable conv: DW(k×k) + PW(1×1).

    Equivalent to a dense Conv2d(in→out, k=k, padding=k//2) in shape, but with
    ~k²-1+in/out × fewer params and MACs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        bias: bool = False,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pw = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class StabHRec40LiteCell(nn.Module):
    """Depthwise-separable variant of StabHRec40Cell.

    Same gating semantics and residual H-highway as the dense cell;
    only the three k=3 dense convs (gate, delta-1, delta-2) are replaced
    by depthwise-separable pairs.
    """

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
        groups = choose_groups(channels)
        self.forget_bias = float(forget_bias)

        self.gate_norm = nn.GroupNorm(groups, channels)
        self.gate_act = nn.SiLU(inplace=True)
        # Replaces dense Conv2d(channels, channels*4, k=3): produces i,f,o,g
        self.gate_conv = DSConv(channels, channels * 4, kernel_size=kernel_size, bias=False)

        self.c_norm = nn.GroupNorm(groups, channels)
        # Replaces two dense Conv2d(channels, channels, k=3) inside delta branch
        self.delta = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            DSConv(channels, channels, kernel_size=kernel_size, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            DSConv(channels, channels, kernel_size=kernel_size, bias=False),
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
        h = h_prev + torch.sigmoid(self.hidden_scale) * hidden + torch.sigmoid(self.delta_scale) * delta
        return h, c


@dataclass(frozen=True)
class GRLNetLiteConfig:
    num_classes: int = 1000
    stem_channels: int = 64
    hidden_channels: int = 192
    steps: int = 12
    kernel_size: int = 3
    forget_bias: float = 1.0
    hidden_scale_init: float = -1.75
    delta_scale_init: float = -2.75
    aux_steps: int = 3
    aux_hidden_dim: int = 256
    main_dropout: float = 0.25
    aux_dropout: float = 0.15
    readout_mode: Literal["hc"] = "hc"

    def to_kwargs(self) -> dict[str, object]:
        return asdict(self)


class GRLNetLite(nn.Module):
    """StabHRec40-Lite: same skeleton as GRLNet but DS-conv inside the recurrent cell."""

    def __init__(
        self,
        *,
        num_classes: int = 1000,
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
        if readout_mode != "hc":
            raise ValueError("Only readout_mode='hc' is supported.")
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
        self.cell = StabHRec40LiteCell(
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
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}")

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


def grlnet_stabhrec40_lite(*, weights=None, progress: bool = True, **kwargs) -> GRLNetLite:
    """Build the GRLNet/StabHRec40-Lite architecture.

    Parameters mirror :func:`grlnet.grlnet_stabhrec40`. ``weights`` may be a
    ``GRLNetLiteWeights`` entry, a registry name, a checkpoint path, or ``None``.
    Note: until the first v0.4.0 release publishes a trained Lite checkpoint,
    ``weights='DEFAULT'`` will raise — pass ``None`` for an untrained model.
    """

    from .weights import GRLNetLiteWeights, load_checkpoint_state_dict

    if weights is not None and not isinstance(weights, (str, bytes, os.PathLike)):
        kwargs = {**weights.model_kwargs, **kwargs}
    elif isinstance(weights, str) and weights in GRLNetLiteWeights.names():
        resolved = GRLNetLiteWeights.get(weights)
        kwargs = {**resolved.model_kwargs, **kwargs}
        weights = resolved

    model = GRLNetLite(**kwargs)
    if weights is not None:
        state_dict = load_checkpoint_state_dict(weights, progress=progress)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
    return model


def warm_start_from_dense(model_ds: GRLNetLite, state_dict_dense: dict) -> dict:
    """Initialize StabHRec40-Lite weights from a trained dense StabHRec40 checkpoint.

    Decomposition (per dense conv W ∈ R^{out × in × k × k}):
        DW kernel K[c, h, w] = (1/out) Σ_o W[o, c, h, w]   # spatial avg across out channels
        PW matrix M[o, c]    = (1/k²)  Σ_{h,w} W[o, c, h, w]  # DC component (mean spatial)

    This is a non-trivial init that captures both the spatial structure (DW) and
    the channel-mixing average (PW). Not optimal in least-squares sense, but
    empirically gives faster convergence than random init for DS-from-dense
    distillation in vision (cf. MobileNet teacher-student literature).

    Identity copies for non-cell weights (stem, h-seed, heads, scalars, GroupNorms).

    Returns the new state_dict that can be loaded into model_ds.
    """
    new_sd = {}
    src = state_dict_dense
    own = model_ds.state_dict()

    cell_dense_prefix = "cell."
    cell_ds_prefix = "cell."

    # Map dense gate_conv weight (768, 192, 3, 3) → DS gate_conv {dw.weight (192,1,3,3), pw.weight (768,192,1,1)}
    def decompose(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # W: [out, in, k, k]
        out_ch, in_ch, kh, kw = W.shape
        # DW: [in, 1, k, k]
        dw = W.mean(dim=0, keepdim=False).unsqueeze(1)  # [in, 1, k, k]
        # PW: [out, in, 1, 1]  — mean over spatial = DC component
        pw = W.mean(dim=(2, 3), keepdim=False).unsqueeze(-1).unsqueeze(-1)
        return dw, pw

    # Direct copy for these keys
    direct_keys = {
        "stem.0.conv.weight", "stem.0.norm.weight", "stem.0.norm.bias",
        "stem.1.conv.weight", "stem.1.norm.weight", "stem.1.norm.bias",
        "stem.3.conv.weight", "stem.3.norm.weight", "stem.3.norm.bias",
        "h_seed.0.weight", "h_seed.1.weight", "h_seed.1.bias",
        "cell.gate_norm.weight", "cell.gate_norm.bias",
        "cell.c_norm.weight", "cell.c_norm.bias",
        "cell.delta.0.weight", "cell.delta.0.bias",
        "cell.delta.3.weight", "cell.delta.3.bias",
        "cell.hidden_scale", "cell.delta_scale",
        "main_head.0.weight", "main_head.0.bias", "main_head.1.weight", "main_head.1.bias",
        "main_head.4.weight", "main_head.4.bias",
        "aux_head.0.weight", "aux_head.0.bias", "aux_head.1.weight", "aux_head.1.bias",
        "aux_head.4.weight", "aux_head.4.bias",
    }
    for k in direct_keys:
        if k in src and k in own and src[k].shape == own[k].shape:
            new_sd[k] = src[k].clone()

    # Decompose gate_conv
    if "cell.gate_conv.weight" in src:
        W = src["cell.gate_conv.weight"]  # [768, 192, 3, 3]
        dw, pw = decompose(W)
        new_sd["cell.gate_conv.dw.weight"] = dw
        new_sd["cell.gate_conv.pw.weight"] = pw

    # Decompose delta.2 (first delta conv in dense; index 2 in nn.Sequential after GN+SiLU)
    if "cell.delta.2.weight" in src:
        W = src["cell.delta.2.weight"]
        dw, pw = decompose(W)
        new_sd["cell.delta.2.dw.weight"] = dw
        new_sd["cell.delta.2.pw.weight"] = pw

    # Decompose delta.5 (second delta conv)
    if "cell.delta.5.weight" in src:
        W = src["cell.delta.5.weight"]
        dw, pw = decompose(W)
        new_sd["cell.delta.5.dw.weight"] = dw
        new_sd["cell.delta.5.pw.weight"] = pw

    return new_sd


__all__ = [
    "DSConv",
    "StabHRec40LiteCell",
    "GRLNetLite",
    "GRLNetLiteConfig",
    "grlnet_stabhrec40_lite",
    "warm_start_from_dense",
]
