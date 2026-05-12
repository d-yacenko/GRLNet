"""torch.hub entry-points for GRLNet/StabHRec40 and GRLNet/StabHRec40-Lite.

Usage (dense baseline, v0.3.0+):

    import torch
    model = torch.hub.load(
        "d-yacenko/GRLNet", "grlnet_stabhrec40",
        weights="DEFAULT", trust_repo=True,
    ).eval()
    logits = model(torch.randn(1, 3, 224, 224))   # shape: [1, 1000]

Usage (depthwise-separable Lite variant, v0.4.0+):

    model = torch.hub.load(
        "d-yacenko/GRLNet", "grlnet_stabhrec40_lite",
        weights="DEFAULT", trust_repo=True,
    ).eval()

Each entry-point fetches its checkpoint from the corresponding GitHub
Release asset and verifies SHA256 in
``GRLNetWeights.DEFAULT.get_state_dict(...)`` /
``GRLNetLiteWeights.DEFAULT.get_state_dict(...)``.
"""
from __future__ import annotations

import os
import sys

dependencies = ["torch"]

# The package lives under ``src/grlnet/`` (PEP 621 src layout).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from grlnet import grlnet_stabhrec40 as _grlnet_stabhrec40  # noqa: E402
from grlnet import grlnet_stabhrec40_lite as _grlnet_stabhrec40_lite  # noqa: E402


def grlnet_stabhrec40(weights="DEFAULT", **kwargs):
    """GRLNet/StabHRec40 --- compact recurrent CNN for image classification.

    Dense baseline, 3.25 M parameters, ~76 GMAC at T=12, Top-1 = 69.768%
    on ImageNet-1K (EMA, epoch 120).

    Parameters
    ----------
    weights : str | grlnet.GRLNetWeights
        ``"DEFAULT"`` loads the published v0.3.0 ImageNet-1K checkpoint.
        Pass ``None`` for an untrained model. Any registered
        ``GRLNetWeights`` entry is also accepted.
    **kwargs
        Forwarded to ``grlnet.grlnet_stabhrec40``; see that function for the
        architecture knobs (``stem_channels``, ``hidden_channels``, ``steps``,
        ``readout``, ...).
    """
    return _grlnet_stabhrec40(weights=weights, **kwargs)


def grlnet_stabhrec40_lite(weights="DEFAULT", **kwargs):
    """GRLNet/StabHRec40-Lite --- depthwise-separable variant of StabHRec40.

    Same architectural skeleton (shared recurrent cell × 12 unroll steps,
    two-stream H/C cell, residual stabilizers, untying-friendly weights),
    but with dense 3×3 convolutions inside the cell replaced by
    depthwise-separable pairs (DW3×3 + PW1×1). Yields ~1.49 M parameters
    (2.2× fewer) and ~9.66 GMAC at T=12 (7.9× less compute) with comparable
    accuracy on the same recipe.

    Parameters
    ----------
    weights : str | grlnet.GRLNetLiteWeights
        ``"DEFAULT"`` loads the published Lite ImageNet-1K checkpoint
        (URL populated in the v0.4.0 release). Pass ``None`` for an
        untrained model. Until v0.4.0 weights are published,
        ``weights="DEFAULT"`` raises a clear error.
    **kwargs
        Forwarded to ``grlnet.grlnet_stabhrec40_lite``.
    """
    return _grlnet_stabhrec40_lite(weights=weights, **kwargs)
