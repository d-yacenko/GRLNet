"""torch.hub entry-point for GRLNet/StabHRec40.

Usage:

    import torch
    model = torch.hub.load(
        "d-yacenko/GRLNet", "grlnet_stabhrec40",
        weights="DEFAULT", trust_repo=True,
    ).eval()
    logits = model(torch.randn(1, 3, 224, 224))   # shape: [1, 1000]

The published v0.3.0 weights are fetched from the GitHub release and verified
by SHA256 inside ``GRLNetWeights.DEFAULT.get_state_dict(...)``.
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


def grlnet_stabhrec40(weights="DEFAULT", **kwargs):
    """GRLNet/StabHRec40 --- compact recurrent CNN for image classification.

    Parameters
    ----------
    weights : str | grlnet.GRLNetWeights
        ``"DEFAULT"`` loads the published v0.3.0 ImageNet-1K checkpoint
        (Top-1 = 69.768%, 3.25M parameters). Pass ``None`` for an
        untrained model. Any registered ``GRLNetWeights`` entry is also
        accepted.
    **kwargs
        Forwarded to ``grlnet.grlnet_stabhrec40``; see that function for
        the architecture knobs (``stem_channels``, ``hidden_channels``,
        ``steps``, ``readout``, ...).
    """
    return _grlnet_stabhrec40(weights=weights, **kwargs)
