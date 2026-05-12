"""Deprecated alias module: kept for backwards-compat of the early DS prototype.

The depthwise-separable variant is now published as ``stabhrec40_lite``.
All public symbols here re-export the corresponding ``_lite`` names so old code
keeps working.

This module will be removed in a future minor release; please migrate to::

    from grlnet.models.stabhrec40_lite import GRLNetLite, grlnet_stabhrec40_lite

or use the package-level alias::

    from grlnet import grlnet_stabhrec40_lite
"""
from __future__ import annotations

import warnings

from .stabhrec40_lite import (
    DSConv,
    GRLNetLite as GRLNetDS,
    GRLNetLiteConfig as GRLNetDSConfig,
    StabHRec40LiteCell as StabHRec40DSCell,
    grlnet_stabhrec40_lite as grlnet_stabhrec40_ds,
    warm_start_from_dense,
)

warnings.warn(
    "grlnet.models.stabhrec40_ds is a deprecated alias; "
    "use grlnet.models.stabhrec40_lite instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "DSConv",
    "StabHRec40DSCell",
    "GRLNetDS",
    "GRLNetDSConfig",
    "grlnet_stabhrec40_ds",
    "warm_start_from_dense",
]
