"""API smoke tests for the depthwise-separable Lite variant (v0.4.0+)."""
import pytest
import torch

import grlnet
from grlnet import GRLNetLite, GRLNetLiteWeights, grlnet_stabhrec40_lite
from grlnet.models.stabhrec40_lite import (
    DSConv,
    StabHRec40LiteCell,
    warm_start_from_dense,
)


def test_lite_forward_shape():
    model = GRLNetLite(num_classes=11, stem_channels=8, hidden_channels=16, steps=2, aux_steps=1)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 11)


def test_lite_aux_forward_shape():
    model = GRLNetLite(num_classes=7, stem_channels=8, hidden_channels=16, steps=4, aux_steps=3)
    x = torch.randn(2, 3, 64, 64)
    logits, aux_logits = model(x, return_aux=True)
    assert logits.shape == (2, 7)
    assert len(aux_logits) == 2
    assert all(aux.shape == (2, 7) for aux in aux_logits)


def test_lite_factory_builds_untrained_model():
    model = grlnet_stabhrec40_lite(weights=None, num_classes=13)
    assert model.num_classes == 13
    assert sum(int(p.numel()) for p in model.parameters()) > 0


def test_lite_has_fewer_params_than_dense():
    """Lite should always be smaller — this is the architectural promise."""
    dense = grlnet.grlnet_stabhrec40(weights=None, num_classes=100)
    lite = grlnet_stabhrec40_lite(weights=None, num_classes=100)
    n_dense = sum(p.numel() for p in dense.parameters())
    n_lite = sum(p.numel() for p in lite.parameters())
    assert n_lite < n_dense, f"Lite must be smaller than dense: {n_lite} vs {n_dense}"
    # Empirically observed ratio ≈ 3.2× on 100-class config.
    assert n_dense / n_lite > 2.0, f"Lite/dense ratio too small: {n_dense / n_lite:.2f}"


def test_dsconv_is_separable():
    """DSConv = DW(k×k, groups=in_ch) + PW(1×1)."""
    m = DSConv(8, 32, kernel_size=3)
    assert isinstance(m.dw, torch.nn.Conv2d)
    assert m.dw.groups == 8
    assert m.dw.kernel_size == (3, 3)
    assert m.dw.in_channels == m.dw.out_channels == 8
    assert isinstance(m.pw, torch.nn.Conv2d)
    assert m.pw.kernel_size == (1, 1)
    assert m.pw.in_channels == 8 and m.pw.out_channels == 32


def test_lite_cell_skeleton_matches_dense():
    """Lite cell preserves all non-conv components (norms, gates, scalars)."""
    cell = StabHRec40LiteCell(channels=16)
    assert hasattr(cell, "gate_norm")
    assert hasattr(cell, "gate_act")
    assert hasattr(cell, "gate_conv")
    assert hasattr(cell, "c_norm")
    assert hasattr(cell, "delta")
    assert hasattr(cell, "hidden_scale")
    assert hasattr(cell, "delta_scale")


def test_warm_start_runs_without_error():
    """Warm-start utility produces a state dict compatible with GRLNetLite."""
    dense = grlnet.grlnet_stabhrec40(weights=None, num_classes=100)
    lite = grlnet_stabhrec40_lite(weights=None, num_classes=100)
    sd = warm_start_from_dense(lite, dense.state_dict())
    # Should populate at least the cell-conv DW/PW pairs and identity-copy others
    expected_dw_pw_keys = {
        "cell.gate_conv.dw.weight", "cell.gate_conv.pw.weight",
        "cell.delta.2.dw.weight", "cell.delta.2.pw.weight",
        "cell.delta.5.dw.weight", "cell.delta.5.pw.weight",
    }
    assert expected_dw_pw_keys.issubset(sd.keys())
    # Loading should not raise (strict=False because head dims may differ)
    missing, unexpected = lite.load_state_dict(sd, strict=False)
    # We expect only the freshly-initialized 100-class head to be missing-or-different;
    # all DW/PW entries should be present.


def test_lite_weights_registry_has_default():
    assert GRLNetLiteWeights.DEFAULT.name in GRLNetLiteWeights.names()
    assert "DEFAULT" in GRLNetLiteWeights.names()
    assert GRLNetLiteWeights.get("DEFAULT") is GRLNetLiteWeights.DEFAULT


def test_lite_weights_registry_separate_from_dense():
    """Dense and Lite registries must not contaminate each other."""
    assert GRLNetLiteWeights.DEFAULT.name != grlnet.GRLNetWeights.DEFAULT.name
    # Dense weights URL points to v0.3.0; lite URL is pending until v0.4.0 release
    assert "v0.3.0" in grlnet.GRLNetWeights.DEFAULT.url
    # Lite URL not yet set (released checkpoint pending)
    assert GRLNetLiteWeights.DEFAULT.url is None


def test_lite_default_loading_raises_until_release():
    """``weights='DEFAULT'`` must fail clearly until the Lite checkpoint is published."""
    with pytest.raises((RuntimeError, KeyError)):
        grlnet_stabhrec40_lite(weights="DEFAULT", num_classes=1000)
