import torch

from grlnet import GRLNet, GRLNetWeights, grlnet_stabhrec40
from grlnet.models.weights import extract_model_state_dict


def test_grlnet_forward_shape():
    model = GRLNet(num_classes=11, stem_channels=8, hidden_channels=16, steps=2, aux_steps=1)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 11)


def test_grlnet_aux_forward_shape():
    model = GRLNet(num_classes=7, stem_channels=8, hidden_channels=16, steps=4, aux_steps=3)
    x = torch.randn(2, 3, 64, 64)
    logits, aux_logits = model(x, return_aux=True)
    assert logits.shape == (2, 7)
    assert len(aux_logits) == 2
    assert all(aux.shape == (2, 7) for aux in aux_logits)


def test_factory_builds_default_model():
    model = grlnet_stabhrec40(weights=None, num_classes=13)
    assert model.num_classes == 13
    assert sum(int(param.numel()) for param in model.parameters()) > 0


def test_extract_prefers_ema_model():
    model = GRLNet(num_classes=3, stem_channels=8, hidden_channels=16, steps=2, aux_steps=1)
    state = model.state_dict()
    ema_state = {key: value.clone() for key, value in state.items()}
    payload = {"model": state, "ema_model": ema_state}
    extracted = extract_model_state_dict(payload)
    assert set(extracted) == set(state)


def test_weights_registry_has_default():
    assert GRLNetWeights.DEFAULT.name in GRLNetWeights.names()
