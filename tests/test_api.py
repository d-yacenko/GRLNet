import torch

from grl_model.data.adapters import (
    apply_gold_protocol,
    build_track_from_images,
    build_pseudotrack_from_image,
    build_pseudotracks_from_images,
    build_track_from_video,
)
from grl_model.models import ConvLSTMCell, grl_tiny


def test_model_accepts_track_batch():
    model = grl_tiny(num_classes=11, track_length=3)
    x = torch.randn(2, 9, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 11)


def test_convlstm_cell_has_forget_bias():
    cell = ConvLSTMCell(3, 16, 3)
    assert tuple(cell.forget_bias.shape) == (1, 16, 1, 1)
    assert torch.allclose(cell.conv.bias, torch.zeros_like(cell.conv.bias))


def test_build_pseudotrack_from_image_shape_and_padding():
    image = torch.randn(3, 32, 32)
    track = build_pseudotrack_from_image(image, track_length=4)
    assert track.shape == (12, 3, 32, 32)
    assert torch.count_nonzero(track[4:]) == 0


def test_build_pseudotracks_from_batch_shape():
    images = torch.randn(5, 3, 32, 32)
    tracks = build_pseudotracks_from_images(images, track_length=2)
    assert tracks.shape == (5, 6, 3, 32, 32)


def test_build_track_from_images_shape():
    frames = torch.randn(6, 3, 32, 32)
    track = build_track_from_images(frames, track_length=4)
    assert track.shape == (12, 3, 32, 32)


def test_build_track_from_video_tensor_shape():
    frames = torch.randn(6, 3, 32, 32)
    track = build_track_from_video(frames, track_length=4)
    assert track.shape == (12, 3, 32, 32)


def test_apply_gold_protocol_preserves_shape():
    track = torch.randn(2, 9, 3, 32, 32)
    gold = apply_gold_protocol(track, anchor_index=0)
    assert gold.shape == track.shape
