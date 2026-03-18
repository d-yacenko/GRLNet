from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, Union

import torch
from PIL import Image
from torch import Tensor, nn

from grl_model.data.adapters import (
    apply_gold_protocol,
    build_track_from_images,
    build_pseudotrack_from_image,
    build_pseudotracks_from_images,
    build_track_from_video,
    canonicalize_track_batch,
)


def _infer_device(model: nn.Module, device: Optional[Union[torch.device, str]]) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.inference_mode()
def predict_track(
    model: nn.Module,
    track: Tensor,
    *,
    device: Optional[Union[torch.device, str]] = None,
    apply_gold: bool = False,
    track_layout: str = "BTCHW",
) -> Tensor:
    """Run inference on one track or a batch of tracks. / Выполнить инференс по одному треку или батчу треков.

    Parameters / Параметры
    ----------------------
    model:
        Track classifier. / Классификатор треков.
    track:
        Track tensor in the layout declared by ``track_layout``. /
        Тензор трека в layout, явно заданном через ``track_layout``.
    track_layout:
        Explicit input layout. Supported values: ``BTCHW``, ``TCHW``, ``BCTHW``, ``CTHW``. /
        Явный layout входа. Поддерживаются ``BTCHW``, ``TCHW``, ``BCTHW``, ``CTHW``.
    apply_gold:
        If ``True``, the notebook-compatible gold protocol is applied before forward. /
        Если ``True``, перед `forward` применяется gold-протокол ноутбука.
    """
    model.eval()
    target_device = _infer_device(model, device)
    track = canonicalize_track_batch(track, layout=track_layout)
    if apply_gold:
        track = apply_gold_protocol(track)
    return model(track.to(target_device))


@torch.inference_mode()
def predict_image(
    model: nn.Module,
    image: Union[Image.Image, Tensor],
    *,
    track_length: int,
    image_transform=None,
    device: Optional[Union[torch.device, str]] = None,
    apply_gold: bool = True,
) -> Tensor:
    """Run inference on a single image by first building a pseudo-track. / Выполнить инференс по одному изображению через предварительное построение pseudo-track."""
    track = build_pseudotrack_from_image(
        image,
        track_length=track_length,
        image_transform=image_transform,
    )
    return predict_track(model, track, device=device, apply_gold=apply_gold, track_layout="TCHW")


@torch.inference_mode()
def predict_images(
    model: nn.Module,
    images: Union[Sequence[Union[Image.Image, Tensor]], Tensor],
    *,
    track_length: int,
    image_transform=None,
    device: Optional[Union[torch.device, str]] = None,
    apply_gold: bool = True,
) -> Tensor:
    """Run inference on a batch of still images by converting each image to a pseudo-track. / Выполнить инференс по батчу изображений, преобразовав каждое в pseudo-track."""
    tracks = build_pseudotracks_from_images(
        images,
        track_length=track_length,
        image_transform=image_transform,
    )
    return predict_track(model, tracks, device=device, apply_gold=apply_gold, track_layout="BTCHW")


@torch.inference_mode()
def predict_group(
    model: nn.Module,
    images: Union[Sequence[Union[Image.Image, Tensor]], Tensor],
    *,
    track_length: int,
    image_transform: Any = None,
    active_frame_transform: Any = None,
    device: Optional[Union[torch.device, str]] = None,
) -> Tensor:
    """Run inference on a grouped image observation. / Выполнить инференс по группе изображений, образующих одно наблюдение.

    Unlike single-image inference, this helper does not apply the notebook gold protocol.
    For grouped observations the track should preserve the original views, and only the
    padding part may optionally be synthesized through ``active_frame_transform``.
    В отличие от инференса по одной картинке, этот helper не применяет notebook gold-протокол.
    Для групповых наблюдений трек должен сохранять исходные ракурсы, а через
    ``active_frame_transform`` при необходимости синтезируется только дополняющая часть.
    """
    track = build_track_from_images(
        images,
        track_length=track_length,
        image_transform=image_transform,
        active_frame_transform=active_frame_transform,
    )
    return predict_track(model, track, device=device, apply_gold=False, track_layout="TCHW")


@torch.inference_mode()
def predict_video(
    model: nn.Module,
    video: Union[str, Any],
    *,
    track_length: int,
    image_transform: Any = None,
    active_frame_transform: Any = None,
    sampling: str = "uniform",
    device: Optional[Union[torch.device, str]] = None,
) -> Tensor:
    """Run inference on a video by sampling frames into a track. / Выполнить инференс по видео, выбрав из него кадры и собрав трек.

    The notebook gold protocol is intentionally not applied here: for video inference it
    would collapse the active third to one anchor frame and destroy the information from
    multiple sampled frames.
    Notebook gold-протокол здесь намеренно не применяется: для видео он схлопнул бы
    активную треть к одному опорному кадру и уничтожил бы информацию из нескольких кадров.
    """
    track = build_track_from_video(
        video,
        track_length=track_length,
        image_transform=image_transform,
        active_frame_transform=active_frame_transform,
        sampling=sampling,
    )
    return predict_track(model, track, device=device, apply_gold=False, track_layout="TCHW")
