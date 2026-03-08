from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import pil_to_tensor

ImageLike = Union[Image.Image, Tensor]
VideoLike = Union[str, Path, Sequence[ImageLike], Tensor]


def _to_tensor_image(image: ImageLike) -> Tensor:
    if isinstance(image, Tensor):
        if image.ndim != 3:
            raise ValueError(f"Expected image tensor [C, H, W], got {tuple(image.shape)}")
        return image.float()
    if isinstance(image, Image.Image):
        return pil_to_tensor(image).float().div(255.0)
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def _normalize_image_group(images: Union[ImageLike, Sequence[ImageLike], Tensor]) -> List[ImageLike]:
    if isinstance(images, Tensor):
        if images.ndim == 3:
            return [images]
        if images.ndim == 4:
            return [images[idx] for idx in range(images.shape[0])]
        raise ValueError(f"Expected image tensor [C, H, W] or image group [N, C, H, W], got {tuple(images.shape)}")
    if isinstance(images, Image.Image):
        return [images]
    return list(images)


def _select_group_items(items: list[ImageLike], *, target_length: int, sampling: str) -> list[ImageLike]:
    if not items:
        raise ValueError("images must not be empty")
    if target_length <= 0:
        raise ValueError("track_length must be positive")

    if len(items) >= target_length:
        if sampling == "uniform":
            indices = torch.linspace(0, len(items) - 1, steps=target_length).round().long().tolist()
            return [items[idx] for idx in indices]
        if sampling == "head":
            return items[:target_length]
        raise ValueError(f"Unsupported sampling mode: {sampling!r}")

    selected = list(items)
    while len(selected) < target_length:
        selected.append(items[len(selected) % len(items)])
    return selected


def build_track_from_images(
    images: Union[ImageLike, Sequence[ImageLike], Tensor],
    *,
    track_length: int,
    image_transform: Optional[Callable[[Any], Tensor]] = None,
    active_frame_transform: Optional[Callable[[Tensor], Tensor]] = None,
    sampling: str = "uniform",
) -> Tensor:
    """Convert one image or a grouped image observation into a track. / Преобразовать одно изображение или группу изображений в трек.

    This is the main adapter for grouped observations:
    Это основной адаптер для групповых наблюдений:

    - one image / одна картинка
    - several views of one object / несколько ракурсов одного объекта
    - frames sampled from a short observation window / кадры, взятые из короткого окна наблюдения

    The active third is built from the provided image group. If the group is shorter than
    ``track_length``, items are repeated cyclically. If ``active_frame_transform`` is provided,
    it is applied to repeated items beyond the original group size. If the group is longer
    than ``track_length``, items are selected according to ``sampling``.
    Активная треть строится из переданной группы изображений. Если группа короче
    ``track_length``, элементы повторяются циклически. Если задан ``active_frame_transform``,
    он применяется к повторённым элементам сверх исходного размера группы. Если группа длиннее
    ``track_length``, элементы выбираются согласно политике ``sampling``.
    """
    items = _normalize_image_group(images)
    selected = _select_group_items(items, target_length=track_length, sampling=sampling)

    active = []
    for idx, image in enumerate(selected):
        tensor = image_transform(image) if image_transform is not None else _to_tensor_image(image)
        if idx >= len(items) and active_frame_transform is not None:
            tensor = active_frame_transform(tensor.clone())
        active.append(tensor)

    active_tensor = torch.stack(active, dim=0)
    zeros = torch.zeros((track_length * 2,) + tuple(active_tensor.shape[1:]), dtype=active_tensor.dtype)
    return torch.cat((active_tensor, zeros), dim=0)


def build_pseudotrack_from_image(
    image: ImageLike,
    *,
    track_length: int,
    image_transform: Optional[Callable[[Any], Tensor]] = None,
    active_frame_transform: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    """Convert one image into a notebook-compatible pseudo-track. / Преобразовать одно изображение в pseudo-track, совместимый с ноутбуком."""
    return build_track_from_images(
        image,
        track_length=track_length,
        image_transform=image_transform,
        active_frame_transform=active_frame_transform,
        sampling="head",
    )


def build_pseudotracks_from_images(
    images: Union[Sequence[ImageLike], Tensor],
    *,
    track_length: int,
    image_transform: Optional[Callable[[Any], Tensor]] = None,
    active_frame_transform: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    """Convert a batch of images into a batch of pseudo-tracks. / Преобразовать батч изображений в батч pseudo-track.

    Passing ``active_frame_transform`` builds gold-like active thirds directly in the adapter.
    Передача ``active_frame_transform`` позволяет строить gold-like активную треть прямо в адаптере.
    """
    if isinstance(images, Tensor):
        if images.ndim != 4:
            raise ValueError(f"Expected image batch [B, C, H, W], got {tuple(images.shape)}")
        image_list = [images[idx] for idx in range(images.shape[0])]
    else:
        image_list = list(images)

    tracks = [
        build_pseudotrack_from_image(
            image,
            track_length=track_length,
            image_transform=image_transform,
            active_frame_transform=active_frame_transform,
        )
        for image in image_list
    ]
    return torch.stack(tracks, dim=0)


def build_track_from_video(
    video: VideoLike,
    *,
    track_length: int,
    image_transform: Optional[Callable[[Any], Tensor]] = None,
    active_frame_transform: Optional[Callable[[Tensor], Tensor]] = None,
    sampling: str = "uniform",
) -> Tensor:
    """Convert a video or pre-decoded frame group into a track. / Преобразовать видео или заранее декодированную группу кадров в трек.

    Accepted inputs:
    Допустимые входы:

    - path to a video file / путь к видеофайлу
    - sequence of PIL images / последовательность PIL-изображений
    - tensor ``[T, C, H, W]`` / тензор ``[T, C, H, W]``

    Frames are sampled with the specified policy and then converted to the standard
    active-third-plus-zero-tail track layout. The default policy is ``uniform``, which
    spreads samples across the whole clip instead of taking only the first frames.
    Кадры выбираются согласно указанной политике, после чего преобразуются
    в стандартный track layout: активная треть плюс нулевой хвост. Политика по умолчанию —
    ``uniform``, то есть кадры равномерно покрывают ролик, а не берутся только из начала.
    """
    if isinstance(video, (str, Path)):
        from torchvision.io import read_video

        frames, _, _ = read_video(str(video), pts_unit="sec")
        if frames.ndim != 4:
            raise ValueError(f"Expected decoded video frames [T, H, W, C], got {tuple(frames.shape)}")
        video = frames.permute(0, 3, 1, 2).contiguous().float().div(255.0)

    return build_track_from_images(
        video,
        track_length=track_length,
        image_transform=image_transform,
        active_frame_transform=active_frame_transform,
        sampling=sampling,
    )


def apply_gold_protocol(
    track_or_batch: Tensor,
    *,
    frame_transform: Optional[Callable[[Tensor], Tensor]] = None,
    anchor_index: Optional[int] = None,
) -> Tensor:
    """Apply the notebook gold protocol to a track tensor on CPU. / Применить gold-протокол ноутбука к тензору трека на CPU.

    This helper mirrors the semantics of ``model.prep_batch`` and is mainly useful when
    the caller wants a standalone transformation step outside the model object.
    Этот helper повторяет семантику ``model.prep_batch`` и полезен, когда вызывающей стороне
    нужен отдельный шаг трансформации вне объекта модели.
    """
    x = track_or_batch.clone()
    squeeze = False
    if x.ndim == 4:
        x = x.unsqueeze(0)
        squeeze = True
    if x.ndim != 5:
        raise ValueError(f"Expected [T, C, H, W] or [B, T, C, H, W], got {tuple(track_or_batch.shape)}")
    if x.is_cuda:
        raise RuntimeError("apply_gold_protocol must run on CPU before moving the batch to device")

    for b in range(len(x)):
        n = x[b].shape[0] // 3
        if n <= 0:
            continue
        t = anchor_index if anchor_index is not None else random.randint(0, n - 1)
        for i in range(n):
            if i == t:
                continue
            src = torch.clone(x[b][t])
            x[b][i] = frame_transform(src) if frame_transform is not None else src
    return x.squeeze(0) if squeeze else x
