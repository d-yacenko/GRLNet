from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import pil_to_tensor

from .adapters import build_pseudotrack_from_image, build_track_from_images


class SequenceFolderDataset(Dataset):
    """Faithful port of the notebook SequenceFolder behavior. / Точный перенос поведения notebook SequenceFolder.

    Samples are grouped by class, chunked into sequences of length ``seq_len``, and then
    expanded to notebook track format: active first third + zero tail.
    Сэмплы группируются по классу, режутся на последовательности длины ``seq_len``, а затем
    расширяются до notebook-формата трека: активная первая треть + нулевой хвост.

    This dataset is useful when the goal is to reproduce the notebook training regime
    rather than model a natural video sequence.
    Этот датасет нужен, когда важно воспроизвести режим обучения из ноутбука,
    а не смоделировать естественную видеопоследовательность.
    """

    def __init__(self, root: Union[str, Path], seq_len: int, allowed_idxs=None, transform=None):
        self.folder = ImageFolder(root)
        self.transform = transform
        self.seq_len = seq_len
        self.allowed = list(range(len(self.folder.samples))) if allowed_idxs is None else list(allowed_idxs)

        self.by_class: dict[int, list[int]] = {}
        for i in self.allowed:
            _, label = self.folder.samples[i]
            self.by_class.setdefault(label, []).append(i)

        self.on_epoch_end()

    def on_epoch_end(self):
        self.sequences = []
        self.labels = []
        for label, idxs in self.by_class.items():
            idxs = list(idxs)
            random.shuffle(idxs)
            if len(idxs) < self.seq_len:
                idxs = (idxs * math.ceil(self.seq_len / len(idxs)))[: self.seq_len]
            for i in range(0, len(idxs) - self.seq_len + 1, self.seq_len):
                chunk = idxs[i : i + self.seq_len]
                self.sequences.append(chunk)
                self.labels.append(label)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        chunk = self.sequences[idx]
        imgs = []
        for sample_idx in chunk:
            path, _ = self.folder.samples[sample_idx]
            image = Image.open(path).convert("RGB")
            if self.transform is not None:
                imgs.append(self.transform(image))
            else:
                imgs.append(pil_to_tensor(image).float().div(255.0))
        active = torch.stack(imgs, dim=0)
        # Исторический notebook-режим: после активной трети всегда идёт нулевой хвост длиной 2 * seq_len.
        zeros = torch.zeros((len(imgs) * 2,) + tuple(active.shape[1:]), dtype=active.dtype)
        return torch.cat((active, zeros), dim=0), self.labels[idx]


class ImageFolderPseudoTrackDataset(Dataset):
    """ImageFolder dataset rendered into notebook-compatible pseudo-tracks. / ImageFolder, приведённый к pseudo-track формату.

    Each image becomes one pseudo-track:
    Каждое изображение превращается в один pseudo-track:

    - first ``track_length`` frames are active copies of the image /
      первые ``track_length`` кадров являются активными копиями изображения
    - last ``2 * track_length`` frames are zero padding /
      последние ``2 * track_length`` кадров заполнены нулями

    This is the most convenient dataset for standard image-classification corpora such as
    ImageNet when the model still expects track-form inputs.
    Это самый удобный датасет для обычных image-classification корпусов вроде ImageNet,
    когда модель при этом всё ещё ожидает входы в формате трека.
    """

    def __init__(self, root: Union[str, Path], *, track_length: int, image_transform=None) -> None:
        self.dataset = ImageFolder(root)
        self.track_length = track_length
        self.image_transform = image_transform
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        path, label = self.dataset.samples[index]
        image = Image.open(path).convert("RGB")
        track = build_pseudotrack_from_image(
            image,
            track_length=self.track_length,
            image_transform=self.image_transform,
        )
        return track, label


class TrackFolderDataset(Dataset):
    """Dataset for explicit grouped tracks. / Датасет для явно заданных групповых треков.

    Expected layout / Ожидаемая структура:

    ``root/class_name/track_id/frame.jpg``

    Use this when the training data already comes as groups of related images that
    describe the same semantic entity.
    Используйте этот вариант, когда обучающие данные уже представлены группами связанных
    изображений, описывающих одну и ту же семантическую сущность.
    """

    def __init__(self, root: Union[str, Path], *, track_length: int, image_transform=None) -> None:
        self.root = Path(root)
        self.track_length = track_length
        self.image_transform = image_transform

        self.classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples: list[tuple[list[Path], int]] = []

        for class_name in self.classes:
            class_dir = self.root / class_name
            for track_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
                frames = sorted(
                    p for p in track_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                )
                if frames:
                    self.samples.append((frames, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        frame_paths, label = self.samples[index]
        images = [Image.open(path).convert("RGB") for path in frame_paths]
        track = build_track_from_images(
            images,
            track_length=self.track_length,
            image_transform=self.image_transform,
        )
        return track, label
