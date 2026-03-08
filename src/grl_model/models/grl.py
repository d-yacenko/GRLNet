from __future__ import annotations

import random
from typing import Iterable

import torch
from torch import Tensor, nn
from torchvision import transforms


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell used by :class:`GRLClassifier`. / ConvLSTM-ячейка для :class:`GRLClassifier`.

    Important notebook-compatible details:
    Важные детали совместимости с ноутбуком:

    - convolution bias is zero-initialized / bias свертки инициализируется нулями
    - a learned ``forget_bias`` is added to the forget gate logits /
      к логитам forget-gate добавляется обучаемый ``forget_bias``
    - gate order matches the notebook implementation: ``i, f, g, o`` /
      порядок ворот совпадает с ноутбуком: ``i, f, g, o``
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3, bias: bool = True) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        self.forget_bias = nn.Parameter(torch.ones(1, hidden_channels, 1, 1) * 1.5)

    def forward(self, x: Tensor, hidden: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        h_prev, c_prev = hidden
        combined = torch.cat([x, h_prev], dim=1)
        conv_out = self.conv(combined)

        cc_i, cc_f, cc_g, cc_o = torch.split(conv_out, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f + self.forget_bias)
        g = torch.tanh(cc_g)
        o = torch.sigmoid(cc_o)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class GRLClassifier(nn.Module):
    """ConvLSTM classifier that consumes track-form inputs. / ConvLSTM-классификатор для входов в формате трека.

    Canonical input shape: ``[B, T, C, H, W]``.
    Каноническая форма входа: ``[B, T, C, H, W]``.

    The current notebook protocol uses tracks of total length ``3 * seq_len_train``:
    the first third contains active frames, the last two thirds are zero padding.
    Текущий протокол ноутбука использует треки полной длины ``3 * seq_len_train``:
    первая треть содержит активные кадры, последние две трети заполнены нулями.

    Parameters / Параметры
    ----------------------
    num_classes:
        Number of output classes. / Число выходных классов.
    in_channels:
        Number of input image channels. / Число входных каналов изображения.
    hidden_channels:
        Hidden widths of the stacked ConvLSTM blocks. /
        Размерности скрытых состояний стековых ConvLSTM-блоков.
    kernel_size:
        Convolution kernel size inside each ConvLSTM cell. /
        Размер ядра свертки внутри каждой ConvLSTM-ячейки.
    global_pool:
        Output size of the adaptive head pooling. /
        Размер выходного адаптивного pooling в head.
    track_length:
        Active track length before notebook-style zero padding. The full notebook
        track length is therefore ``3 * track_length``. /
        Длина активной части трека до notebook-style дополнения нулями.
        Полная длина трека при этом равна ``3 * track_length``.
    """

    def __init__(
        self,
        *,
        num_classes: int = 1000,
        in_channels: int = 3,
        hidden_channels: Iterable[int] = (32, 64, 128, 256),
        kernel_size: int = 3,
        global_pool: int = 2,
        track_length: int = 10,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.seq_len_train = int(track_length)
        self.global_pool = global_pool
        self.hidden_channels = tuple(hidden_channels)
        self.num_layers = len(self.hidden_channels)

        prev_channels = in_channels
        self.cells = nn.ModuleList()
        for hidden in self.hidden_channels:
            self.cells.append(ConvLSTMCell(prev_channels, hidden, kernel_size))
            prev_channels = hidden

        self.maxpool = nn.MaxPool2d(2)
        self.head_pool = nn.AdaptiveAvgPool2d((global_pool, global_pool))
        self.fc = nn.Linear(self.hidden_channels[-1] * global_pool * global_pool, num_classes)
        self.trans = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(
                (-10, 10),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
        ])

    @staticmethod
    def _ensure_track_batch(x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input [B, T, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] in (1, 3) and x.shape[2] not in (1, 3):
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x

    def prep_batch(self, inputs: Tensor) -> None:
        """Apply the notebook-compatible gold protocol in place. / Применить gold-протокол ноутбука in-place.

        The function:
        Функция:

        - expects a CPU tensor with shape ``[B, T, C, H, W]`` /
          ожидает CPU-тензор формы ``[B, T, C, H, W]``
        - operates only on the first third of frames /
          работает только с первой третью кадров
        - samples one anchor frame per track /
          выбирает один опорный кадр на каждый трек
        - rewrites the rest of the first third through the model augmentation pipeline /
          переписывает остальные кадры первой трети через пайплайн аугментаций модели

        This is part of the historical training and evaluation recipe and is intentionally
        kept as a public method.
        Это часть исторического training/eval recipe, поэтому метод намеренно оставлен публичным.
        """
        if inputs.ndim != 5:
            raise ValueError(f"prep_batch expects [B, T, C, H, W], got {tuple(inputs.shape)}")
        if inputs.is_cuda:
            raise RuntimeError("prep_batch must run on CPU before inputs.to(device)")

        for b in range(len(inputs)):
            n = inputs[b].shape[0] // 3
            if n <= 0:
                continue
            t = random.randint(0, n - 1)
            for i in range(n):
                if i == t:
                    continue
                inputs[b][i] = self.trans(torch.clone(inputs[b][t]))

    def forward_features(self, x_seq: Tensor) -> Tensor:
        x_seq = self._ensure_track_batch(x_seq)
        batch_size, _, _, height, width = x_seq.shape

        h: list[Tensor] = []
        h_size = height
        w_size = width
        for cell in self.cells:
            h.append(x_seq.new_zeros((batch_size, cell.hidden_channels, h_size, w_size)))
            h_size //= 2
            w_size //= 2
        c = [torch.zeros_like(h_i) for h_i in h]

        for t in range(x_seq.shape[1]):
            x = x_seq[:, t]
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx], c[layer_idx] = cell(x, (h[layer_idx], c[layer_idx]))
                x = self.maxpool(h[layer_idx])

        return self.head_pool(h[-1]).flatten(1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(self.forward_features(x))


def grl_tiny(*, num_classes: int = 1000, track_length: int = 10, **kwargs: object) -> GRLClassifier:
    return GRLClassifier(
        num_classes=num_classes,
        hidden_channels=(16, 32, 64),
        global_pool=2,
        track_length=track_length,
        **kwargs,
    )


def grl_base(*, num_classes: int = 1000, track_length: int = 10, **kwargs: object) -> GRLClassifier:
    return GRLClassifier(
        num_classes=num_classes,
        hidden_channels=(32, 64, 128, 256),
        global_pool=2,
        track_length=track_length,
        **kwargs,
    )
