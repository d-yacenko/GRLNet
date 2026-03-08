# GRL Track Classifier

EN: A standalone PyTorch package for **track-form classification** with a ConvLSTM-based Gate-Residual Lattice (GRL) architecture.

RU: Автономный пакет на PyTorch для **классификации в трек-форме** на основе архитектуры Gate-Residual Lattice (GRL) с ConvLSTM.

## Overview / Обзор

EN: This project targets cases where the native unit of recognition is not a single image and not a classical time-ordered video sequence, but a **group of related observations** describing the same entity.

RU: Проект ориентирован на случаи, когда естественной единицей распознавания является не одна картинка и не классическая временная видеопоследовательность, а **группа связанных наблюдений**, описывающих одну и ту же сущность.

Typical use cases / Типичные сценарии:

- EN: one object observed from several viewpoints
  RU: один объект, наблюдаемый с нескольких ракурсов
- EN: one object seen by several cameras
  RU: один объект, наблюдаемый несколькими камерами
- EN: one scene or phenomenon observed through a short grouped window
  RU: одна сцена или явление, наблюдаемое через короткое сгруппированное окно
- EN: one still image converted into a pseudo-track for compatibility with the architecture
  RU: одна статическая картинка, преобразованная в псевдо-трек для совместимости с архитектурой

## Why Tracks Instead of Frames or Videos? / Почему треки, а не кадры или видео?

EN: The architecture is built around a **track** abstraction:

- a track is a group of images referring to the same object, scene, or observation
- temporal order is not the main modeling target
- track members may be different views, different cameras, or repeated versions of one image

RU: Архитектура построена вокруг абстракции **трека**:

- трек — это группа изображений, относящихся к одному объекту, сцене или наблюдению
- временной порядок не является основной целью моделирования
- элементы трека могут быть разными ракурсами, разными камерами или повторениями одной картинки

This follows the intended motivation of the architecture: iterative feature refinement and multi-view fusion without depending on strict video-style temporal dynamics.

Это соответствует исходной мотивации архитектуры: итеративное уточнение признаков и multi-view fusion без зависимости от строгой видео-семантики временной динамики.

## Core Properties / Ключевые свойства

- EN: native model contract: `Tensor[B, T, C, H, W] -> Tensor[B, K]`
  RU: нативный контракт модели: `Tensor[B, T, C, H, W] -> Tensor[B, K]`
- EN: notebook-compatible ConvLSTM implementation with learned `forget_bias`
  RU: совместимая с ноутбуком реализация ConvLSTM с обучаемым `forget_bias`
- EN: notebook-compatible `gold` protocol through `prep_batch()`
  RU: совместимый с ноутбуком `gold`-протокол через `prep_batch()`
- EN: support for single-image inference through pseudo-track adapters
  RU: поддержка инференса по одной картинке через pseudo-track adapters
- EN: support for grouped-image and video-to-track inference
  RU: поддержка инференса по группам изображений и по видео через преобразование в трек
- EN: grouped-image and video inference preserve the original views; augmentation is used only to fill missing positions in the active third
  RU: grouped-image и video inference сохраняют исходные ракурсы; аугментации используются только для достройки недостающих позиций активной трети
- EN: reference training recipe aligned with the current notebook
  RU: reference-рецепт обучения, согласованный с текущим ноутбуком

## Repository Layout / Структура репозитория

```text
src/grl_model/
  models/
    grl.py
    weights.py
  data/
    adapters.py
    datasets.py
  utils/
    predict.py
    training.py
scripts/
  train_imagenet.py
  eval_imagenet.py
docs/
examples/
tests/
```

## Install / Установка

```bash
pip install -e .
```

## Quickstart / Быстрый старт

### Create a model / Создать модель

```python
from grl_model.models import grl_base

model = grl_base(num_classes=1000, track_length=10)
```

### Train with the reference recipe / Обучить по reference-рецепту

```python
from grl_model.models import grl_base
from grl_model.utils import ReferenceTrainConfig, fit_reference_imagefolders, plot_history

model = grl_base(num_classes=1000, track_length=10)
config = ReferenceTrainConfig(epochs=100)

result = fit_reference_imagefolders(
    model,
    data_root="/path/to/imagefolder/root",
    # eval_root="/path/to/external/eval_root",  # optional
    track_length=10,
    batch_size=64,
    workers=8,
    image_size=224,
    center_crop=False,
    config=config,
    output_dir="runs/grl_reference",
)

plot_history(result.history)
print(result.best_val_acc, result.best_val_loss, result.best_epoch)
```

### Run inference on one image / Инференс по одной картинке

```python
from PIL import Image
from torchvision import transforms

from grl_model.models import grl_base
from grl_model.utils.predict import predict_image

model = grl_base(num_classes=1000, track_length=10).eval()
image = Image.open("sample.jpg").convert("RGB")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

logits = predict_image(model, image, track_length=10, image_transform=transform, apply_gold=True)
```

## Dataset Options / Варианты датасетов

### `ImageFolderPseudoTrackDataset`

EN: Auxiliary dataset for single-image to pseudo-track conversion.

RU: Вспомогательный датасет для преобразования одиночной картинки в псевдо-трек.

### `SequenceFolderDataset`

EN: Notebook-faithful grouped-by-class sequence construction. This is the main dataset abstraction used by the public training helper.

RU: Перенос notebook-логики с группировкой по классам. Это основная датасетная абстракция, которую использует публичный training helper.

### `TrackFolderDataset`

EN: For explicit grouped tracks stored as `root/class_name/track_id/frame.jpg`.

RU: Для явных треков, хранящихся как `root/class_name/track_id/frame.jpg`.

## Reference Training Semantics / Семантика reference-обучения

The default reference trainer preserves the important notebook behavior / Reference trainer сохраняет важные особенности ноутбука:

- EN: `train` applies the gold protocol with probability `0.5`
  RU: на `train` gold-протокол применяется с вероятностью `0.5`
- EN: `gold` always applies the gold protocol
  RU: на `gold` gold-протокол применяется всегда
- EN: `SequenceFolderDataset.on_epoch_end()` is called automatically after each training epoch
  RU: `SequenceFolderDataset.on_epoch_end()` автоматически вызывается в конце каждой эпохи
- EN: optimizer defaults follow the notebook: `AdamW` with split bias decay
  RU: optimizer по умолчанию повторяет ноутбук: `AdamW` с разным `weight_decay` для bias/non-bias
- EN: scheduler follows the notebook smoothed plateau logic
  RU: scheduler повторяет notebook-логику smoothed plateau
- EN: best model is selected by validation loss
  RU: лучшая модель выбирается по validation loss

High-level trainer behavior / Поведение high-level trainer:

- EN: `data_root` is the main grouped training source
  RU: `data_root` — основной источник grouped training данных
- EN: if `eval_root` is omitted, `train/val/gold` are split from `data_root`
  RU: если `eval_root` не задан, `train/val/gold` режутся из `data_root`
- EN: if `eval_root` is provided, it is reused for both `val` and `gold`
  RU: если `eval_root` задан, он используется и для `val`, и для `gold`
- EN: default transforms follow the notebook recipe, with optional `CenterCrop`
  RU: дефолтные трансформы повторяют notebook-рецепт, а `CenterCrop` остаётся опциональным

## Documentation / Документация

- [Core Concepts / Основные понятия](docs/concepts.md)
- [Architecture / Архитектура](docs/architecture.md)
- [Datasets / Датасеты](docs/datasets.md)
- [Training / Обучение](docs/training.md)
- [Inference / Инференс](docs/inference.md)
- [API Reference / API](docs/api.md)

## Examples / Примеры

- [Basic training / Базовое обучение](examples/basic_training.py)
- [Basic inference / Базовый инференс](examples/basic_inference.py)
- [Dataset overview / Обзор датасетов](examples/datasets_overview.py)

## Project Status / Статус проекта

EN: This repository is currently positioned as a standalone research-and-engineering package. It is not presented as part of any upstream model library.

RU: Репозиторий сейчас оформлен как самостоятельный research-and-engineering пакет. Он не позиционируется как часть какой-либо upstream-библиотеки моделей.

## Citation / Цитирование

EN: If you use this repository, see [CITATION.cff](CITATION.cff).

RU: Если вы используете этот репозиторий, см. [CITATION.cff](CITATION.cff).
