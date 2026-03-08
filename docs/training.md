# Training / Обучение

## Reference recipe / Reference-рецепт

EN: The library provides a notebook-aligned training path through:

RU: Библиотека предоставляет notebook-aligned путь обучения через:

- `ReferenceTrainConfig`
- `fit_reference(...)`
- `fit_reference_imagefolders(...)`

EN: The recommended public entrypoint is `fit_reference_imagefolders(...)`. It builds grouped `SequenceFolderDataset` splits and trains the model in the same conceptual regime as the notebook.

RU: Рекомендуемая публичная точка входа — `fit_reference_imagefolders(...)`. Она строит grouped split'ы на базе `SequenceFolderDataset` и обучает модель в той же концептуальной схеме, что и ноутбук.

## High-level trainer semantics / Семантика high-level trainer

EN:
- `data_root` is the main `ImageFolder` root
- if `eval_root` is omitted, `train/val/gold` are split from `data_root`
- if `eval_root` is provided, the whole `data_root` is used for training, while `eval_root` is reused for both `val` and `gold`
- default transforms follow the notebook recipe: `Resize`, `RandomHorizontalFlip`, `RandomRotation`, `Normalize`
- `CenterCrop` is optional and disabled by default

RU:
- `data_root` — основной `ImageFolder` root
- если `eval_root` не задан, `train/val/gold` режутся из `data_root`
- если `eval_root` задан, весь `data_root` идёт в train, а `eval_root` используется и для `val`, и для `gold`
- дефолтные трансформы повторяют рецепт ноутбука: `Resize`, `RandomHorizontalFlip`, `RandomRotation`, `Normalize`
- `CenterCrop` опционален и по умолчанию выключен

## Notebook semantics preserved / Сохранённые семантики ноутбука

### 1. Gold on train / Gold на train

EN: During `train`, `prep_batch()` is applied with probability `0.5` by default.

RU: Во время `train` по умолчанию `prep_batch()` применяется с вероятностью `0.5`.

### 2. Gold on gold / Gold на gold

EN: During `gold`, `prep_batch()` is always applied.

RU: Во время `gold` `prep_batch()` применяется всегда.

### 3. End-of-epoch reshuffle / Перемешивание в конце эпохи

EN: If the training dataset exposes `on_epoch_end()`, the reference trainer calls it automatically.

RU: Если training-dataset реализует `on_epoch_end()`, reference trainer вызывает его автоматически.

### 4. Optimizer and scheduler / Optimizer и scheduler

EN: The default reference recipe uses `AdamW` and the notebook smoothed plateau scheduler.

RU: По умолчанию reference-рецепт использует `AdamW` и notebook-реализацию smoothed plateau scheduler.

### 5. Best checkpoint criterion / Критерий лучшего checkpoint

EN: The best checkpoint is selected by minimum validation loss.

RU: Лучший checkpoint выбирается по минимуму validation loss.
