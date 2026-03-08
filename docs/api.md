# API Reference / API

## Models / Модели

### `grl_model.models.ConvLSTMCell`

EN: Low-level recurrent cell used inside the classifier.

RU: Низкоуровневая рекуррентная ячейка, используемая внутри классификатора.

### `grl_model.models.GRLClassifier`

EN: Main track-form classifier.

RU: Основной классификатор для входов в track-форме.

### `grl_model.models.grl_tiny`

EN: Small preset.

RU: Малый preset.

### `grl_model.models.grl_base`

EN: Main preset aligned with the current notebook experiments.

RU: Основной preset, согласованный с текущими notebook-экспериментами.

## Data adapters / Адаптеры данных

- `build_track_from_images`
- `build_track_from_video`
- `build_pseudotrack_from_image`
- `build_pseudotracks_from_images`
- `apply_gold_protocol`

## Datasets / Датасеты

- `ImageFolderPseudoTrackDataset`
- `SequenceFolderDataset`
- `TrackFolderDataset`

## Prediction helpers / Функции предикта

- `predict_track`
- `predict_image`
- `predict_images`
- `predict_group`
- `predict_video`

EN: `predict_group` and `predict_video` preserve the original multi-view content and do not expose `apply_gold`.

RU: `predict_group` и `predict_video` сохраняют исходное multi-view содержимое и не используют параметр `apply_gold`.

## Training utilities / Утилиты обучения

- `ReferenceTrainConfig`
- `ReferenceTrainResult`
- `SmoothedReduceLROnPlateau`
- `build_reference_optimizer`
- `build_reference_scheduler`
- `set_reference_seed`
- `fit_reference`
- `fit_reference_imagefolders`
- `plot_history`
