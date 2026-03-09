# GRLNet Agent Context

Этот файл предназначен для агента и разработчика, который входит в проект без полного контекста переписки. Он фиксирует текущее состояние репозитория, архитектурные допущения, ключевые артефакты и уже обнаруженные проблемы.

## 1. Что такое GRLNet

GRLNet — это библиотечная реализация архитектуры GRL для классификации `track`-представлений.

Ключевая идея:
- модель не является обычным image-classifier;
- модель не является video-model в классическом смысле, где критичен временной порядок кадров;
- основной вход — `track`, то есть группа изображений одной сущности;
- эта группа может быть:
  - одной картинкой, превращённой в pseudo-track;
  - несколькими ракурсами одного объекта;
  - несколькими наблюдениями одной сущности с разных камер;
  - реальным short track;
  - видеофрагментом, из которого выбраны кадры.

Модель особенно полезна там, где несколько изображений одной сущности несут больше информации, чем один снимок.

## 2. Канонический контракт модели

Канонический вход модели:
- `Tensor[B, T, C, H, W]`

Канонический выход модели:
- `Tensor[B, K]`

Где:
- `B` — batch size;
- `T` — длина track;
- `C, H, W` — каналы и размер изображения;
- `K` — число классов.

Одна картинка не подаётся в модель напрямую. Она должна быть сначала преобразована в `track` через адаптеры.

## 3. Ключевые сущности проекта

### 3.1 Модель

Основной файл:
- `src/grl_model/models/grl.py`

Содержит:
- `ConvLSTMCell`
- `GRLClassifier`
- factory `grl_base(...)`

Критически важные свойства, восстановленные из ноутбука:
- zero-init bias в `ConvLSTMCell`;
- `forget_bias`;
- правильный порядок ворот `i, f, g, o`;
- `seq_len_train`;
- встроенный `prep_batch()`;
- встроенный аугментатор `self.trans`.

### 3.2 Датасеты

Основной файл:
- `src/grl_model/data/datasets.py`

Ключевые датасеты:
- `SequenceFolderDataset`
- `TrackFolderDataset`
- `ImageFolderPseudoTrackDataset`

Текущее позиционирование:
- `SequenceFolderDataset` — основной grouped training path и наиболее близкий к ноутбуку режим;
- `TrackFolderDataset` — когда треки уже существуют как группы файлов на диске;
- `ImageFolderPseudoTrackDataset` — вспомогательная сущность для pseudo-track use-case, но не главный режим обучения.

### 3.3 Адаптеры

Основной файл:
- `src/grl_model/data/adapters.py`

Ключевые функции:
- `build_pseudotrack_from_image(...)`
- `build_pseudotracks_from_images(...)`
- `build_track_from_images(...)`
- `build_track_from_video(...)`
- `apply_gold_protocol(...)`

Текущая семантика:
- для одной картинки допустим `gold`-режим;
- для группы изображений и видео старый `gold` через `prep_batch()` считается плохой практикой и не должен быть default;
- для группы и видео допустим только `gold-like` смысл как политика заполнения недостающих кадров активной части через повторение и/или аугментацию.

### 3.4 Инференс

Основной файл:
- `src/grl_model/utils/predict.py`

Публичные сценарии:
- `predict_track(...)`
- `predict_image(...)`
- `predict_images(...)`
- `predict_group(...)`
- `predict_video(...)`

Текущее правило:
- `predict_image(...)` может использовать `gold`;
- `predict_group(...)` и `predict_video(...)` не должны схлопывать наблюдение к одному кадру через `prep_batch()`.

### 3.5 Обучение

Основной файл:
- `src/grl_model/utils/training.py`

Ключевые сущности:
- `ReferenceTrainConfig`
- `fit_reference(...)`
- `fit_reference_imagefolders(...)`
- `plot_history(...)`
- `set_reference_seed(...)`

Текущий принцип:
- `fit_reference(...)` — читаемый reference trainer;
- `fit_reference_imagefolders(...)` — high-level helper, строящий grouped split'ы через `SequenceFolderDataset` и обучающий в концепции ноутбука.

## 4. Исторический источник истины

Главный исходный экспериментальный код:
- `6_rcnn_transfear_ovis.ipynb`

Верификация библиотечной реализации против ноутбука выполнялась через:
- `7_rcnn_final.ipynb`

Критические части, которые библиотека уже воспроизводит по смыслу:
- `ConvLSTMCell` с `forget_bias`;
- grouped training regime;
- `gold` на `train` с вероятностью `0.5`;
- `gold` на phase `gold` всегда;
- notebook-style `prep_batch()`;
- `SequenceFolderDataset.on_epoch_end()`;
- AdamW с раздельной обработкой bias/non-bias;
- plateau-style scheduler;
- best model selection по `val_loss`.

## 5. Что уже было проверено

### 5.1 Эквивалентность notebook 6 vs notebook 7/library

Сравнение делалось по:
- старым report-файлам из ноутбука 6;
- новому `grl_iruo_verify_history.json` из библиотечного прогона.

Результат:
- `train` и `val` динамика практически эквивалентны;
- `gold` более шумный, но в том же режиме значений;
- библиотечный training path считается практически согласованным с исходным ноутбуком.

Сгенерированные артефакты сравнения:
- `grl_track_classifier_runs/iruo_verify/grl_iruo_verify_vs_notebook6_all_smoothed.png`
- другие PNG/JSON в `grl_track_classifier_runs/iruo_verify/`

### 5.2 Проверка на ImageNet-1k benchmark

Подготовлены артефакты:
- `scripts/prepare_imagenet_layout.py`
- `scripts/run_one_epoch_benchmark.py`
- `slurm/one_epoch_cpu_ram.slurm`
- `slurm/one_epoch_gpu_t4.slurm`
- `slurm/one_epoch_gpu_a100_80gb.slurm`

Фактические измерения:
- `1 x T4`: примерно `9.18 часа` на эпоху;
- `CPU large-RAM node`: примерно `5.5-6 суток` на эпоху;
- `A100`: на момент составления документа замер ожидался или выполнялся.

## 6. Серьёзные проблемы, с которыми уже столкнулись

### 6.1 Неправильный mapping ImageNet val

Проблема:
- официальный `ILSVRC2012_img_val.tar` плоский;
- разметка `val` не восстанавливается сортировкой `wnid` из `train`;
- нужен официальный порядок `ILSVRC2012 class index -> wnid` из `devkit/meta.mat`.

Следствие:
- если строить `imagenet_wnids.txt` лексикографически, `val` будет разложен по неверным классам.

### 6.2 Python 3.9 compatibility

Проблема:
- исходно пакет требовал `>=3.10`;
- серверное окружение было `Python 3.9.25`.

Что исправлено:
- `pyproject.toml` переведён на `>=3.9`;
- заменён синтаксис `X | Y` на `Optional/Union`.

### 6.3 Runtime bug в adapters

Проблема:
- отсутствовали импорты `Union`, `Optional`, `List` в `adapters.py`;
- job на сервере падал на импорте.

Что исправлено:
- импорты добавлены.

### 6.4 Проблема с aspect ratio в grouped dataset

Проблема:
- при `Resize(image_size)` без `CenterCrop` картинки внутри одного track могли иметь разные `W`;
- `torch.stack(...)` в `SequenceFolderDataset` падал.

Текущее решение:
- для benchmark и ImageNet run включён `--center-crop`.

Вывод:
- для grouped training path uniform spatial size внутри track обязателен.

### 6.5 Буферизация stdout в slurm

Проблема:
- прогресс-лог в benchmark не был виден сразу в `slurm-*.out`;
- причина — буферизация stdout.

Решение:
- запускать benchmark через `python -u ...`.

### 6.6 Неправильный путь к логам slurm

Проблема:
- job запускались из `GRLNet`, а пользователь искал `slurm-*.out` во внешней директории.

Реальное поведение:
- `WorkDir` и `StdOut` указывали на каталог `GRLNet` в scratch.

### 6.7 Квота диска и временные артефакты ImageNet

Проблема:
- помимо `layout/train` и `layout/val`, место занимали:
  - исходные `.tar`;
  - `.prepare_tmp`;
  - `devkit`.

Вывод:
- после завершения подготовки и запуска benchmark эти артефакты нужно удалять.

## 7. Наиболее вероятные bottleneck'и текущего кода

### Очень сильные
- `SequenceFolderDataset`: `Image.open + JPEG decode + transform` на множество мелких файлов;
- `prep_batch()` на CPU;
- `RandomRotation` в preprocessing.

### Средние
- `.item()` / `.sum(...).item()` как GPU sync на каждом batch;
- `stack + zeros + cat` в формировании track;
- сетевое storage и latency на мелких файлах.

### Низкие
- `pin_memory=True` на CPU run;
- прогресс-лог;
- `matplotlib` import.

## 8. Архитектурная дилемма проекта

Проект одновременно должен:
- быть понятным как академическая reference implementation;
- быть пригодным для прикладного обучения на ImageNet-scale данных;
- оставаться CPU/GPU совместимым в базовом варианте;
- не перегружать основной trainer DDP/checkpoint/cache логикой.

Уже принятое рабочее понимание:
- ядро библиотеки должно оставаться чистым и читаемым;
- оптимизационный large-scale training stack не должен ломать читаемость core API;
- official recipe для выпуска весов может жить отдельно от core, но должен оставаться публичным.

## 9. Что НЕ нужно ломать в ядре

Следующие вещи считаются полезными и должны остаться понятными:
- `GRLClassifier` и `ConvLSTMCell`;
- `SequenceFolderDataset` как основной grouped dataset;
- `fit_reference(...)` как reference trainer;
- CPU/GPU совместимость базового режима;
- понятные adapters и predict helpers.

## 10. Что нужно помнить следующему агенту

1. Не сравнивать raw accuracy на `IRUO` и `ImageNet-1000` напрямую без учёта числа классов.
2. Для official ImageNet `val` нужен порядок классов из `meta.mat`, а не сортировка папок `train`.
3. `gold` для single-image и `gold-like padding policy` для group/video — разные вещи.
4. Не стоит без необходимости усложнять основной trainer DDP/checkpoint логикой.
5. Для large-scale обучения правильнее идти в сторону отдельного recipe layer, а не перегружать core.
6. Если следующий шаг — оптимизированное обучение, смотреть файл `docs/optimization_v2_plan.md`.
