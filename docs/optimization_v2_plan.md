# GRLNet v2: План оптимизированного обучения и выпуска весов ImageNet-1k

Этот документ описывает целевую `v2`-архитектуру репозитория и план работ для получения полноценных pretrained weights на `ImageNet-1k` без деградации академической читаемости проекта.

## 1. Задача

Нужно разрешить противоречие между двумя требованиями.

### Требование A: академическое ядро
Нужен чистый, понятный, короткий и расширяемый код, чтобы:
- студент мог его понять;
- исследователь мог унаследовать модель, датасет и trainer под свой проект;
- проект оставался CPU/GPU-совместимым;
- reference training path оставался прозрачным.

### Требование B: practical large-scale training
Нужен код, который позволяет:
- учить модель на `ImageNet-1k` в разумные сроки;
- использовать `DDP` на нескольких GPU;
- делать checkpoint/resume;
- продолжать обучение на другой инфраструктуре и на другой GPU;
- поддерживать long-running jobs на кластере;
- контролировать bottleneck'и data pipeline.

Эти требования нельзя безболезненно сплавить в один trainer без ухудшения читаемости.

## 2. Принятое архитектурное решение

### 2.1 Не менять философию core
Текущее ядро остаётся:
- `src/grl_model/...`
- `fit_reference(...)`
- `fit_reference_imagefolders(...)`
- `SequenceFolderDataset`
- текущие adapters/predict helpers

Это остаётся reference implementation.

### 2.2 Добавить отдельный recipe layer
Во `v2` нужно создать отдельный слой для official training recipes.

Рекомендуемая структура:

```text
GRLNet/
  src/grl_model/              # чистое ядро библиотеки
  docs/
  examples/
  recipes/
    imagenet/
      README.md
      config_base.py
      train_ddp.py
      resume.py
      checkpointing.py
      data_pipeline.py
      launch/
        slurm_a100.sh
        slurm_t4_ddp.sh
      configs/
        grl_imagenet_a100.yaml
        grl_imagenet_t4_ddp.yaml
```

Принцип:
- `src/` — библиотека;
- `recipes/` — официальный large-scale training stack.

Именно `recipes/imagenet/...` должны становиться источником для выпущенных весов.

## 3. Почему это решение правильно

1. Публичное ядро остаётся читаемым.
2. Official training code остаётся публичным и воспроизводимым.
3. Веса не будут “получены на каком-то приватном коде”.
4. CPU-only пользователь по-прежнему сможет пользоваться core API.
5. GPU/DDP/cluster-специфика не загрязнит основной trainer.

## 4. Целевой статус v2

Во `v2` должны сосуществовать два официальных режима.

### 4.1 Reference mode
Назначение:
- исследования;
- быстрые эксперименты;
- небольшие датасеты;
- CPU/GPU;
- понятный код.

Артефакты:
- `fit_reference(...)`
- `fit_reference_imagefolders(...)`
- `SequenceFolderDataset`

### 4.2 Official optimized ImageNet recipe
Назначение:
- pretraining на `ImageNet-1k`;
- выпуск официальных весов;
- работа на кластере;
- DDP;
- checkpoint/resume;
- производительность.

Артефакты:
- `recipes/imagenet/train_ddp.py`
- `recipes/imagenet/checkpointing.py`
- `recipes/imagenet/data_pipeline.py`
- `recipes/imagenet/configs/*.yaml`

## 5. Известная инфраструктура, на которой уже работали

На кластере уже были доступны следующие ресурсы:

### CPU ноды
- partition: `amd_1Tb`
- пример ноды: `hydra-brain1`, `hydra-brain3`, `hydra-brain4`
- `256 CPU`
- около `1 TB RAM`

### GPU ноды
- partition: `gpu_T4`
- пример: `hydra-gpu1`
- `2 x T4`

- partition: `gpu_A100`
- пример: `hydra-gpu2`, `hydra-gpu3`
- `2 x A100`

### Наблюдаемые ограничения
- storage сетевой;
- много мелких JPEG-файлов;
- квота места ограничена;
- stdout/slurm требует учитывать buffering;
- для `ImageNet val` нужен правильный mapping из devkit.

## 6. Фактические измерения, уже полученные

### 6.1 1 x T4
Из `benchmark_summary.json`:
- `elapsed_sec ≈ 33049`
- `~9.18 часа / эпоха`

Пересчёт:
- `30 эпох ≈ 11.5 суток`
- `100 эпох ≈ 38.3 суток`

### 6.2 CPU large-RAM node
По progress benchmark:
- `~5.5-6 суток / эпоха`

Пересчёт:
- `30 эпох ≈ 171-180 суток`
- `100 эпох ≈ 570-600 суток`

Вывод:
- CPU path непригоден для реального pretraining;
- T4 пригоден только как fallback или benchmark;
- реальный production path должен ориентироваться на `A100` или multi-GPU recipe.

## 7. Текущие bottleneck'и и их приоритет

## 7.1 Самые сильные bottleneck'и

### A. `SequenceFolderDataset` на сетевом storage
Проблема:
- на каждый sample читается `seq_len` JPEG;
- множество `open/decode/transform` операций;
- высокая цена работы с мелкими файлами.

Оценка вклада:
- очень высокий.

### B. `prep_batch()` на CPU
Проблема:
- Python-циклы по batch и active frames;
- применение аугментаций до `inputs.to(device)`;
- дорого на ImageNet масштабе.

Оценка вклада:
- очень высокий.

### C. `RandomRotation`
Проблема:
- дорогая CPU augmentation;
- используется и в train transform, и в `gold`-логике.

Оценка вклада:
- высокий.

## 7.2 Средние bottleneck'и

### D. `.item()` на каждом batch
Проблема:
- sync GPU с CPU при сборе статистик.

Оценка вклада:
- средний.

### E. Track assembly в dataset
Проблема:
- `stack + zeros + cat` на каждый sample.

Оценка вклада:
- средний.

## 7.3 Низкие bottleneck'и
- `pin_memory=True` на CPU;
- progress logging;
- `matplotlib` import.

## 8. План оптимизации по пакетам работ

## 8.1 Пакет M1: минимальные безопасные ускорения

Цель:
- получить выигрыш без ломки логики модели.

Изменения:
1. `recipes/imagenet/train_ddp.py` как отдельный launcher.
2. `python -u` и structured progress logging.
3. `pin_memory = (device.type == "cuda")`.
4. убрать лишние batch-level `.item()` sync или сократить их частоту.
5. `persistent_workers=True`, `prefetch_factor`, tuning числа workers.
6. централизованная конфигурация run через YAML.

Объём кода:
- примерно `100-180` строк.

Ожидаемая экономия:
- `8-20%`.

Риск для качества:
- низкий.

## 8.2 Пакет M2: оптимизация preprocessing без ломки core

Цель:
- убрать главный CPU bottleneck, но не переписать всё ядро.

Изменения:
1. В `recipes/imagenet/data_pipeline.py` сделать ускоренный dataset wrapper.
2. Добавить lazy RAM cache:
   - либо JPEG bytes cache;
   - либо decoded image cache.
3. Отдельно реализовать accelerated `gold` preprocessing для recipe layer.
4. При возможности перевести `prep_batch`-подобную логику на tensor-level/GPU-side path внутри recipe.

Объём кода:
- примерно `180-350` строк.

Ожидаемая экономия:
- `20-50%` в зависимости от storage.

Риск для качества:
- средний, если изменится точная семантика `gold`.

Требование:
- семантика recipe должна быть документирована и проверена на эквивалентность reference path на малом эксперименте.

## 8.3 Пакет M3: checkpoint/resume для длинных прогонов

Цель:
- сделать обучение прерываемым и переносимым между инфраструктурами.

Изменения:
1. `recipes/imagenet/checkpointing.py`
2. checkpoint должен сохранять:
   - `model.state_dict()`
   - `optimizer.state_dict()`
   - `scheduler.state_dict()`
   - `scaler.state_dict()`
   - `epoch`
   - `history`
   - `rng state`
   - config
   - split-state / dataset-state, если split генерировался внутри recipe
3. `resume.py` должен уметь:
   - продолжить run с того же checkpoint;
   - продолжить run на другой GPU;
   - продолжить run без DDP после DDP или наоборот, если модель грузится через `state_dict` без жёсткой привязки к `module.`

Объём кода:
- примерно `140-260` строк.

Ожидаемая экономия времени:
- не ускоряет одну эпоху;
- резко уменьшает риск потери уже потраченных суток вычислений.

Практическая ценность:
- очень высокая.

## 8.4 Пакет M4: DDP для multi-GPU

Цель:
- научить модель эффективно использовать несколько GPU.

Изменения:
1. `train_ddp.py`
2. `DistributedSampler`
3. `set_epoch(...)` для sampler
4. rank-aware logging
5. checkpointing только с rank 0
6. корректная загрузка `state_dict` между single-GPU и DDP
7. cluster launcher scripts

Объём кода:
- примерно `180-320` строк.

Ожидаемый выигрыш:
- сам по себе `2.5x - 3.2x` ускорение на `4 x T4` относительно `1 x T4`, но только если bottleneck не в data pipeline.

Риск:
- средний.

Ключевая оговорка:
- делать только после M1/M2, иначе scaling будет плохим.

## 9. Порядок внедрения

Рекомендуемый порядок:

1. `M1` — минимальные безопасные ускорения
2. `M3` — checkpoint/resume
3. `M2` — data pipeline / preprocessing optimization
4. повторный single-GPU benchmark
5. `M4` — DDP

Почему так:
- сначала надо сделать обучение устойчивым и наблюдаемым;
- затем снизить риск потерять длинный run;
- затем ускорять bottleneck;
- затем масштабироваться на несколько GPU.

## 10. Как решается вопрос академичности

Во `v2` нужно явно зафиксировать два типа кода.

### Core library
- читаемость выше скорости;
- CPU/GPU совместимость;
- reference implementation;
- хороша для чтения, наследования и paper-level reproducibility.

### Official recipe layer
- throughput выше простоты;
- допускает DDP, checkpointing, cache;
- может быть GPU-first;
- именно на нём обучаются выкладываемые weights.

Это академически корректно, если:
1. recipe публичен;
2. config публичен;
3. commit публичен;
4. веса ссылаются на точный recipe.

## 11. Как публиковать веса корректно

При релизе weights нужно публиковать:
- commit SHA;
- точный training config;
- hardware summary;
- batch size;
- optimizer/scheduler;
- число эпох;
- checkpoint policy;
- точный recipe entrypoint.

Пример формулировки:
- `Weights were trained with GRLNet commit <sha> using recipes/imagenet/train_ddp.py and config recipes/imagenet/configs/grl_imagenet_a100.yaml.`

Это полностью снимает проблему “веса получены не на том коде, который опубликован”.

## 12. Можно ли продолжать обучение на другой инфраструктуре

Да, и это должно быть явно заложено в recipe layer.

`v2` должен поддерживать сценарии:
- single GPU -> single GPU;
- single GPU -> DDP;
- DDP -> single GPU;
- T4 -> A100;
- A100 -> T4;
- кластер A -> кластер B.

Для этого нужны:
- `state_dict`-based checkpoints;
- аккуратная работа с `module.` префиксами;
- конфигурируемый batch size;
- понятная политика LR при смене batch size.

## 13. Как относиться к смене batch size при resume

Смена batch size допустима.

Что это значит:
- обучение не ломается;
- trajectory уже не будет побитово той же;
- но как practical resume это валидно.

Рекомендация для `v2`:
- явно документировать, что batch size можно менять при resume;
- при большой смене batch size позволять задавать новый LR через config.

## 14. Что НЕ нужно делать

1. Не перегружать `fit_reference(...)` DDP/checkpoint/cache логикой.
2. Не делать private training stack, если на его основе будут публиковаться official weights.
3. Не смешивать в одном trainer:
   - educational clarity;
   - cluster-resume fault tolerance;
   - DDP;
   - aggressive caching;
   - throughput hacks.

## 15. Целевая дорожная карта

### Этап 1
- зафиксировать текущее ядро как `reference core`;
- добавить внутренний `recipes/imagenet/README.md` с архитектурой recipe layer.

### Этап 2
- реализовать `M1 + M3`;
- научиться безопасно запускать и возобновлять long runs.

### Этап 3
- реализовать `M2`;
- сравнить single-GPU throughput до и после.

### Этап 4
- реализовать `M4`;
- получить `4 x T4` и/или `A100` training recipe.

### Этап 5
- обучить final ImageNet weights;
- опубликовать weights вместе с recipe/config/commit.

## 16. Ожидаемые сроки после оптимизаций

По уже сделанным оценкам:
- `1 x T4`: ~`9.18 часа/эпоха`
- `CPU`: ~`5.5-6 суток/эпоха`

При `4 x T4 + DDP + оптимизация preprocessing` ожидаемый диапазон:
- `~1.6 - 2.8 часа/эпоха`

Грубый пересчёт:
- `50 эпох ≈ 3.3 - 5.8 суток`
- `100 эпох ≈ 6.7 - 11.7 суток`

Это уже делает ImageNet pretraining практичным.

## 17. Итоговое решение дилеммы

Оптимальное решение:
- не усложнять текущее ядро;
- не держать optimized training private;
- создать `v2` с разделением:
  - `core library`;
  - `official recipe layer`.

Именно так можно одновременно сохранить:
- академическую ясность;
- инженерную пригодность;
- воспроизводимость weights;
- масштабируемость на ImageNet.
