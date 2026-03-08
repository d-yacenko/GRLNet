# Core Concepts / Основные понятия

## What this model consumes / Что потребляет модель

EN: The model does not operate on a single image in the same way as a conventional image classifier. Its native input is a **track**.

RU: Модель не работает с одной картинкой так же, как обычный image classifier. Её нативный вход — **трек**.

Canonical tensor shape / Каноническая форма тензора:

```python
[B, T, C, H, W]
```

## What a track means here / Что здесь означает трек

EN: A track is a small group of images that all refer to the same entity, scene, or observation.

RU: Трек — это небольшая группа изображений, все элементы которой относятся к одной сущности, сцене или наблюдению.

EN: The key property is semantic identity, not strict temporal order.

RU: Ключевое свойство — семантическая идентичность, а не строгий временной порядок.

Examples / Примеры:

- EN: one suitcase observed from several X-ray viewpoints
  RU: один чемодан, наблюдаемый с нескольких рентген-ракурсов
- EN: one object seen by several surveillance cameras
  RU: один объект, наблюдаемый несколькими камерами
- EN: one wildfire or smoke plume observed through a grouped set of images
  RU: один лесной пожар или шлейф дыма, наблюдаемый через группу изображений
- EN: one still image converted into a pseudo-track
  RU: одна картинка, преобразованная в псевдо-трек

## Why this is not a standard video model / Почему это не обычная video-модель

EN: In many video models, the temporal order itself is a major source of information.

RU: Во многих video-моделях временной порядок сам по себе является важным источником информации.

EN: In GRL, the main goal is feature refinement across grouped observations of the same entity.

RU: В GRL главная цель — уточнение признаков на группе наблюдений одной и той же сущности.

## Gold protocol / Gold-протокол

EN: The notebook-compatible track format uses an active first third and a zero-padded last two thirds.

RU: Совместимый с ноутбуком формат трека использует активную первую треть и заполненную нулями оставшуюся часть.

EN: During the gold protocol, the active part is rebuilt from one selected anchor frame via `prep_batch()`.

RU: Во время gold-протокола активная часть перестраивается из одного выбранного опорного кадра через `prep_batch()`.

## Single-image inference / Инференс по одной картинке

EN: A single image is supported through a pseudo-track adapter.

RU: Одна картинка поддерживается через адаптер псевдо-трека.
