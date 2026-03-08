# Architecture / Архитектура

## Summary / Кратко

EN: The model is a ConvLSTM-based classifier over track-form inputs.

RU: Модель — это classifier на основе ConvLSTM для входов в track-форме.

Native contract / Нативный контракт:

```python
[B, T, C, H, W] -> [B, K]
```

## Main components / Основные компоненты

- EN: stacked `ConvLSTMCell` layers
  RU: стек слоёв `ConvLSTMCell`
- EN: spatial max-pooling between recurrent layers
  RU: пространственный max-pooling между рекуррентными слоями
- EN: adaptive global pooling at the head
  RU: адаптивный global pooling в голове
- EN: linear classifier over the final hidden state
  RU: линейный классификатор над финальным скрытым состоянием

## Important implementation details / Важные детали реализации

### `forget_bias`

EN: Each ConvLSTM cell includes a learned `forget_bias` initialized to `1.5`.

RU: Каждая ConvLSTM-ячейка содержит обучаемый `forget_bias`, инициализированный значением `1.5`.

### Notebook-compatible `prep_batch`

EN: The model exposes `prep_batch()` to reproduce the historical gold protocol.

RU: Модель предоставляет `prep_batch()` для воспроизведения исторического gold-протокола.

### Track length convention / Соглашение о длине трека

EN: `track_length` is the active part of the track. Full notebook-compatible length is `3 * track_length`.

RU: `track_length` — это активная часть трека. Полная длина в notebook-compatible режиме равна `3 * track_length`.

## Intended use / Предполагаемое использование

EN: The architecture is appropriate when several images jointly describe one semantic entity.

RU: Архитектура подходит, когда несколько изображений совместно описывают одну семантическую сущность.
