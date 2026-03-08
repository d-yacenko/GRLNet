# Datasets / Датасеты

## 1. `ImageFolderPseudoTrackDataset`

EN: Utility dataset for image-to-pseudo-track conversion. Each image becomes one pseudo-track.

RU: Вспомогательный датасет для преобразования одиночных картинок в pseudo-track. Каждая картинка становится одним псевдо-треком.

## 2. `SequenceFolderDataset`

EN: This is the notebook-faithful dataset. It groups samples by class, shuffles inside class buckets, chunks them into sequences, and converts them to notebook track format.

RU: Это faithful-перенос notebook-датасета. Он группирует изображения по классам, перемешивает внутри class-bucket’ов, режет на последовательности и затем переводит их в notebook-формат трека.

Important / Важно:

- EN: not a natural video dataset
  RU: это не natural video dataset
- EN: supports `on_epoch_end()` reshuffle
  RU: поддерживает reshuffle через `on_epoch_end()`

- EN: this is the primary dataset abstraction behind the public trainer
  RU: это основная датасетная абстракция, лежащая в основе публичного trainer'а

## 3. `TrackFolderDataset`

EN: Use this when tracks already exist explicitly as `root/class_name/track_id/frame.jpg`.

RU: Используйте этот вариант, когда треки уже существуют явно в виде `root/class_name/track_id/frame.jpg`.

## Which one to choose / Что выбирать

- EN: `SequenceFolderDataset` for the main training workflow and grouped-observation learning
  RU: `SequenceFolderDataset` для основного training workflow и grouped-observation обучения
- EN: `ImageFolderPseudoTrackDataset` only for auxiliary pseudo-track scenarios
  RU: `ImageFolderPseudoTrackDataset` только для вспомогательных pseudo-track сценариев
- EN: `TrackFolderDataset` for real grouped observations
  RU: `TrackFolderDataset` для реальных сгруппированных наблюдений
