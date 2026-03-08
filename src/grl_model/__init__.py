"""Public package surface for the GRL track classifier. / Публичная поверхность пакета GRL track classifier.

The package is centered around a track-form ConvLSTM classifier and a small set of
consumer-facing helpers for:
Пакет построен вокруг ConvLSTM-классификатора, работающего с треками, и набора
пользовательских утилит для:

- dataset construction / построения датасетов
- notebook-compatible training / обучения, совместимого с ноутбуком
- prediction on images, grouped images, and tracks /
  предикта по одиночным изображениям, группам изображений и трекам
"""

from .models import ConvLSTMCell, GRLClassifier, GRLWeights, grl_base, grl_tiny

__all__ = ["ConvLSTMCell", "GRLClassifier", "GRLWeights", "grl_base", "grl_tiny"]
