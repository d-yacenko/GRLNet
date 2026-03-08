"""Minimal training example. / Минимальный пример обучения."""

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
