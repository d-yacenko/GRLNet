from torchvision import transforms

from grl_model.data.datasets import (
    ImageFolderPseudoTrackDataset,
    SequenceFolderDataset,
    TrackFolderDataset,
)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# 1. Standard image dataset -> pseudo-track per image
# 1. Стандартный image dataset -> один pseudo-track на изображение
imagefolder_ds = ImageFolderPseudoTrackDataset(
    "/path/to/imagefolder/train",
    track_length=10,
    image_transform=transform,
)

# 2. Notebook-faithful grouped-by-class sequence dataset
# 2. Notebook-совместимый датасет последовательностей, сгруппированных по классу
sequence_ds = SequenceFolderDataset(
    "/path/to/imagefolder/train",
    seq_len=10,
    transform=transform,
)

# 3. Explicit tracks stored as root/class/track/frame.jpg
# 3. Явные треки, сохранённые как root/class/track/frame.jpg
trackfolder_ds = TrackFolderDataset(
    "/path/to/track_dataset",
    track_length=10,
    image_transform=transform,
)

print(len(imagefolder_ds), len(sequence_ds), len(trackfolder_ds))
