from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_eval_transform(*, image_size: int = 224, resize_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(resize_size)),
            transforms.CenterCrop(int(image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def imagenet_train_transform(*, image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(int(image_size), scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "imagenet_eval_transform",
    "imagenet_train_transform",
]
