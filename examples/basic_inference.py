"""Minimal inference example. / Минимальный пример инференса."""

from PIL import Image
from torchvision import transforms

from grl_model.models import grl_base
from grl_model.utils.predict import predict_group, predict_image, predict_images, predict_track, predict_video
from grl_model.data.adapters import build_pseudotrack_from_image

model = grl_base(num_classes=1000, track_length=10).eval()

tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open("sample.jpg").convert("RGB")
logits_from_one = predict_image(model, image, track_length=10, image_transform=tfm, apply_gold=True)

track = build_pseudotrack_from_image(image, track_length=10, image_transform=tfm)
logits_from_track = predict_track(model, track, apply_gold=True)

frames = [image, image, image]
logits_from_group = predict_group(
    model,
    frames,
    track_length=10,
    image_transform=tfm,
    active_frame_transform=model.trans,
)

images = [image, image]
logits_from_batch = predict_images(model, images, track_length=10, image_transform=tfm, apply_gold=True)

video_path = "sample.mp4"
logits_from_video = predict_video(
    model,
    video_path,
    track_length=10,
    image_transform=tfm,
    active_frame_transform=model.trans,
    sampling="uniform",
)
