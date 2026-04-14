"""Minimal GRLNet/StabHRec40 inference example."""

from PIL import Image

from grlnet.inference import decode_topk, load_categories, load_model, predict_image, topk

model = load_model(
    checkpoint="stabhrec40_a100_single_50e/stabhrec40_a100_single_50e_best.pth",
    num_classes=1000,
)

image = Image.open("sample.jpg").convert("RGB")
logits = predict_image(model, image, image_size=224, resize_size=224)

categories = load_categories()
for item in decode_topk(topk(logits, k=5), categories):
    print(f"{item['class_id']:4d}  {item['score']:.4f}  {item['label']}")
