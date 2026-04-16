"""Minimal GRLNet/StabHRec40 ImageNet-1K inference example."""

from pathlib import Path

from PIL import Image

from grlnet import GRLNetWeights, grlnet_stabhrec40
from grlnet.inference import decode_topk, load_categories, predict_image, topk

image_path = Path("sample.jpg")

model = grlnet_stabhrec40(weights=GRLNetWeights.DEFAULT)
image = Image.open(image_path).convert("RGB")
logits = predict_image(model, image)

categories = load_categories()
for item in decode_topk(topk(logits, k=5), categories):
    print(f"{item['class_id']:4d}  {item['score']:.4f}  {item['label']}")
