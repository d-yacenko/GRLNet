from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from grlnet.inference import decode_topk, load_categories, load_model, predict_image, topk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRLNet/StabHRec40 prediction for one image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Local .pth checkpoint.")
    parser.add_argument("--weights", default=None, help="Registered weights name or release URL descriptor.")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--categories", type=Path, default=None, help="Optional newline-delimited class labels.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(
        checkpoint=args.checkpoint,
        weights=args.weights,
        num_classes=args.num_classes,
        device=device,
    )
    image = Image.open(args.image).convert("RGB")
    logits = predict_image(
        model,
        image,
        image_size=args.image_size,
        resize_size=args.resize_size,
        device=device,
    )
    categories = load_categories(args.categories)
    print(json.dumps(decode_topk(topk(logits, k=args.topk), categories), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
