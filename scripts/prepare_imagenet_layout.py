from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path


def extract_tar(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(src) as tf:
        tf.extractall(dst)


def build_train_layout(train_archive: Path, train_root: Path, tmp_root: Path) -> None:
    stage_root = tmp_root / "train_stage"
    extract_tar(train_archive, stage_root)
    class_tars = sorted(stage_root.glob("*.tar"))
    if not class_tars:
        raise RuntimeError(f"No class tar files found in {stage_root}")

    for class_tar in class_tars:
        wnid = class_tar.stem
        class_dir = train_root / wnid
        class_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(class_tar) as tf:
            tf.extractall(class_dir)


def reorganize_flat_val(val_stage: Path, val_root: Path, val_ground_truth: Path, wnids_file: Path) -> None:
    wnids = [line.strip() for line in wnids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(wnids) != 1000:
        raise RuntimeError(f"Expected 1000 wnids in {wnids_file}, got {len(wnids)}")

    labels = [int(line.strip()) for line in val_ground_truth.read_text(encoding="utf-8").splitlines() if line.strip()]
    images = sorted([p for p in val_stage.iterdir() if p.is_file()])
    if len(images) != len(labels):
        raise RuntimeError(f"Val images count {len(images)} != labels count {len(labels)}")

    for image_path, cls_idx in zip(images, labels):
        wnid = wnids[cls_idx - 1]
        dst_dir = val_root / wnid
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(image_path), str(dst_dir / image_path.name))


def build_val_layout(val_archive: Path, val_root: Path, tmp_root: Path, val_ground_truth: Path | None, wnids_file: Path | None) -> None:
    stage_root = tmp_root / "val_stage"
    extract_tar(val_archive, stage_root)

    class_dirs = sorted([p for p in stage_root.iterdir() if p.is_dir()])
    if class_dirs:
        for class_dir in class_dirs:
            dst_dir = val_root / class_dir.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for item in class_dir.iterdir():
                target = dst_dir / item.name
                if item.is_file():
                    shutil.move(str(item), str(target))
        return

    if val_ground_truth is None or wnids_file is None:
        raise RuntimeError(
            "Val archive is flat. Provide --val-ground-truth and --wnids-file to reorganize it."
        )
    reorganize_flat_val(stage_root, val_root, val_ground_truth, wnids_file)


def summarize_layout(train_root: Path, val_root: Path) -> dict[str, object]:
    train_classes = sorted([p.name for p in train_root.iterdir() if p.is_dir()])
    val_classes = sorted([p.name for p in val_root.iterdir() if p.is_dir()])
    return {
        "train_root": str(train_root),
        "val_root": str(val_root),
        "train_classes": len(train_classes),
        "val_classes": len(val_classes),
        "same_class_order": train_classes == val_classes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ImageNet archives into ImageFolder-style train/val layout.")
    parser.add_argument("--train-archive", type=Path, required=True)
    parser.add_argument("--val-archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--val-ground-truth", type=Path, default=None)
    parser.add_argument("--wnids-file", type=Path, default=None)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    train_root = args.output_root / "train"
    val_root = args.output_root / "val"
    tmp_root = args.output_root / ".prepare_tmp"

    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    build_train_layout(args.train_archive, train_root, tmp_root)
    build_val_layout(args.val_archive, val_root, tmp_root, args.val_ground_truth, args.wnids_file)

    summary = summarize_layout(train_root, val_root)
    print(json.dumps(summary, indent=2))

    if not args.keep_temp:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
