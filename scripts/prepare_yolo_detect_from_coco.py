import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CLASS_NAMES = [
    "Acne",
    "Blackheads",
    "Dark-Spots",
    "Dry-Skin",
    "Enlarged-Pores",
    "Eyebags",
    "Oily-Skin",
    "Skin-Redness",
    "Whiteheads",
    "Wrinkles",
]
RENAMED_CATEGORY_NAMES = {"Englarged-Pores": "Enlarged-Pores"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Roboflow COCO bounding boxes to Ultralytics YOLO detection format."
    )
    parser.add_argument("--coco-dir", type=Path, default=Path("data/skincaria-dataset.coco"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/skincaria-yolo-detect"))
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-images", action="store_true")
    return parser.parse_args()


def should_ignore_category(name: str) -> bool:
    return name.startswith("Acne-Blackhead-Wrinkles-")


def normalize_category_name(name: str) -> str:
    return RENAMED_CATEGORY_NAMES.get(name, name)


def find_annotation_path(split_dir: Path) -> Path:
    candidates = sorted(split_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No COCO JSON annotation file found in {split_dir}")
    for candidate in candidates:
        if "annotation" in candidate.name.lower():
            return candidate
    return candidates[0]


def build_category_mapping(coco: dict[str, Any]) -> dict[int, int]:
    output_index_by_name = {name: index for index, name in enumerate(DEFAULT_CLASS_NAMES)}
    mapping: dict[int, int] = {}
    for category in coco.get("categories", []):
        original_name = str(category.get("name", ""))
        if should_ignore_category(original_name):
            continue
        name = normalize_category_name(original_name)
        if name not in output_index_by_name:
            raise ValueError(f"Unexpected COCO category: {original_name}")
        mapping[int(category["id"])] = output_index_by_name[name]
    return mapping


def ensure_clean_output(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{out_dir} already exists and is not empty. Re-run with --overwrite to rebuild it."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def link_or_copy_image(source: Path, destination: Path, copy_images: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if copy_images:
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def bbox_to_yolo_line(class_index: int, bbox: Any, width: int, height: int) -> str | None:
    if not isinstance(bbox, list) or len(bbox) < 4 or width <= 0 or height <= 0:
        return None
    try:
        x, y, box_width, box_height = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if box_width <= 0 or box_height <= 0:
        return None

    x_center = (x + box_width / 2) / width
    y_center = (y + box_height / 2) / height
    norm_width = box_width / width
    norm_height = box_height / height
    values = [x_center, y_center, norm_width, norm_height]
    values = [min(max(value, 0.0), 1.0) for value in values]
    return " ".join([str(class_index), *[f"{value:.6f}" for value in values]])


def write_dataset_yaml(out_dir: Path) -> Path:
    yaml_path = out_dir / "data.yaml"
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(DEFAULT_CLASS_NAMES))
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve().as_posix()}",
                "train: train/images",
                "val: val/images",
                "",
                "names:",
                names,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    args = parse_args()
    split_dir = args.coco_dir / args.source_split
    annotation_path = find_annotation_path(split_dir)
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    category_mapping = build_category_mapping(coco)

    ensure_clean_output(args.out_dir, args.overwrite)

    image_info_by_id = {int(item["id"]): item for item in coco.get("images", [])}
    annotations_by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        category_id = int(annotation.get("category_id", -1))
        if category_id in category_mapping:
            annotations_by_image_id[int(annotation["image_id"])].append(annotation)

    image_ids = [
        image_id
        for image_id, item in image_info_by_id.items()
        if (split_dir / item["file_name"]).exists()
        and Path(item["file_name"]).suffix.lower() in IMAGE_EXTENSIONS
    ]
    random.Random(args.seed).shuffle(image_ids)
    val_count = max(1, int(len(image_ids) * args.val_ratio)) if len(image_ids) > 1 else 0
    val_ids = set(image_ids[:val_count])

    written_annotations = 0
    skipped_annotations = 0
    for image_id in image_ids:
        item = image_info_by_id[image_id]
        file_name = item["file_name"]
        split = "val" if image_id in val_ids else "train"

        source_image = split_dir / file_name
        target_image = args.out_dir / split / "images" / file_name
        target_label = args.out_dir / split / "labels" / f"{Path(file_name).stem}.txt"
        target_label.parent.mkdir(parents=True, exist_ok=True)
        link_or_copy_image(source_image, target_image, args.copy_images)

        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        lines: list[str] = []
        for annotation in annotations_by_image_id.get(image_id, []):
            class_index = category_mapping[int(annotation["category_id"])]
            line = bbox_to_yolo_line(class_index, annotation.get("bbox"), width, height)
            if line is None:
                skipped_annotations += 1
                continue
            lines.append(line)
            written_annotations += 1
        target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    yaml_path = write_dataset_yaml(args.out_dir)
    print(f"source: {annotation_path}")
    print(f"output: {args.out_dir}")
    print(f"yaml: {yaml_path}")
    print(f"images: {len(image_ids)}")
    print(f"train_images: {len(image_ids) - len(val_ids)}")
    print(f"val_images: {len(val_ids)}")
    print(f"written_annotations: {written_annotations}")
    print(f"skipped_annotations: {skipped_annotations}")
    print(f"classes: {DEFAULT_CLASS_NAMES}")


if __name__ == "__main__":
    main()
