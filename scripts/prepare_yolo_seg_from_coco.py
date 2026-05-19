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
IGNORED_CATEGORY_NAMES = {"Acne-Blackhead-Wrinkles-f6HR-vJCz-Vyox"}
RENAMED_CATEGORY_NAMES = {"Englarged-Pores": "Enlarged-Pores"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the Roboflow COCO segmentation export to Ultralytics YOLO segmentation format."
    )
    parser.add_argument("--coco-dir", type=Path, default=Path("data/skincaria-dataset.coco"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/skincaria-yolo-seg"))
    parser.add_argument("--source-split", default="train", help="Roboflow COCO split folder to convert.")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--bbox-as-segment",
        action="store_true",
        help=(
            "When a COCO annotation has no polygon segmentation, convert its bbox to a "
            "rectangular segmentation polygon. Useful as a baseline, but not as precise as true masks."
        ),
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of creating hardlinks. Uses more disk space but works across drives.",
    )
    return parser.parse_args()


def normalized_category_name(name: str) -> str:
    return RENAMED_CATEGORY_NAMES.get(name, name)


def should_ignore_category(name: str) -> bool:
    return name in IGNORED_CATEGORY_NAMES or name.startswith("Acne-Blackhead-Wrinkles-")


def find_annotation_path(split_dir: Path) -> Path:
    candidates = sorted(split_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No COCO JSON annotation file found in {split_dir}")
    for candidate in candidates:
        if "annotation" in candidate.name.lower():
            return candidate
    return candidates[0]


def load_coco(annotation_path: Path) -> dict[str, Any]:
    return json.loads(annotation_path.read_text(encoding="utf-8"))


def build_category_mapping(coco: dict[str, Any]) -> dict[int, int]:
    output_index_by_name = {name: index for index, name in enumerate(DEFAULT_CLASS_NAMES)}
    category_mapping: dict[int, int] = {}

    for category in coco.get("categories", []):
        original_name = str(category.get("name", ""))
        if should_ignore_category(original_name):
            continue
        name = normalized_category_name(original_name)
        if name not in output_index_by_name:
            raise ValueError(f"Unexpected COCO category: {original_name}")
        category_mapping[int(category["id"])] = output_index_by_name[name]

    return category_mapping


def polygon_area(flat_points: list[float]) -> float:
    points = list(zip(flat_points[0::2], flat_points[1::2]))
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def choose_largest_polygon(segmentation: Any) -> list[float] | None:
    if not isinstance(segmentation, list):
        return None

    polygons: list[list[float]] = []
    if segmentation and all(coerce_float(value) is not None for value in segmentation):
        polygons = [[float(value) for value in segmentation]]
    else:
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6:
                continue
            values = [coerce_float(value) for value in polygon]
            if any(value is None for value in values):
                continue
            polygons.append([float(value) for value in values if value is not None])

    if not polygons:
        return None
    return max(polygons, key=polygon_area)


def polygon_is_plausible(polygon: list[float], width: int, height: int) -> bool:
    if width <= 0 or height <= 0 or len(polygon) < 6:
        return False
    xs = polygon[0::2]
    ys = polygon[1::2]
    # Allow a small export-rounding margin, but reject obviously malformed
    # coordinates such as string-concatenated "15224.00" values.
    return (
        min(xs) >= -1
        and min(ys) >= -1
        and max(xs) <= width + 1
        and max(ys) <= height + 1
    )


def bbox_to_polygon(bbox: Any) -> list[float] | None:
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    x, y, width, height = [float(value) for value in bbox[:4]]
    if width <= 0 or height <= 0:
        return None
    return [
        x,
        y,
        x + width,
        y,
        x + width,
        y + height,
        x,
        y + height,
    ]


def polygon_to_yolo_line(class_index: int, polygon: list[float], width: int, height: int) -> str | None:
    if width <= 0 or height <= 0 or len(polygon) < 6:
        return None

    normalized: list[str] = []
    for x, y in zip(polygon[0::2], polygon[1::2]):
        nx = min(max(float(x) / width, 0.0), 1.0)
        ny = min(max(float(y) / height, 0.0), 1.0)
        normalized.extend([f"{nx:.6f}", f"{ny:.6f}"])

    if len(normalized) < 6:
        return None
    return " ".join([str(class_index), *normalized])


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
    coco = load_coco(annotation_path)
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

    skipped_rle_or_empty = 0
    bbox_as_segment_annotations = 0
    invalid_polygon_fallback_annotations = 0
    written_annotations = 0
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
            polygon = choose_largest_polygon(annotation.get("segmentation"))
            if polygon is not None and not polygon_is_plausible(polygon, width, height):
                fallback_polygon = bbox_to_polygon(annotation.get("bbox"))
                if fallback_polygon is not None:
                    polygon = fallback_polygon
                    invalid_polygon_fallback_annotations += 1
                else:
                    skipped_rle_or_empty += 1
                    continue
            if polygon is None:
                if args.bbox_as_segment:
                    polygon = bbox_to_polygon(annotation.get("bbox"))
                    if polygon is not None:
                        bbox_as_segment_annotations += 1
                    else:
                        skipped_rle_or_empty += 1
                        continue
                else:
                    skipped_rle_or_empty += 1
                    continue
            line = polygon_to_yolo_line(class_index, polygon, width, height)
            if line is None:
                skipped_rle_or_empty += 1
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
    print(f"bbox_as_segment_annotations: {bbox_as_segment_annotations}")
    print(f"invalid_polygon_fallback_annotations: {invalid_polygon_fallback_annotations}")
    print(f"skipped_rle_or_empty_annotations: {skipped_rle_or_empty}")
    print(f"classes: {DEFAULT_CLASS_NAMES}")


if __name__ == "__main__":
    main()
