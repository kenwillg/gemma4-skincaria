import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils
from transformers import AutoModelForCausalLM


DEFAULT_MODEL = "tiiuae/Falcon-Perception"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
IGNORED_CATEGORY_NAMES = {"Acne-Blackhead-Wrinkles-f6HR-vJCz-Vyox"}
PROMPT_MAP = {
    "Acne": "acne",
    "Blackheads": "blackheads",
    "Dark-Spots": "dark spots",
    "Dry-Skin": "dry skin",
    "Englarged-Pores": "enlarged pores",
    "Enlarged-Pores": "enlarged pores",
    "Eyebags": "eye bags",
    "Oily-Skin": "oily skin",
    "Skin-Redness": "skin redness",
    "Whiteheads": "whiteheads",
    "Wrinkles": "wrinkles",
}
CLASS_COLORS = {
    "Acne": (239, 68, 68),
    "Blackheads": (31, 41, 55),
    "Dark-Spots": (146, 64, 14),
    "Dry-Skin": (245, 158, 11),
    "Englarged-Pores": (14, 165, 233),
    "Enlarged-Pores": (14, 165, 233),
    "Eyebags": (99, 102, 241),
    "Oily-Skin": (34, 197, 94),
    "Skin-Redness": (236, 72, 153),
    "Whiteheads": (226, 232, 240),
    "Wrinkles": (168, 85, 247),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Falcon Perception prompts for multiple COCO classes on the same images."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/skincaria-dataset.coco"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--min-classes", type=int, default=3)
    parser.add_argument("--max-dimension", type=int, default=512)
    parser.add_argument("--max-masks-per-class", type=int, default=5)
    parser.add_argument("--mode", choices=["annotated", "all"], default="annotated")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/falcon_coco_multiclass"))
    return parser.parse_args()


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "image"


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in ["arial.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def find_annotation_path(split_dir: Path) -> Path:
    candidates = sorted(split_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No COCO JSON found in {split_dir}")
    for candidate in candidates:
        if "annotation" in candidate.name.lower():
            return candidate
    return candidates[0]


def decode_mask(rle: dict) -> np.ndarray:
    rle = dict(rle)
    counts = rle.get("counts")
    if isinstance(counts, str):
        rle["counts"] = counts.encode("ascii")
    return mask_utils.decode(rle).astype(bool)


def mask_to_image(mask: np.ndarray, color: tuple[int, int, int], alpha: int) -> Image.Image:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*color, alpha)
    return Image.fromarray(rgba, mode="RGBA")


def load_candidates(annotation_path: Path, split_dir: Path, min_classes: int) -> tuple[list[dict], list[str]]:
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {
        item["id"]: item["name"]
        for item in coco.get("categories", [])
        if item.get("name") not in IGNORED_CATEGORY_NAMES
    }
    images = {item["id"]: item for item in coco.get("images", [])}
    classes_by_image: dict[int, set[str]] = defaultdict(set)

    for annotation in coco.get("annotations", []):
        class_name = categories.get(annotation.get("category_id"))
        if class_name:
            classes_by_image[annotation["image_id"]].add(class_name)

    candidates = []
    for image_id, class_names in classes_by_image.items():
        if len(class_names) < min_classes:
            continue
        image_info = images.get(image_id)
        if not image_info:
            continue
        path = split_dir / image_info["file_name"]
        if path.exists() and path.suffix.lower() in IMAGE_EXTENSIONS:
            candidates.append(
                {
                    "path": path,
                    "classes": sorted(class_names),
                    "class_count": len(class_names),
                }
            )

    candidates.sort(key=lambda item: (-item["class_count"], item["path"].name))
    return candidates, sorted(set(categories.values()))


def draw_multiclass_overlay(
    image: Image.Image,
    *,
    image_name: str,
    annotated_classes: list[str],
    class_predictions: dict[str, list[dict]],
) -> Image.Image:
    base = image.convert("RGBA")
    draw = ImageDraw.Draw(base)

    for class_name, predictions in class_predictions.items():
        color = CLASS_COLORS.get(class_name, (100, 116, 139))
        for pred in predictions:
            rle = pred.get("mask_rle")
            if not rle:
                continue
            mask = decode_mask(rle)
            overlay = mask_to_image(mask, color, alpha=82)
            if overlay.size != base.size:
                overlay = overlay.resize(base.size, Image.Resampling.NEAREST)
            base.alpha_composite(overlay)

    title_font = load_font(17)
    body_font = load_font(13)
    small_font = load_font(12)
    row_height = 24
    rows = list(class_predictions.items())
    panel_width = 360
    panel_height = max(base.height, 92 + row_height * len(rows))
    canvas = Image.new("RGB", (base.width + panel_width, panel_height), (248, 250, 252))
    canvas.paste(base.convert("RGB"), (0, 0))

    legend = ImageDraw.Draw(canvas)
    x = base.width + 18
    y = 16
    legend.line((base.width, 0, base.width, panel_height), fill=(203, 213, 225), width=1)
    legend.text((x, y), "Falcon multiclass prompts", fill=(15, 23, 42), font=title_font)
    y += 25
    legend.text((x, y), "Color = class, not instance", fill=(71, 85, 105), font=small_font)
    y += 21
    legend.text((x, y), f"Image: {image_name[:38]}", fill=(71, 85, 105), font=small_font)
    y += 30

    for class_name, predictions in rows:
        color = CLASS_COLORS.get(class_name, (100, 116, 139))
        display = class_name.replace("Englarged", "Enlarged").replace("-", " ")
        total_area = 0
        for pred in predictions:
            if pred.get("mask_rle"):
                total_area += int(decode_mask(pred["mask_rle"]).sum())
        count = len(predictions)
        gt_mark = "GT" if class_name in annotated_classes else "no GT"
        text = f"{display}: {count} masks, {total_area:,} px ({gt_mark})"
        legend.rounded_rectangle((x, y + 3, x + 16, y + 19), radius=3, fill=color)
        border = (15, 23, 42) if class_name == "Whiteheads" else color
        legend.rounded_rectangle((x, y + 3, x + 16, y + 19), radius=3, outline=border, width=1)
        legend.text((x + 26, y), text, fill=(15, 23, 42), font=body_font)
        y += row_height

    return canvas


def summarize_predictions(class_predictions: dict[str, list[dict]]) -> dict[str, dict]:
    summary = {}
    for class_name, predictions in class_predictions.items():
        total_area = 0
        for pred in predictions:
            if pred.get("mask_rle"):
                total_area += int(decode_mask(pred["mask_rle"]).sum())
        summary[class_name] = {
            "prompt": PROMPT_MAP.get(class_name, class_name),
            "prediction_count": len(predictions),
            "mask_area_pixels": total_area,
        }
    return summary


def main() -> None:
    args = parse_args()
    split_dir = args.dataset_dir / args.split
    annotation_path = find_annotation_path(split_dir)
    candidates, all_classes = load_candidates(annotation_path, split_dir, args.min_classes)
    if not candidates:
        raise RuntimeError(f"No images with at least {args.min_classes} classes found.")

    selected = candidates[: args.limit]
    out_dir = args.out_dir / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
    ).to(device).eval()

    report = {
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "annotation_path": str(annotation_path),
        "mode": args.mode,
        "device": device,
        "max_dimension": args.max_dimension,
        "max_masks_per_class": args.max_masks_per_class,
        "images": [],
    }

    for item in selected:
        image_path = item["path"]
        image = Image.open(image_path).convert("RGB")
        prompt_classes = item["classes"] if args.mode == "annotated" else all_classes
        class_predictions = {}

        for class_name in prompt_classes:
            prompt = PROMPT_MAP.get(class_name, class_name.replace("-", " "))
            predictions = model.generate(
                image,
                prompt,
                max_dimension=args.max_dimension,
                compile=False,
            )[0]
            class_predictions[class_name] = predictions[: args.max_masks_per_class]
            print(f"{image_path.name} | {prompt}: {len(class_predictions[class_name])} masks")

        base_name = slugify(image_path.stem)
        overlay_path = out_dir / f"{base_name}_multiclass_overlay.jpg"
        draw_multiclass_overlay(
            image,
            image_name=image_path.name,
            annotated_classes=item["classes"],
            class_predictions=class_predictions,
        ).save(overlay_path, quality=92)

        report["images"].append(
            {
                "image": str(image_path),
                "overlay": str(overlay_path),
                "annotated_classes": item["classes"],
                "predicted_classes": summarize_predictions(class_predictions),
            }
        )
        print(f"overlay: {overlay_path}")

    summary_path = out_dir / "_multiclass_summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
