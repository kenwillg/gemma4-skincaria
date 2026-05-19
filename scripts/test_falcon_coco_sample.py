import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from transformers import AutoModelForCausalLM


DEFAULT_MODEL = "tiiuae/Falcon-Perception"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Falcon Perception on a sample from a Roboflow COCO export."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/skincaria-dataset.coco"),
        help="Roboflow COCO export folder.",
    )
    parser.add_argument("--split", default="train", help="COCO split folder, usually train/valid/test.")
    parser.add_argument("--prompt", default="acne", help="Falcon prompt to test.")
    parser.add_argument("--limit", type=int, default=5, help="Number of images to test.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-dimension", type=int, default=512)
    parser.add_argument("--max-masks", type=int, default=50)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/falcon_coco_tests"))
    return parser.parse_args()


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "prompt"


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


def draw_predictions(image: Image.Image, predictions: list[dict]) -> Image.Image:
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    colors = [
        (239, 68, 68),
        (14, 165, 233),
        (34, 197, 94),
        (245, 158, 11),
        (168, 85, 247),
        (236, 72, 153),
    ]

    for index, pred in enumerate(predictions):
        rle = pred.get("mask_rle")
        if not rle:
            continue

        mask = decode_mask(rle)
        color = colors[index % len(colors)]
        overlay = mask_to_image(mask, color, alpha=96)
        if overlay.size != canvas.size:
            overlay = overlay.resize(canvas.size, Image.Resampling.NEAREST)
        canvas.alpha_composite(overlay)

        xy = pred.get("xy") or {}
        x = int(float(xy.get("x", 0)) * canvas.width)
        y = int(float(xy.get("y", 0)) * canvas.height)
        radius = 5
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, 255))
        draw.text((x + 8, y - 8), str(index + 1), fill=(255, 255, 255, 255))

    return canvas.convert("RGB")


def prediction_summary(predictions: list[dict]) -> list[dict]:
    summary = []
    for index, pred in enumerate(predictions):
        xy = pred.get("xy") or {}
        hw = pred.get("hw") or {}
        mask_area = 0
        if pred.get("mask_rle"):
            mask_area = int(decode_mask(pred["mask_rle"]).sum())
        summary.append(
            {
                "index": index + 1,
                "x": xy.get("x"),
                "y": xy.get("y"),
                "h": hw.get("h"),
                "w": hw.get("w"),
                "mask_area_pixels": mask_area,
            }
        )
    return summary


def load_coco_sample(annotation_path: Path, split_dir: Path, limit: int, seed: int) -> list[Path]:
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = coco.get("images") or []
    image_paths = []
    for item in images:
        file_name = item.get("file_name")
        if not file_name:
            continue
        path = split_dir / file_name
        if path.exists() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)

    if not image_paths:
        image_paths = [
            path
            for path in sorted(split_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

    random.Random(seed).shuffle(image_paths)
    return image_paths[:limit]


def main() -> None:
    args = parse_args()
    split_dir = args.dataset_dir / args.split
    annotation_path = find_annotation_path(split_dir)
    image_paths = load_coco_sample(annotation_path, split_dir, args.limit, args.seed)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {split_dir}")

    out_dir = args.out_dir / args.split / slugify(args.prompt)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
    ).to(device).eval()

    run_summary = {
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "annotation_path": str(annotation_path),
        "prompt": args.prompt,
        "device": device,
        "max_dimension": args.max_dimension,
        "images": [],
    }

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        predictions = model.generate(
            image,
            args.prompt,
            max_dimension=args.max_dimension,
            compile=False,
        )[0]
        predictions = predictions[: args.max_masks]

        base_name = f"{image_path.stem}_{slugify(args.prompt)}"
        overlay_path = out_dir / f"{base_name}_overlay.jpg"
        draw_predictions(image, predictions).save(overlay_path, quality=92)

        run_summary["images"].append(
            {
                "image": str(image_path),
                "overlay": str(overlay_path),
                "prediction_count": len(predictions),
                "predictions": prediction_summary(predictions),
            }
        )
        print(f"{image_path.name}: {len(predictions)} predictions -> {overlay_path}")

    summary_path = out_dir / "_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
