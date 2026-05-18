import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from transformers import AutoModelForCausalLM


DEFAULT_MODEL = "tiiuae/Falcon-Perception"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Falcon Perception and save mask overlays.")
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument("--prompt", default="acne", help="Open-vocabulary prompt, e.g. acne or skin redness.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/falcon_tests"))
    parser.add_argument("--max-dimension", type=int, default=512)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-masks", type=int, default=50)
    return parser.parse_args()


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "prompt"


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


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    image = Image.open(args.image).convert("RGB")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
    ).to(device).eval()

    predictions = model.generate(
        image,
        args.prompt,
        max_dimension=args.max_dimension,
        compile=False,
    )[0]
    predictions = predictions[: args.max_masks]

    base_name = f"{args.image.stem}_{slugify(args.prompt)}"
    overlay_path = args.out_dir / f"{base_name}_overlay.jpg"
    json_path = args.out_dir / f"{base_name}_summary.json"

    overlay = draw_predictions(image, predictions)
    overlay.save(overlay_path, quality=92)

    payload = {
        "image": str(args.image),
        "prompt": args.prompt,
        "device": device,
        "max_dimension": args.max_dimension,
        "prediction_count": len(predictions),
        "predictions": prediction_summary(predictions),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"device: {device}")
    print(f"prediction_count: {len(predictions)}")
    print(f"overlay: {overlay_path}")
    print(f"summary: {json_path}")


if __name__ == "__main__":
    main()
