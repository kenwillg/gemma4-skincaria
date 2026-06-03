import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont


DEFAULT_MODEL_ID = "Doha000/detr-skin-problems"
PALETTE = {
    "Acne": "#ef4444",
    "Blackheads": "#111827",
    "Dark-Spots": "#92400e",
    "Englarged-Pores": "#0ea5e9",
    "Enlarged-Pores": "#0ea5e9",
    "Eyebags": "#8b5cf6",
    "Skin-Redness": "#ec4899",
    "Whiteheads": "#e5e7eb",
    "Wrinkles": "#64748b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a Hugging Face DETR skin-problem detector.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--image", type=Path, required=True, help="Image file or directory of images.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/detr_skin_tests"))
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=30)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device", default=None, help="Use cuda, cpu, or omit for auto.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
    except ImportError as exc:
        raise SystemExit(
            "Missing transformers. Install it with:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install transformers huggingface_hub safetensors\n"
        ) from exc

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"model: {args.model_id}")
    print(f"device: {device}")
    print(f"threshold: {args.threshold}")

    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForObjectDetection.from_pretrained(args.model_id).to(device)
    model.eval()

    image_paths = collect_images(args.image)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found: {args.image}")

    all_results = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=device)
        processed = processor.post_process_object_detection(
            outputs,
            threshold=args.threshold,
            target_sizes=target_sizes,
        )[0]

        detections = to_detections(processed, model.config.id2label, image.size)
        detections = nms(detections, iou_threshold=args.nms_iou, max_det=args.max_det)
        annotated = draw_detections(image, detections)

        output_image = args.out_dir / f"{image_path.stem}_detr.jpg"
        output_json = args.out_dir / f"{image_path.stem}_detr.json"
        annotated.save(output_image, quality=92)

        result = {
            "image": str(image_path),
            "model": args.model_id,
            "threshold": args.threshold,
            "nms_iou": args.nms_iou,
            "detections": detections,
            "counts": class_counts(detections),
            "annotated_image": str(output_image),
        }
        output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        all_results.append(result)

        print(f"\n{image_path}")
        if detections:
            for detection in detections[:20]:
                print(
                    f"  {detection['label']}: {detection['score']:.3f} "
                    f"{detection['box']}"
                )
        else:
            print("  no detections")
        print(f"  saved: {output_image}")

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary_path}")


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(item for item in path.rglob("*") if item.suffix.lower() in exts)


def to_detections(
    processed: dict[str, Any],
    id2label: dict[Any, str],
    image_size: tuple[int, int],
) -> list[dict[str, Any]]:
    detections = []
    scores = processed["scores"].detach().cpu().tolist()
    labels = processed["labels"].detach().cpu().tolist()
    boxes = processed["boxes"].detach().cpu().tolist()
    for score, label_id, box in zip(scores, labels, boxes):
        label = id2label.get(label_id, id2label.get(str(label_id), str(label_id)))
        width, height = image_size
        x1, y1, x2, y2 = [float(value) for value in box]
        x1 = min(max(x1, 0.0), float(width))
        x2 = min(max(x2, 0.0), float(width))
        y1 = min(max(y1, 0.0), float(height))
        y2 = min(max(y2, 0.0), float(height))
        x1, y1, x2, y2 = [round(value, 1) for value in (x1, y1, x2, y2)]
        detections.append(
            {
                "label": str(label),
                "score": round(float(score), 4),
                "box": [x1, y1, x2, y2],
                "area_px": int(max(0.0, x2 - x1) * max(0.0, y2 - y1)),
            }
        )
    detections.sort(key=lambda item: item["score"], reverse=True)
    return detections


def class_counts(detections: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for detection in detections:
        label = detection["label"]
        counts[label] = counts.get(label, 0) + 1
    return counts


def nms(
    detections: list[dict[str, Any]],
    *,
    iou_threshold: float,
    max_det: int,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
        if len(kept) >= max_det:
            break
        if any(box_iou(detection["box"], existing["box"]) >= iou_threshold for existing in kept):
            continue
        kept.append(detection)
    return kept


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def draw_detections(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")
    font = ImageFont.load_default()
    line_width = max(2, round(min(image.size) / 260))

    for detection in detections[:80]:
        label = detection["label"]
        rgb = hex_to_rgb(PALETTE.get(label, "#38bdf8"))
        x1, y1, x2, y2 = detection["box"]
        draw.rectangle((x1, y1, x2, y2), outline=rgb + (235,), width=line_width)
        draw.rectangle((x1, y1, x2, y2), fill=rgb + (24,))
        text = f"{label} {detection['score']:.2f}"
        bbox = draw.textbbox((x1, y1), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        label_y = max(0, y1 - text_h - 6)
        draw.rectangle((x1, label_y, x1 + text_w + 8, label_y + text_h + 6), fill=rgb + (220,))
        text_fill = (17, 24, 39, 255) if label in {"Whiteheads"} else (255, 255, 255, 255)
        draw.text((x1 + 4, label_y + 3), text, fill=text_fill, font=font)
    return annotated


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


if __name__ == "__main__":
    main()
