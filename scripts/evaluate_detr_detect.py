import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image


DETR_LABELS = [
    "Acne",
    "Blackheads",
    "Dark-Spots",
    "Enlarged-Pores",
    "Eyebags",
    "Skin-Redness",
    "Whiteheads",
    "Wrinkles",
]

YOLO_TO_DETR = {
    0: 0,
    1: 1,
    2: 2,
    4: 3,
    5: 4,
    7: 5,
    8: 6,
    9: 7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a DETR checkpoint on Skincaria YOLO labels.")
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/skincaria-yolo-detect"))
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--pr-conf", type=float, default=0.25)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
    except ImportError as exc:
        raise SystemExit("Missing transformers. Install requirements first.") from exc

    image_dir = args.data / args.split / "images"
    label_dir = args.data / args.split / "labels"
    image_paths = collect_images(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    print(f"model: {args.model_id}")
    print(f"data: {args.data} ({args.split})")
    print(f"device: {device}")
    print(f"images: {len(image_paths)}")

    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForObjectDetection.from_pretrained(args.model_id).to(device)
    model.eval()

    all_ground_truths: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []

    for image_index, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        ground_truths = read_yolo_ground_truths(label_dir / f"{image_path.stem}.txt", width, height)
        for gt in ground_truths:
            gt["image_id"] = image_index
        all_ground_truths.extend(ground_truths)

        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=device)
        processed = processor.post_process_object_detection(
            outputs,
            threshold=args.score_threshold,
            target_sizes=target_sizes,
        )[0]
        predictions = to_predictions(processed, image_index, max_det=args.max_det)
        all_predictions.extend(predictions)

        done = image_index + 1
        if done == 1 or done % args.log_every == 0 or done == len(image_paths):
            print(f"evaluated {done}/{len(image_paths)}", flush=True)

    metrics = compute_metrics(
        ground_truths=all_ground_truths,
        predictions=all_predictions,
        pr_conf=args.pr_conf,
    )
    metrics["model"] = str(args.model_id)
    metrics["data"] = str(args.data)
    metrics["split"] = args.split
    metrics["num_images"] = len(image_paths)
    metrics["num_ground_truths"] = len(all_ground_truths)
    metrics["num_predictions"] = len(all_predictions)
    metrics["score_threshold"] = args.score_threshold
    metrics["pr_conf"] = args.pr_conf

    print(json.dumps(metrics, indent=2))

    out_json = args.out_json or args.model_id / f"{args.split}_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved: {out_json}")


def collect_images(path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(item for item in path.rglob("*") if item.suffix.lower() in exts)


def read_yolo_ground_truths(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    if not label_path.exists():
        return []
    ground_truths = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        yolo_class = int(float(parts[0]))
        if yolo_class not in YOLO_TO_DETR:
            continue
        x_center, y_center, box_w, box_h = [float(value) for value in parts[1:5]]
        x1 = (x_center - box_w / 2.0) * width
        y1 = (y_center - box_h / 2.0) * height
        x2 = (x_center + box_w / 2.0) * width
        y2 = (y_center + box_h / 2.0) * height
        ground_truths.append(
            {
                "class_id": YOLO_TO_DETR[yolo_class],
                "box": [
                    min(max(x1, 0.0), float(width)),
                    min(max(y1, 0.0), float(height)),
                    min(max(x2, 0.0), float(width)),
                    min(max(y2, 0.0), float(height)),
                ],
            }
        )
    return ground_truths


def to_predictions(
    processed: dict[str, torch.Tensor],
    image_id: int,
    *,
    max_det: int,
) -> list[dict[str, Any]]:
    scores = processed["scores"].detach().cpu().tolist()
    labels = processed["labels"].detach().cpu().tolist()
    boxes = processed["boxes"].detach().cpu().tolist()
    predictions = []
    for score, label, box in zip(scores, labels, boxes):
        label_id = int(label)
        if label_id < 0 or label_id >= len(DETR_LABELS):
            continue
        predictions.append(
            {
                "image_id": image_id,
                "class_id": label_id,
                "score": float(score),
                "box": [float(value) for value in box],
            }
        )
    predictions.sort(key=lambda item: item["score"], reverse=True)
    return predictions[:max_det]


def compute_metrics(
    *,
    ground_truths: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    pr_conf: float,
) -> dict[str, Any]:
    iou_thresholds = [round(0.5 + idx * 0.05, 2) for idx in range(10)]
    per_class = []
    for class_id, label in enumerate(DETR_LABELS):
        class_gt = [item for item in ground_truths if item["class_id"] == class_id]
        class_pred = [item for item in predictions if item["class_id"] == class_id]
        ap_by_iou = {
            f"{threshold:.2f}": average_precision(class_gt, class_pred, threshold)
            for threshold in iou_thresholds
        }
        per_class.append(
            {
                "class": label,
                "gt": len(class_gt),
                "pred": len(class_pred),
                "ap50": ap_by_iou["0.50"],
                "map50_95": sum(ap_by_iou.values()) / len(ap_by_iou),
                "ap_by_iou": ap_by_iou,
            }
        )

    aps50 = [item["ap50"] for item in per_class if item["gt"] > 0]
    aps_all = [item["map50_95"] for item in per_class if item["gt"] > 0]
    precision, recall = precision_recall_at_conf(ground_truths, predictions, pr_conf, iou_threshold=0.5)
    return {
        "precision_at_conf": precision,
        "recall_at_conf": recall,
        "mAP50": sum(aps50) / len(aps50) if aps50 else 0.0,
        "mAP50-95": sum(aps_all) / len(aps_all) if aps_all else 0.0,
        "per_class": per_class,
    }


def average_precision(
    ground_truths: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> float:
    if not ground_truths:
        return 0.0
    predictions = sorted(predictions, key=lambda item: item["score"], reverse=True)
    matched: set[tuple[int, int]] = set()
    tp = []
    fp = []
    gt_by_image: dict[int, list[dict[str, Any]]] = {}
    for gt in ground_truths:
        gt_by_image.setdefault(gt["image_id"], []).append(gt)

    for pred in predictions:
        candidates = gt_by_image.get(pred["image_id"], [])
        best_iou = 0.0
        best_gt_index = -1
        for gt_index, gt in enumerate(candidates):
            key = (pred["image_id"], gt_index)
            if key in matched:
                continue
            overlap = box_iou(pred["box"], gt["box"])
            if overlap > best_iou:
                best_iou = overlap
                best_gt_index = gt_index
        if best_iou >= iou_threshold and best_gt_index >= 0:
            matched.add((pred["image_id"], best_gt_index))
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)

    if not tp:
        return 0.0
    cum_tp = cumulative_sum(tp)
    cum_fp = cumulative_sum(fp)
    recalls = [value / len(ground_truths) for value in cum_tp]
    precisions = [
        cum_tp[idx] / max(cum_tp[idx] + cum_fp[idx], 1e-12)
        for idx in range(len(cum_tp))
    ]
    return interpolated_ap(recalls, precisions)


def precision_recall_at_conf(
    ground_truths: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    conf: float,
    iou_threshold: float,
) -> tuple[float, float]:
    matched: set[tuple[int, int]] = set()
    gt_by_image_class: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for gt in ground_truths:
        gt_by_image_class.setdefault((gt["image_id"], gt["class_id"]), []).append(gt)

    filtered = [pred for pred in predictions if pred["score"] >= conf]
    filtered.sort(key=lambda item: item["score"], reverse=True)
    tp = 0
    fp = 0
    for pred in filtered:
        key_prefix = (pred["image_id"], pred["class_id"])
        candidates = gt_by_image_class.get(key_prefix, [])
        best_iou = 0.0
        best_gt_index = -1
        for gt_index, gt in enumerate(candidates):
            key = (pred["image_id"], pred["class_id"], gt_index)
            if key in matched:
                continue
            overlap = box_iou(pred["box"], gt["box"])
            if overlap > best_iou:
                best_iou = overlap
                best_gt_index = gt_index
        if best_iou >= iou_threshold and best_gt_index >= 0:
            matched.add((pred["image_id"], pred["class_id"], best_gt_index))
            tp += 1
        else:
            fp += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(len(ground_truths), 1)
    return precision, recall


def cumulative_sum(values: list[float]) -> list[float]:
    total = 0.0
    result = []
    for value in values:
        total += value
        result.append(total)
    return result


def interpolated_ap(recalls: list[float], precisions: list[float]) -> float:
    points = [idx / 100 for idx in range(101)]
    ap = 0.0
    for point in points:
        candidates = [precision for recall, precision in zip(recalls, precisions) if recall >= point]
        ap += max(candidates) if candidates else 0.0
    return ap / len(points)


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


if __name__ == "__main__":
    main()
