import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from train_efficientnetv2_multilabel import (
    ALL_CLASS_NAMES,
    YoloMultiLabelDataset,
    classification_metrics,
    val_transform,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune per-class thresholds for a trained EfficientNetV2 multi-label classifier."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--min-threshold", type=float, default=0.05)
    parser.add_argument("--max-threshold", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    try:
        import timm
    except ImportError as exc:
        raise SystemExit(
            "Missing timm. Install it with:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install timm\n"
        ) from exc

    class_ids = config["class_ids"]
    class_names = config.get("class_names") or [ALL_CLASS_NAMES[class_id] for class_id in class_ids]
    data_dir = args.data or Path(config["data"])
    imgsz = args.imgsz or int(config["imgsz"])

    dataset = YoloMultiLabelDataset(
        data_dir=data_dir,
        split=args.split,
        class_ids=class_ids,
        transform=val_transform(imgsz),
        crop=config.get("crop", "none"),
        face_margin=float(config.get("face_margin", 0.25)),
        skin_mask=config.get("skin_mask", "none"),
        mask_background=config.get("mask_background", "mean"),
        max_images=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device == "cuda",
    )

    model = timm.create_model(
        config.get("model", "tf_efficientnetv2_b0"),
        pretrained=False,
        num_classes=len(class_ids),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    probs, targets = collect_predictions(model=model, loader=loader, device=device)
    tuned = tune_thresholds(
        probs=probs,
        targets=targets,
        class_names=class_names,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        step=args.step,
    )
    thresholds = torch.tensor([row["threshold"] for row in tuned["per_class"]], dtype=torch.float32)
    tuned_preds = (probs >= thresholds).float()
    tuned_metrics = classification_metrics(tuned_preds, targets, class_names)
    default_metrics = classification_metrics((probs >= 0.5).float(), targets, class_names)

    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "data": str(data_dir),
        "split": args.split,
        "class_names": class_names,
        "thresholds": {row["class"]: row["threshold"] for row in tuned["per_class"]},
        "default_0_5": {
            "macro_f1": default_metrics["macro_f1"],
            "micro_f1": default_metrics["micro_f1"],
            "per_class": default_metrics["per_class"],
        },
        "tuned": {
            "macro_f1": tuned_metrics["macro_f1"],
            "micro_f1": tuned_metrics["micro_f1"],
            "per_class": tuned_metrics["per_class"],
        },
        "threshold_search": tuned["per_class"],
    }

    out_path = args.out or args.checkpoint.parent / "thresholds.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"checkpoint: {args.checkpoint}")
    print(f"default macro_f1={default_metrics['macro_f1']:.4f} micro_f1={default_metrics['micro_f1']:.4f}")
    print(f"tuned   macro_f1={tuned_metrics['macro_f1']:.4f} micro_f1={tuned_metrics['micro_f1']:.4f}")
    print("thresholds:")
    for row in tuned["per_class"]:
        print(f"  {row['class']}: {row['threshold']:.2f} f1={row['f1']:.4f}")
    print(f"saved: {out_path}")


@torch.no_grad()
def collect_predictions(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs_all = []
    targets_all = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs_all.append(torch.sigmoid(logits).cpu())
        targets_all.append(targets.cpu())
    return torch.cat(probs_all), torch.cat(targets_all)


def tune_thresholds(
    *,
    probs: torch.Tensor,
    targets: torch.Tensor,
    class_names: list[str],
    min_threshold: float,
    max_threshold: float,
    step: float,
) -> dict[str, Any]:
    thresholds = make_threshold_grid(min_threshold, max_threshold, step)
    per_class = []
    for idx, class_name in enumerate(class_names):
        best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
        class_probs = probs[:, idx]
        class_targets = targets[:, idx]
        for threshold in thresholds:
            pred = (class_probs >= threshold).float()
            metrics = binary_metrics(pred, class_targets)
            if metrics["f1"] > best["f1"]:
                best = {"threshold": threshold, **metrics}
        per_class.append({"class": class_name, **best})
    return {"per_class": per_class}


def make_threshold_grid(min_threshold: float, max_threshold: float, step: float) -> list[float]:
    values = []
    current = min_threshold
    while current <= max_threshold + 1e-9:
        values.append(round(current, 4))
        current += step
    return values


def binary_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    eps = 1e-9
    tp = float(((pred == 1) & (target == 1)).sum())
    fp = float(((pred == 1) & (target == 0)).sum())
    fn = float(((pred == 0) & (target == 1)).sum())
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    f1 = 2 * precision * recall / max(precision + recall, eps)
    return {"precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    main()
