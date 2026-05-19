import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Ultralytics YOLO detection model for Skincaria.")
    parser.add_argument("--data", type=Path, default=Path("data/skincaria-yolo-detect/data.yaml"))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", default="skincaria-detect")
    parser.add_argument("--name", default="yolo11n")
    parser.add_argument("--device", default=None, help="Use 0 for GPU, cpu for CPU, or omit for auto.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(
            f"{args.data} not found. First run scripts/prepare_yolo_detect_from_coco.py."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Install it first:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install ultralytics\n"
        ) from exc

    device = args.device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    print(f"data: {args.data}")
    print(f"model: {args.model}")
    print(f"device: {device}")
    print(f"cuda_available: {torch.cuda.is_available()}")

    model = YOLO(args.model)
    results = model.train(
        task="detect",
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        device=device,
        plots=True,
        save=True,
    )
    print(results)


if __name__ == "__main__":
    main()
