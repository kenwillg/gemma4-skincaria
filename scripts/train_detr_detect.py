import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DEFAULT_MODEL_ID = "facebook/detr-resnet-50"
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

# Existing YOLO class ids:
# 0 Acne, 1 Blackheads, 2 Dark-Spots, 3 Dry-Skin, 4 Enlarged-Pores,
# 5 Eyebags, 6 Oily-Skin, 7 Skin-Redness, 8 Whiteheads, 9 Wrinkles.
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
    parser = argparse.ArgumentParser(
        description="Fine-tune DETR on the Skincaria YOLO detection dataset."
    )
    parser.add_argument("--data", type=Path, default=Path("data/skincaria-yolo-detect"))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/detr/skincaria-detr-resnet50"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=25, help="Print progress every N batches.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="Use cuda, cpu, or omit for auto.")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
    except ImportError as exc:
        raise SystemExit(
            "Missing transformers. Install it with:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install transformers timm safetensors\n"
        ) from exc

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"base model: {args.model_id}")
    print(f"dataset: {args.data}")
    print(f"output: {args.out_dir}")
    print(f"device: {device}")
    print("labels:", ", ".join(DETR_LABELS))
    print("skipping YOLO labels: Dry-Skin, Oily-Skin")

    id2label = {idx: label for idx, label in enumerate(DETR_LABELS)}
    label2id = {label: idx for idx, label in id2label.items()}

    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForObjectDetection.from_pretrained(
        args.model_id,
        num_labels=len(DETR_LABELS),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    train_dataset = YoloDetectionDataset(
        args.data,
        split="train",
        max_images=args.max_train_images,
    )
    val_dataset = YoloDetectionDataset(
        args.data,
        split="val",
        max_images=args.max_val_images,
    )
    print(f"train images: {len(train_dataset)}")
    print(f"val images: {len(val_dataset)}")
    print("train target counts:", json.dumps(train_dataset.class_counts(), indent=2))
    print("val target counts:", json.dumps(val_dataset.class_counts(), indent=2))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=DetrCollator(processor),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=DetrCollator(processor),
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p for n, p in model.named_parameters() if p.requires_grad and "backbone" not in n
                ],
                "lr": args.lr,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if p.requires_grad and "backbone" in n
                ],
                "lr": args.backbone_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device == "cuda")
    best_val_loss = math.inf
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_accum=max(1, args.grad_accum),
            amp=args.amp and device == "cuda",
            epoch=epoch,
            log_every=max(1, args.log_every),
        )
        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            device=device,
            epoch=epoch,
            log_every=max(1, args.log_every),
        )
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )

        save_checkpoint(args.out_dir / "last", model, processor, row, history)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(args.out_dir / "best", model, processor, row, history)
            print(f"  saved new best: {args.out_dir / 'best'}")

    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"done. best val_loss={best_val_loss:.4f}")
    print(f"test with: .\\.venv\\Scripts\\python.exe scripts\\test_hf_detr_skin.py --model-id {args.out_dir / 'best'} --image data\\skincaria-yolo-detect\\val\\images --threshold 0.25")


class YoloDetectionDataset(Dataset):
    def __init__(self, data_dir: Path, *, split: str, max_images: int | None = None) -> None:
        self.data_dir = data_dir
        self.split = split
        self.image_dir = data_dir / split / "images"
        self.label_dir = data_dir / split / "labels"
        self.image_paths = collect_images(self.image_dir)
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Image.Image, dict[str, Any]]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        annotations = []
        label_path = self.label_dir / f"{image_path.stem}.txt"

        if label_path.exists():
            for object_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                yolo_class = int(float(parts[0]))
                if yolo_class not in YOLO_TO_DETR:
                    continue
                x_center, y_center, box_w, box_h = [float(value) for value in parts[1:5]]
                x = (x_center - box_w / 2.0) * width
                y = (y_center - box_h / 2.0) * height
                w = box_w * width
                h = box_h * height
                x = min(max(x, 0.0), float(width))
                y = min(max(y, 0.0), float(height))
                w = min(max(w, 1.0), float(width) - x)
                h = min(max(h, 1.0), float(height) - y)
                if w <= 0 or h <= 0:
                    continue
                annotations.append(
                    {
                        "id": object_index,
                        "image_id": index,
                        "category_id": YOLO_TO_DETR[yolo_class],
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )

        target = {"image_id": index, "annotations": annotations}
        return image, target

    def class_counts(self) -> dict[str, int]:
        counts = {label: 0 for label in DETR_LABELS}
        for idx in range(len(self)):
            _, target = self[idx]
            for annotation in target["annotations"]:
                counts[DETR_LABELS[annotation["category_id"]]] += 1
        return counts


class DetrCollator:
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, batch: list[tuple[Image.Image, dict[str, Any]]]) -> dict[str, Any]:
        images, annotations = zip(*batch)
        return self.processor(
            images=list(images),
            annotations=list(annotations),
            return_tensors="pt",
        )


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str,
    grad_accum: int,
    amp: bool,
    epoch: int,
    log_every: int,
) -> float:
    model.train()
    total_loss = 0.0
    seen = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", enabled=amp):
            outputs = model(**batch)
            loss = outputs.loss / grad_accum

        scaler.scale(loss).backward()
        if step % grad_accum == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = batch["pixel_values"].shape[0]
        total_loss += float(loss.detach().cpu()) * grad_accum * batch_size
        seen += batch_size
        if step == 1 or step % log_every == 0 or step == len(loader):
            running = total_loss / max(seen, 1)
            print(
                f"epoch {epoch:03d} train batch {step:04d}/{len(loader)} "
                f"loss={running:.4f}",
                flush=True,
            )

    return total_loss / max(seen, 1)


@torch.no_grad()
def evaluate_loss(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    epoch: int,
    log_every: int,
) -> float:
    model.eval()
    total_loss = 0.0
    seen = 0
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        outputs = model(**batch)
        batch_size = batch["pixel_values"].shape[0]
        total_loss += float(outputs.loss.detach().cpu()) * batch_size
        seen += batch_size
        if step == 1 or step % log_every == 0 or step == len(loader):
            running = total_loss / max(seen, 1)
            print(
                f"epoch {epoch:03d} val batch {step:04d}/{len(loader)} "
                f"loss={running:.4f}",
                flush=True,
            )
    return total_loss / max(seen, 1)


def move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if key == "labels":
            moved[key] = [
                {label_key: label_value.to(device) for label_key, label_value in item.items()}
                for item in value
            ]
        elif hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    processor: Any,
    row: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    (output_dir / "trainer_state.json").write_text(
        json.dumps({"latest": row, "history": history}, indent=2),
        encoding="utf-8",
    )


def collect_images(path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(item for item in path.rglob("*") if item.suffix.lower() in exts)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
