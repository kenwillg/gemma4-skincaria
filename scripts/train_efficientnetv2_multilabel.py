import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ALL_CLASS_NAMES = {
    0: "Acne",
    1: "Blackheads",
    2: "Dark-Spots",
    3: "Dry-Skin",
    4: "Enlarged-Pores",
    5: "Eyebags",
    6: "Oily-Skin",
    7: "Skin-Redness",
    8: "Whiteheads",
    9: "Wrinkles",
}

CLASS_PRESETS = {
    "texture": [3, 4, 6, 7, 9],
    "lesion": [0, 1, 2, 5, 8],
    "all": list(ALL_CLASS_NAMES),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EfficientNetV2-B0 as a multi-label skin condition classifier."
    )
    parser.add_argument("--data", type=Path, default=Path("data/skincaria-yolo-detect"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/classify/efficientnetv2-b0-texture"))
    parser.add_argument("--model", default="tf_efficientnetv2_b0")
    parser.add_argument("--classes", choices=sorted(CLASS_PRESETS), default="texture")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--crop", choices=["none", "face"], default="none")
    parser.add_argument("--face-margin", type=float, default=0.25)
    parser.add_argument("--skin-mask", choices=["none", "hsl"], default="none")
    parser.add_argument("--mask-background", choices=["mean", "black"], default="mean")
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    class_ids = CLASS_PRESETS[args.classes]
    class_names = [ALL_CLASS_NAMES[class_id] for class_id in class_ids]

    try:
        import timm
    except ImportError as exc:
        raise SystemExit(
            "Missing timm. Install it with:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install timm\n"
        ) from exc

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"model: {args.model}")
    print(f"data: {args.data}")
    print(f"output: {args.out_dir}")
    print(f"device: {device}")
    print(f"classes: {', '.join(class_names)}")
    print(f"crop: {args.crop}")
    print(f"skin_mask: {args.skin_mask}")

    train_dataset = YoloMultiLabelDataset(
        data_dir=args.data,
        split="train",
        class_ids=class_ids,
        transform=train_transform(args.imgsz),
        crop=args.crop,
        face_margin=args.face_margin,
        skin_mask=args.skin_mask,
        mask_background=args.mask_background,
        max_images=args.max_train_images,
    )
    val_dataset = YoloMultiLabelDataset(
        data_dir=args.data,
        split="val",
        class_ids=class_ids,
        transform=val_transform(args.imgsz),
        crop=args.crop,
        face_margin=args.face_margin,
        skin_mask=args.skin_mask,
        mask_background=args.mask_background,
        max_images=args.max_val_images,
    )
    print(f"train images: {len(train_dataset)}")
    print(f"val images: {len(val_dataset)}")
    print("train positives:", json.dumps(train_dataset.positive_counts(), indent=2))
    print("val positives:", json.dumps(val_dataset.positive_counts(), indent=2))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device == "cuda",
    )

    model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=len(class_ids),
    ).to(device)

    pos_weight = train_dataset.pos_weight().to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device == "cuda")

    best_macro_f1 = -1.0
    stale_epochs = 0
    history = []

    config = {
        "model": args.model,
        "classes": args.classes,
        "class_ids": class_ids,
        "class_names": class_names,
        "imgsz": args.imgsz,
        "threshold": args.threshold,
        "data": str(args.data),
        "crop": args.crop,
        "face_margin": args.face_margin,
        "skin_mask": args.skin_mask,
        "mask_background": args.mask_background,
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp=args.amp and device == "cuda",
        )
        val_loss, metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=args.threshold,
            class_names=class_names,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **metrics,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} micro_f1={metrics['micro_f1']:.4f}",
            flush=True,
        )

        save_checkpoint(args.out_dir / "last.pt", model, config, row, history)
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            stale_epochs = 0
            save_checkpoint(args.out_dir / "best.pt", model, config, row, history)
            print(f"  saved new best: {args.out_dir / 'best.pt'}")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping: no macro_f1 improvement for {args.patience} epochs")
                break

    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"done. best macro_f1={best_macro_f1:.4f}")


class YoloMultiLabelDataset(Dataset):
    def __init__(
        self,
        *,
        data_dir: Path,
        split: str,
        class_ids: list[int],
        transform: transforms.Compose,
        crop: str,
        face_margin: float,
        skin_mask: str,
        mask_background: str,
        max_images: int | None,
    ) -> None:
        self.data_dir = data_dir
        self.split = split
        self.class_ids = class_ids
        self.class_id_to_index = {class_id: idx for idx, class_id in enumerate(class_ids)}
        self.transform = transform
        self.crop = crop
        self.face_margin = face_margin
        self.face_detector = FaceCropper() if crop == "face" else None
        self.skin_mask = skin_mask
        self.skin_masker = HslSkinMasker(background=mask_background) if skin_mask == "hsl" else None
        self.image_dir = data_dir / split / "images"
        self.label_dir = data_dir / split / "labels"
        self.image_paths = collect_images(self.image_dir)
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]
        if not self.image_paths:
            raise FileNotFoundError(f"No images found: {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.face_detector is not None:
            image = self.face_detector.crop(image, margin=self.face_margin)
        if self.skin_masker is not None:
            image = self.skin_masker.apply(image)
        target = self.read_target(self.label_dir / f"{image_path.stem}.txt")
        return self.transform(image), target

    def read_target(self, label_path: Path) -> torch.Tensor:
        target = torch.zeros(len(self.class_ids), dtype=torch.float32)
        if not label_path.exists():
            return target
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(float(parts[0]))
            if class_id in self.class_id_to_index:
                target[self.class_id_to_index[class_id]] = 1.0
        return target

    def positive_counts(self) -> dict[str, int]:
        counts = {ALL_CLASS_NAMES[class_id]: 0 for class_id in self.class_ids}
        for image_path in self.image_paths:
            target = self.read_target(self.label_dir / f"{image_path.stem}.txt")
            for idx, value in enumerate(target.tolist()):
                if value > 0:
                    counts[ALL_CLASS_NAMES[self.class_ids[idx]]] += 1
        return counts

    def pos_weight(self) -> torch.Tensor:
        counts = torch.zeros(len(self.class_ids), dtype=torch.float32)
        for image_path in self.image_paths:
            target = self.read_target(self.label_dir / f"{image_path.stem}.txt")
            counts += target
        negatives = len(self.image_paths) - counts
        return torch.clamp(negatives / torch.clamp(counts, min=1.0), min=1.0, max=20.0)


def train_transform(imgsz: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def val_transform(imgsz: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class FaceCropper:
    def __init__(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise SystemExit(
                "Face crop requires OpenCV. Install it with:\n"
                "  .\\.venv\\Scripts\\python.exe -m pip install opencv-python\n"
            ) from exc

        self.cv2 = cv2
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(str(cascade_path))
        if self.detector.empty():
            raise FileNotFoundError(f"Could not load Haar cascade: {cascade_path}")

    def crop(self, image: Image.Image, *, margin: float) -> Image.Image:
        np_image = np_from_pil(image)
        gray = self.cv2.cvtColor(np_image, self.cv2.COLOR_RGB2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(48, 48),
        )
        if len(faces) == 0:
            return image

        x, y, width, height = max(faces, key=lambda item: item[2] * item[3])
        pad_x = int(width * margin)
        pad_y = int(height * margin)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image.width, x + width + pad_x)
        y2 = min(image.height, y + height + pad_y)
        return image.crop((x1, y1, x2, y2))


class HslSkinMasker:
    def __init__(self, *, background: str) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise SystemExit(
                "HSL skin mask requires OpenCV. Install it with:\n"
                "  .\\.venv\\Scripts\\python.exe -m pip install opencv-python\n"
            ) from exc

        self.cv2 = cv2
        self.background = background

    def apply(self, image: Image.Image) -> Image.Image:
        import numpy as np

        rgb = np.asarray(image).copy()
        hls = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2HLS)
        h = hls[:, :, 0].astype(np.float32)
        l = hls[:, :, 1].astype(np.float32)
        s = hls[:, :, 2].astype(np.float32)
        ls_ratio = l / np.maximum(s, 1.0)

        mask = (
            (s >= 50.0)
            & (ls_ratio > 0.5)
            & (ls_ratio < 3.0)
            & ((h <= 14.0) | (h >= 165.0))
        ).astype(np.uint8) * 255
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_OPEN, kernel)
        mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_CLOSE, kernel)
        alpha = self.cv2.GaussianBlur(mask, (9, 9), 0).astype(np.float32) / 255.0
        alpha = alpha[:, :, None]

        if self.background == "black":
            background = np.zeros_like(rgb, dtype=np.float32)
        else:
            background = np.empty_like(rgb, dtype=np.float32)
            background[:, :, 0] = 123.675
            background[:, :, 1] = 116.28
            background[:, :, 2] = 103.53

        masked = rgb.astype(np.float32) * alpha + background * (1.0 - alpha)
        return Image.fromarray(np.clip(masked, 0, 255).astype(np.uint8), mode="RGB")


def np_from_pil(image: Image.Image):
    import numpy as np

    return np.asarray(image)


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str,
    amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    seen = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu()) * images.shape[0]
        seen += images.shape[0]
    return total_loss / max(seen, 1)


@torch.no_grad()
def evaluate(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: str,
    threshold: float,
    class_names: list[str],
) -> tuple[float, dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    seen = 0
    logits_all = []
    targets_all = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += float(loss.detach().cpu()) * images.shape[0]
        seen += images.shape[0]
        logits_all.append(logits.detach().cpu())
        targets_all.append(targets.detach().cpu())

    logits_tensor = torch.cat(logits_all)
    targets_tensor = torch.cat(targets_all)
    probs = torch.sigmoid(logits_tensor)
    preds = (probs >= threshold).float()
    metrics = classification_metrics(preds, targets_tensor, class_names)
    return total_loss / max(seen, 1), metrics


def classification_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    class_names: list[str],
) -> dict[str, Any]:
    eps = 1e-9
    per_class = []
    macro_f1 = 0.0
    total_tp = total_fp = total_fn = 0.0
    for idx, name in enumerate(class_names):
        pred = preds[:, idx]
        target = targets[:, idx]
        tp = float(((pred == 1) & (target == 1)).sum())
        fp = float(((pred == 1) & (target == 0)).sum())
        fn = float(((pred == 0) & (target == 1)).sum())
        precision = tp / max(tp + fp, eps)
        recall = tp / max(tp + fn, eps)
        f1 = 2 * precision * recall / max(precision + recall, eps)
        macro_f1 += f1
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_class.append(
            {
                "class": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(target.sum().item()),
                "predicted_positive": int(pred.sum().item()),
            }
        )

    macro_f1 /= max(len(class_names), 1)
    micro_precision = total_tp / max(total_tp + total_fp, eps)
    micro_recall = total_tp / max(total_tp + total_fn, eps)
    micro_f1 = 2 * micro_precision * micro_recall / max(micro_precision + micro_recall, eps)
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "per_class": per_class,
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: dict[str, Any],
    row: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "latest": row,
            "history": history,
        },
        path,
    )


def collect_images(path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(item for item in path.rglob("*") if item.suffix.lower() in exts)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    if math.isnan(seed):
        raise ValueError("seed must be numeric")


if __name__ == "__main__":
    main()
