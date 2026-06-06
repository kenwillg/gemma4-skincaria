import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = [
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


@dataclass(frozen=True)
class Instance:
    image_path: Path
    class_id: int
    x: float
    y: float
    w: float
    h: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an exactly instance-balanced YOLO detection train split using "
            "bbox-level copy-paste augmentation. Validation remains the original split."
        )
    )
    parser.add_argument("--src", type=Path, default=Path("data/skincaria-yolo-detect"))
    parser.add_argument("--out", type=Path, default=Path("data/skincaria-yolo-detect-instance-balanced"))
    parser.add_argument("--target", type=int, default=1000, help="Synthetic training instances per class.")
    parser.add_argument("--imgsz", type=int, default=960, help="Output square image size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pad", type=float, default=1.8, help="Context padding multiplier around bbox crop.")
    parser.add_argument("--blur-background", type=float, default=18.0)
    parser.add_argument("--jpeg-quality", type=int, default=91)
    parser.add_argument("--copy-val", action="store_true", help="Copy validation files instead of referencing source val.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    if not args.src.exists():
        raise FileNotFoundError(args.src)

    instances = load_instances(args.src)
    by_class: dict[int, list[Instance]] = defaultdict(list)
    for instance in instances:
        by_class[instance.class_id].append(instance)

    missing = [CLASS_NAMES[class_id] for class_id in range(len(CLASS_NAMES)) if not by_class[class_id]]
    if missing:
        raise RuntimeError(f"No source instances for classes: {', '.join(missing)}")

    image_out = args.out / "train" / "images"
    label_out = args.out / "train" / "labels"
    reset_dir(image_out)
    reset_dir(label_out)

    train_images = sorted((args.src / "train" / "images").glob("*"))
    train_images = [path for path in train_images if path.suffix.lower() in IMAGE_EXTS]
    if not train_images:
        raise FileNotFoundError(args.src / "train" / "images")

    written = []
    for class_id in range(len(CLASS_NAMES)):
        source_pool = by_class[class_id]
        for index in range(args.target):
            instance = random.choice(source_pool)
            background_path = random.choice(train_images)
            image, label_line = make_synthetic_image(
                instance=instance,
                background_path=background_path,
                imgsz=args.imgsz,
                pad=args.pad,
                blur_background=args.blur_background,
            )
            stem = f"{class_id:02d}_{CLASS_NAMES[class_id].lower().replace('-', '_')}_{index:05d}"
            image_path = image_out / f"{stem}.jpg"
            label_path = label_out / f"{stem}.txt"
            image.save(image_path, format="JPEG", quality=args.jpeg_quality)
            label_path.write_text(label_line + "\n", encoding="utf-8")
            written.append(label_path)

    if args.copy_val:
        copy_tree(args.src / "val", args.out / "val")

    write_yaml(src=args.src, out=args.out, copy_val=args.copy_val)
    after_counts = count_labels(written)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.src),
        "output": str(args.out),
        "target_per_class": args.target,
        "imgsz": args.imgsz,
        "method": "bbox copy-paste onto blurred training-image backgrounds; one labeled instance per synthetic image",
        "validation": "copied" if args.copy_val else "original validation referenced by absolute path",
        "source_train_instance_counts": {
            CLASS_NAMES[class_id]: len(by_class[class_id])
            for class_id in range(len(CLASS_NAMES))
        },
        "balanced_train_instance_counts": {
            CLASS_NAMES[class_id]: after_counts[class_id]
            for class_id in range(len(CLASS_NAMES))
        },
        "balanced_train_images": len(written),
        "imbalance_ratio": round(max(after_counts.values()) / max(min(after_counts.values()), 1), 4),
        "data_yaml": str(args.out / "data.yaml"),
    }
    report_path = args.out / "balance_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


def load_instances(src: Path) -> list[Instance]:
    image_dir = src / "train" / "images"
    label_dir = src / "train" / "labels"
    image_by_stem = {
        path.stem: path
        for path in sorted(image_dir.glob("*"))
        if path.suffix.lower() in IMAGE_EXTS
    }
    instances: list[Instance] = []
    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = image_by_stem.get(label_path.stem)
        if image_path is None:
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(float(parts[0]))
            x, y, w, h = [float(value) for value in parts[1:]]
            if w <= 0 or h <= 0:
                continue
            instances.append(Instance(image_path=image_path, class_id=class_id, x=x, y=y, w=w, h=h))
    return instances


def make_synthetic_image(
    *,
    instance: Instance,
    background_path: Path,
    imgsz: int,
    pad: float,
    blur_background: float,
) -> tuple[Image.Image, str]:
    with Image.open(background_path) as bg_raw:
        background = bg_raw.convert("RGB").resize((imgsz, imgsz), Image.Resampling.BICUBIC)
    background = background.filter(ImageFilter.GaussianBlur(radius=blur_background))
    background = jitter_image(background, brightness=(0.86, 1.14), contrast=(0.9, 1.12), color=(0.92, 1.1))

    with Image.open(instance.image_path) as src_raw:
        source = src_raw.convert("RGB")

    crop, obj_box = crop_instance(source, instance, pad=pad)
    crop, obj_box = augment_patch(crop, obj_box)

    scale = random.uniform(0.75, 1.35)
    patch_w = max(8, int(crop.width * scale))
    patch_h = max(8, int(crop.height * scale))
    if patch_w > int(imgsz * 0.72) or patch_h > int(imgsz * 0.72):
        shrink = min((imgsz * 0.72) / patch_w, (imgsz * 0.72) / patch_h)
        patch_w = max(8, int(patch_w * shrink))
        patch_h = max(8, int(patch_h * shrink))

    crop = crop.resize((patch_w, patch_h), Image.Resampling.BICUBIC)
    ox1, oy1, ox2, oy2, original_patch_w, original_patch_h = obj_box
    sx = patch_w / max(1, obj_box[4])
    sy = patch_h / max(1, obj_box[5])
    obj_scaled = [ox1 * sx, oy1 * sy, ox2 * sx, oy2 * sy]

    max_x = max(0, imgsz - patch_w)
    max_y = max(0, imgsz - patch_h)
    paste_x = random.randint(0, max_x)
    paste_y = random.randint(0, max_y)
    mask = feather_mask(patch_w, patch_h)
    background.paste(crop, (paste_x, paste_y), mask)

    x1 = paste_x + obj_scaled[0]
    y1 = paste_y + obj_scaled[1]
    x2 = paste_x + obj_scaled[2]
    y2 = paste_y + obj_scaled[3]
    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, imgsz, imgsz)
    xc = ((x1 + x2) / 2) / imgsz
    yc = ((y1 + y2) / 2) / imgsz
    bw = (x2 - x1) / imgsz
    bh = (y2 - y1) / imgsz
    label_line = f"{instance.class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
    return background, label_line


def crop_instance(image: Image.Image, instance: Instance, pad: float) -> tuple[Image.Image, list[float]]:
    width, height = image.size
    box_w = instance.w * width
    box_h = instance.h * height
    cx = instance.x * width
    cy = instance.y * height
    crop_w = max(box_w * pad, box_w + 8)
    crop_h = max(box_h * pad, box_h + 8)
    left = max(0, cx - crop_w / 2)
    top = max(0, cy - crop_h / 2)
    right = min(width, cx + crop_w / 2)
    bottom = min(height, cy + crop_h / 2)
    crop = image.crop((int(left), int(top), int(right), int(bottom)))
    obj_x1 = cx - box_w / 2 - left
    obj_y1 = cy - box_h / 2 - top
    obj_x2 = cx + box_w / 2 - left
    obj_y2 = cy + box_h / 2 - top
    return crop, [obj_x1, obj_y1, obj_x2, obj_y2, crop.width, crop.height]


def augment_patch(crop: Image.Image, obj_box: list[float]) -> tuple[Image.Image, list[float]]:
    crop = jitter_image(crop, brightness=(0.82, 1.18), contrast=(0.88, 1.15), color=(0.88, 1.12))
    if random.random() < 0.5:
        crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        x1, y1, x2, y2, width, height = obj_box
        obj_box = [width - x2, y1, width - x1, y2, width, height]
    if random.random() < 0.25:
        crop = crop.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.15, 0.55)))
    return crop, obj_box


def jitter_image(
    image: Image.Image,
    *,
    brightness: tuple[float, float],
    contrast: tuple[float, float],
    color: tuple[float, float],
) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(random.uniform(*brightness))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(*contrast))
    image = ImageEnhance.Color(image).enhance(random.uniform(*color))
    return image


def feather_mask(width: int, height: int) -> Image.Image:
    mask = Image.new("L", (width, height), 255)
    edge = max(2, int(min(width, height) * 0.12))
    soft = Image.new("L", (width, height), 0)
    inner = Image.new("L", (max(1, width - edge * 2), max(1, height - edge * 2)), 255)
    soft.paste(inner, (edge, edge))
    soft = soft.filter(ImageFilter.GaussianBlur(radius=max(1, edge // 2)))
    return Image.composite(mask, soft, soft)


def clamp_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1:
        x2 = min(float(width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height), y1 + 1.0)
    return x1, y1, x2, y2


def count_labels(paths: list[Path]) -> Counter:
    counts: Counter = Counter()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                counts[int(float(parts[0]))] += 1
    return counts


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def write_yaml(*, src: Path, out: Path, copy_val: bool) -> None:
    val_path = "val/images" if copy_val else str((src / "val" / "images").resolve()).replace("\\", "/")
    lines = [
        f"path: {str(out.resolve()).replace('\\', '/')}",
        "train: train/images",
        f"val: {val_path}",
        "",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES))
    (out / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
