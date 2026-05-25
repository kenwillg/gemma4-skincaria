import argparse
import shutil
from pathlib import Path

import numpy as np


CLASS_NAMES = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a preprocessed copy of a YOLO detection dataset."
    )
    parser.add_argument("--src", type=Path, default=Path("data/skincaria-yolo-detect"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["clahe", "grayworld", "grayworld-clahe"],
        default="clahe",
        help="Preprocessing mode to apply to every image.",
    )
    parser.add_argument("--clahe-clip", type=float, default=2.0)
    parser.add_argument("--clahe-grid", type=int, default=8)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Debug limit per split. Omit for the full dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.src.exists():
        raise FileNotFoundError(args.src)

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "Missing OpenCV. Install it with:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install opencv-python\n"
        ) from exc

    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        process_split(
            cv2=cv2,
            src=args.src,
            out=args.out,
            split=split,
            mode=args.mode,
            clahe_clip=args.clahe_clip,
            clahe_grid=args.clahe_grid,
            max_images=args.max_images,
        )

    write_data_yaml(args.out)
    print(f"wrote: {args.out}")
    print(f"yaml:  {args.out / 'data.yaml'}")


def process_split(
    *,
    cv2,
    src: Path,
    out: Path,
    split: str,
    mode: str,
    clahe_clip: float,
    clahe_grid: int,
    max_images: int | None,
) -> None:
    src_image_dir = src / split / "images"
    src_label_dir = src / split / "labels"
    out_image_dir = out / split / "images"
    out_label_dir = out / split / "labels"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(src_image_dir)
    if max_images is not None:
        image_paths = image_paths[:max_images]

    for index, image_path in enumerate(image_paths, start=1):
        image_bgr = read_image(cv2, image_path)
        processed = preprocess(
            cv2=cv2,
            image_bgr=image_bgr,
            mode=mode,
            clahe_clip=clahe_clip,
            clahe_grid=clahe_grid,
        )
        write_image(cv2, out_image_dir / image_path.name, processed)

        label_path = src_label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, out_label_dir / label_path.name)
        else:
            (out_label_dir / f"{image_path.stem}.txt").write_text("", encoding="utf-8")

        if index == 1 or index % 250 == 0 or index == len(image_paths):
            print(f"{split}: {index}/{len(image_paths)}", flush=True)


def preprocess(
    *,
    cv2,
    image_bgr: np.ndarray,
    mode: str,
    clahe_clip: float,
    clahe_grid: int,
) -> np.ndarray:
    result = image_bgr
    if mode in {"grayworld", "grayworld-clahe"}:
        result = gray_world_bgr(result)
    if mode in {"clahe", "grayworld-clahe"}:
        result = clahe_lab_bgr(
            cv2=cv2,
            image_bgr=result,
            clip_limit=clahe_clip,
            tile_grid_size=clahe_grid,
        )
    return result


def gray_world_bgr(image_bgr: np.ndarray) -> np.ndarray:
    image = image_bgr.astype(np.float32)
    channel_means = image.reshape(-1, 3).mean(axis=0)
    gray_mean = channel_means.mean()
    scale = gray_mean / np.maximum(channel_means, 1e-6)
    balanced = image * scale.reshape(1, 1, 3)
    return np.clip(balanced, 0, 255).astype(np.uint8)


def clahe_lab_bgr(
    *,
    cv2,
    image_bgr: np.ndarray,
    clip_limit: float,
    tile_grid_size: int,
) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    enhanced_l = clahe.apply(l_channel)
    enhanced = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def collect_images(path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(item for item in path.rglob("*") if item.suffix.lower() in exts)


def read_image(cv2, path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def write_image(cv2, path: Path, image_bgr: np.ndarray) -> None:
    suffix = path.suffix.lower()
    params = []
    if suffix in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    elif suffix == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    ok, encoded = cv2.imencode(suffix, image_bgr, params)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    encoded.tofile(path)


def write_data_yaml(out: Path) -> None:
    lines = [
        f"path: {out.resolve().as_posix()}",
        "train: train/images",
        "val: val/images",
        "",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in CLASS_NAMES.items())
    (out / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
