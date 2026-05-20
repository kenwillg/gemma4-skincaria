import argparse
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path
    counts: Counter

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a class-balanced YOLO detection train split while keeping val unchanged."
    )
    parser.add_argument("--src", type=Path, default=Path("data/skincaria-yolo-detect"))
    parser.add_argument("--out", type=Path, default=Path("data/skincaria-yolo-detect-balanced"))
    parser.add_argument(
        "--target",
        type=int,
        default=1200,
        help="Target training boxes per class. Use 900-1500 for this dataset.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--copy-val",
        action="store_true",
        help="Copy val files into the balanced folder. Default data.yaml points val back to source.",
    )
    return parser.parse_args()


def read_label_counts(label_path: Path) -> Counter:
    counts: Counter = Counter()
    if not label_path.exists():
        return counts
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        counts[int(float(parts[0]))] += 1
    return counts


def find_samples(src: Path) -> list[Sample]:
    image_dir = src / "train" / "images"
    label_dir = src / "train" / "labels"
    samples: list[Sample] = []
    for image in sorted(image_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_EXTS:
            continue
        label = label_dir / f"{image.stem}.txt"
        counts = read_label_counts(label)
        if counts:
            samples.append(Sample(image=image, label=label, counts=counts))
    return samples


def class_totals(samples: list[Sample]) -> Counter:
    totals: Counter = Counter()
    for sample in samples:
        totals.update(sample.counts)
    return totals


def choose_initial_samples(samples: list[Sample], target: int) -> list[Sample]:
    selected: list[Sample] = []
    totals: Counter = Counter()
    remaining = samples[:]

    while remaining and any(totals[c] < target for c in range(10)):
        def score(sample: Sample) -> tuple[float, int]:
            useful = 0
            overflow = 0
            for cls, count in sample.counts.items():
                deficit = max(0, target - totals[cls])
                useful += min(count, deficit)
                overflow += max(0, totals[cls] + count - target)
            return (useful / max(sample.total, 1), -overflow)

        best = max(remaining, key=score)
        useful_ratio, _ = score(best)
        if useful_ratio <= 0:
            break
        selected.append(best)
        totals.update(best.counts)
        remaining.remove(best)

    return selected


def oversample_to_target(selected: list[Sample], all_samples: list[Sample], target: int) -> list[Sample]:
    totals = class_totals(selected)
    output = selected[:]
    guard = 0

    while any(totals[c] < target for c in range(10)):
        guard += 1
        if guard > target * 20:
            raise RuntimeError("Oversampling guard tripped; check labels or target.")

        deficits = {c: target - totals[c] for c in range(10) if totals[c] < target}
        best = max(
            all_samples,
            key=lambda sample: (
                sum(min(sample.counts.get(c, 0), deficit) for c, deficit in deficits.items()),
                -sample.total,
            ),
        )
        useful = sum(min(best.counts.get(c, 0), deficit) for c, deficit in deficits.items())
        if useful <= 0:
            missing = ", ".join(str(c) for c in deficits)
            raise RuntimeError(f"No samples available for classes: {missing}")
        output.append(best)
        totals.update(best.counts)

    return output


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_pair(sample: Sample, image_out: Path, label_out: Path, index: int) -> None:
    suffix = f"_bal{index:05d}"
    out_image = image_out / f"{sample.image.stem}{suffix}{sample.image.suffix}"
    out_label = label_out / f"{sample.image.stem}{suffix}.txt"
    shutil.copy2(sample.image, out_image)
    shutil.copy2(sample.label, out_label)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def write_yaml(src: Path, out: Path, copy_val: bool) -> None:
    names = [
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
    val_path = "val/images" if copy_val else str((src / "val" / "images").resolve()).replace("\\", "/")
    lines = [
        f"path: {str(out.resolve()).replace('\\', '/')}",
        "train: train/images",
        f"val: {val_path}",
        "",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(names))
    (out / "data.yaml").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    samples = find_samples(args.src)
    if not samples:
        raise FileNotFoundError(f"No labeled train samples found in {args.src}")

    initial = choose_initial_samples(samples, args.target)
    balanced = oversample_to_target(initial, samples, args.target)

    image_out = args.out / "train" / "images"
    label_out = args.out / "train" / "labels"
    reset_dir(image_out)
    reset_dir(label_out)

    for index, sample in enumerate(balanced):
        copy_pair(sample, image_out, label_out, index)

    if args.copy_val:
        copy_tree(args.src / "val", args.out / "val")

    write_yaml(args.src, args.out, args.copy_val)

    before = class_totals(samples)
    after = class_totals(
        [
            Sample(
                image=Path(""),
                label=Path(""),
                counts=read_label_counts(label),
            )
            for label in label_out.glob("*.txt")
        ]
    )

    print("Original train class counts:")
    for cls in range(10):
        print(f"  {cls}: {before[cls]}")
    print("Balanced train class counts:")
    for cls in range(10):
        print(f"  {cls}: {after[cls]}")
    print(f"Images written: {len(list(image_out.iterdir()))}")
    print(f"Dataset yaml: {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
