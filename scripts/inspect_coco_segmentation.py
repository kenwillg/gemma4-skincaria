import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect COCO segmentation field types.")
    parser.add_argument(
        "annotation_path",
        nargs="?",
        type=Path,
        default=Path("data/skincaria-dataset.coco/train/_annotations.coco.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_path = args.annotation_path
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {item["id"]: item["name"] for item in coco.get("categories", [])}
    counts: Counter[str] = Counter()
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    sample = None

    for annotation in coco.get("annotations", []):
        segmentation = annotation.get("segmentation")
        if isinstance(segmentation, list):
            if not segmentation:
                kind = "empty_list"
            elif all(isinstance(value, (int, float)) for value in segmentation):
                kind = "flat_polygon"
            else:
                kind = "polygon_list"
        elif isinstance(segmentation, dict):
            kind = "rle_dict"
        else:
            kind = "missing"

        counts[kind] += 1
        by_category[categories[annotation["category_id"]]][kind] += 1
        if kind != "empty_list" and sample is None:
            sample = annotation

    print(f"annotation_path: {annotation_path}")
    print(f"segmentation_types: {dict(counts)}")
    for category, category_counts in by_category.items():
        print(f"{category}: {dict(category_counts)}")
    print(f"sample_nonempty: {sample}")


if __name__ == "__main__":
    main()
