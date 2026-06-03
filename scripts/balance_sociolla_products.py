import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


FIELDS = [
    "product_name",
    "brand",
    "category",
    "price",
    "ingredients",
    "skin_type",
    "concerns",
    "allergen_flag",
    "pregnancy_safe",
    "ingredient_functions",
    "ingredient_warnings",
    "bpom_id",
    "product_url",
    "rating",
    "review_count",
    "source",
    "scraped_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Balance normalized Sociolla products to N rows per category.")
    parser.add_argument("--input", type=Path, default=Path("data/sociolla_products.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/sociolla_products.csv"))
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--category", action="append", default=["serum", "toner", "facial wash"])
    parser.add_argument("--report", type=Path, default=Path("reports/sociolla_products_balance_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input)
    deduped = dedupe(rows)
    selected = []
    shortfalls = {}
    for category in args.category:
        category_rows = [row for row in deduped if row.get("category") == category]
        ranked = sorted(category_rows, key=quality_key, reverse=True)
        selected.extend(ranked[: args.target])
        if len(ranked) < args.target:
            shortfalls[category] = args.target - len(ranked)

    selected = sorted(selected, key=lambda row: (row.get("category", ""), row.get("brand", ""), row.get("product_name", "")))
    write_csv(args.output, selected)
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "target_per_category": args.target,
        "input_rows": len(rows),
        "deduped_rows": len(deduped),
        "output_rows": len(selected),
        "category_counts": dict(Counter(row.get("category", "missing") for row in selected)),
        "shortfalls": shortfalls,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if shortfalls:
        raise SystemExit("Not enough products to meet target for all categories.")


def quality_key(row: dict[str, str]) -> tuple[int, float, int, str]:
    completeness = sum(
        1
        for field in [
            "price",
            "ingredients",
            "concerns",
            "skin_type",
            "bpom_id",
            "rating",
            "review_count",
            "product_url",
        ]
        if row.get(field)
    )
    try:
        rating = float((row.get("rating") or "0").replace(",", "."))
    except ValueError:
        rating = 0.0
    try:
        reviews = int(row.get("review_count") or 0)
    except ValueError:
        reviews = 0
    return completeness, rating, reviews, row.get("product_name", "")


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_key = {}
    for row in rows:
        key = f"{row.get('brand', '').casefold()}|{row.get('product_name', '').casefold()}"
        if key in by_key:
            by_key[key] = better_row(by_key[key], row)
        else:
            by_key[key] = row
    return list(by_key.values())


def better_row(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    return right if quality_key(right) > quality_key(left) else left


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [{field: (row.get(field, "") or "").strip() for field in FIELDS} for row in csv.DictReader(file)]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


if __name__ == "__main__":
    main()
