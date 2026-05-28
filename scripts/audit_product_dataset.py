import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_FILES = [
    Path("data/Indonesian Skincare Sample Dataset/product.csv"),
    Path("data/sociolla_products.csv"),
    Path("data/mixed_products.csv"),
]

WEB_ARTIFACT_RE = re.compile(
    r"\"@type\"\s*:|\"reviewRating\"\s*:|\"offers\"\s*:|@font-face|-webkit-|\.av_[a-z0-9_-]+|\bamp-[a-z0-9_-]+",
    flags=re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit skincare product CSV quality and class balance.")
    parser.add_argument("--csv", type=Path, action="append", default=None, help="CSV file to audit. Can repeat.")
    parser.add_argument("--out", type=Path, default=Path("reports/product_dataset_audit.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = args.csv or DEFAULT_FILES
    reports = []
    for path in paths:
        if not path.exists():
            print(f"missing: {path}")
            continue
        rows = read_csv(path)
        report = audit(path, rows)
        reports.append(report)
        print_report(report)
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {args.out}")


def audit(path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    schema = detect_schema(rows)
    normalized = [normalize_row(row, schema=schema) for row in rows]
    category_counts = Counter(row["category"] or "missing" for row in normalized)
    duplicate_keys = duplicate_count(normalized)
    fields = [
        "product_name",
        "brand",
        "category",
        "price",
        "ingredients",
        "concerns",
        "skin_type",
        "bpom_id",
        "product_url",
        "rating",
        "review_count",
    ]
    missing = {field: sum(1 for row in normalized if not row.get(field)) for field in fields}
    completeness = {
        field: round(1.0 - (count / max(len(normalized), 1)), 4)
        for field, count in missing.items()
    }
    return {
        "path": str(path),
        "rows": len(normalized),
        "schema": schema,
        "category_counts": dict(category_counts),
        "balance": balance_metrics(category_counts),
        "duplicate_products": duplicate_keys,
        "missing_counts": missing,
        "completeness": completeness,
        "invalid_price_count": sum(1 for row in normalized if row.get("price") and not valid_price(row["price"])),
        "invalid_rating_count": sum(1 for row in normalized if row.get("rating") and not valid_rating(row["rating"])),
        "invalid_bpom_count": sum(1 for row in normalized if row.get("bpom_id") and not valid_bpom(row["bpom_id"])),
        "web_artifact_count": sum(1 for row in normalized if has_web_artifact(row)),
    }


def detect_schema(rows: list[dict[str, str]]) -> str:
    fields = set(rows[0]) if rows else set()
    if "product_type" in fields or "ingredients_list" in fields:
        return "original_product_csv"
    return "normalized_kb_csv"


def normalize_row(row: dict[str, str], *, schema: str) -> dict[str, str]:
    if schema == "original_product_csv":
        return {
            "product_name": clean(row.get("product_name")),
            "brand": clean(row.get("brand")),
            "category": clean(row.get("product_type")).casefold(),
            "price": clean(row.get("discount_price") or row.get("normal_price")),
            "ingredients": clean(row.get("ingredients_list")),
            "concerns": clean(row.get("description_product")),
            "skin_type": "",
            "bpom_id": clean(row.get("bpom_id")),
            "product_url": clean(row.get("product_url")),
            "rating": clean(row.get("rating")),
            "review_count": clean(row.get("review_count")),
        }
    return {
        "product_name": clean(row.get("product_name")),
        "brand": clean(row.get("brand")),
        "category": clean(row.get("category")).casefold(),
        "price": clean(row.get("price")),
        "ingredients": clean(row.get("ingredients")),
        "concerns": clean(row.get("concerns")),
        "skin_type": clean(row.get("skin_type")),
        "bpom_id": clean(row.get("bpom_id")),
        "product_url": clean(row.get("product_url")),
        "rating": clean(row.get("rating")),
        "review_count": clean(row.get("review_count")),
    }


def balance_metrics(counts: Counter[str]) -> dict[str, Any]:
    values = [count for category, count in counts.items() if category != "missing"]
    if not values:
        return {"min": 0, "max": 0, "max_min_ratio": None, "majority_share": None}
    total = sum(values)
    minimum = min(values)
    maximum = max(values)
    return {
        "min": minimum,
        "max": maximum,
        "max_min_ratio": None if minimum == 0 else round(maximum / minimum, 3),
        "majority_share": round(maximum / total, 4),
    }


def duplicate_count(rows: list[dict[str, str]]) -> int:
    counts = Counter(
        f"{row.get('brand', '').casefold()}|{row.get('product_name', '').casefold()}"
        for row in rows
        if row.get("brand") or row.get("product_name")
    )
    return sum(count - 1 for count in counts.values() if count > 1)


def valid_price(value: str) -> bool:
    return bool(re.search(r"\d", value))


def valid_rating(value: str) -> bool:
    try:
        rating = float(value.replace(",", "."))
    except ValueError:
        return False
    return 0 <= rating <= 5


def valid_bpom(value: str) -> bool:
    return bool(re.fullmatch(r"N[A-Z]\d{9,}", value.upper()))


def has_web_artifact(row: dict[str, str]) -> bool:
    text = " ".join(row.get(field, "") for field in ["ingredients", "concerns", "skin_type"])
    return bool(WEB_ARTIFACT_RE.search(text))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [{key: clean(value) for key, value in row.items()} for row in csv.DictReader(file)]


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def print_report(report: dict[str, Any]) -> None:
    print(report["path"])
    print(f"  rows: {report['rows']}")
    print(f"  schema: {report['schema']}")
    print(f"  categories: {report['category_counts']}")
    print(f"  balance: {report['balance']}")
    print(f"  duplicate_products: {report['duplicate_products']}")
    print(f"  missing_counts: {report['missing_counts']}")
    print(f"  invalid_price_count: {report['invalid_price_count']}")
    print(f"  invalid_rating_count: {report['invalid_rating_count']}")
    print(f"  invalid_bpom_count: {report['invalid_bpom_count']}")
    print(f"  web_artifact_count: {report['web_artifact_count']}")


if __name__ == "__main__":
    main()
