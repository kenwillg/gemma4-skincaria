import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE_DIR / "data" / "sociolla_products.csv"
DEFAULT_OUTPUT = BASE_DIR / "data" / "sociolla_products.csv"
DEFAULT_REPORT = BASE_DIR / "reports" / "sociolla_products_cleaning_report.json"

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

CATEGORY_MAP = {
    "serum": "serum",
    "face serum": "serum",
    "facial serum": "serum",
    "toner": "toner",
    "face toner": "toner",
    "facial toner": "toner",
    "facial wash": "facial wash",
    "face wash": "facial wash",
    "cleanser": "facial wash",
    "cleansing foam": "facial wash",
    "foam": "facial wash",
}

CONCERN_ORDER = [
    "Acne Care",
    "Pore Care",
    "Oil Control",
    "Soothing",
    "Hydrating",
    "Barrier Care",
    "Brightening",
    "Smoothing",
    "Exfoliation",
    "Anti-Aging",
    "Cleansing",
    "Sensitive-Safe",
    "Skin Conditioning",
]

SKIN_TYPE_ORDER = ["dry/dehydrated", "oily/acne-prone", "sensitive", "all/unspecified"]

WEB_ARTIFACT_PATTERNS = [
    r'"@type"\s*:',
    r'"reviewRating"\s*:',
    r'"offers"\s*:',
    r"@font-face",
    r"-webkit-",
    r"\.av_[a-z0-9_-]+",
    r"\bamp-[a-z0-9_-]+",
    r"\bNuxt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and validate scraped Sociolla products for the Skincaria knowledge base."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--min-required",
        type=int,
        default=3,
        help="Minimum required fields among product_name, brand, category, product_url, concerns.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    raw_rows = read_csv(args.input)
    cleaned_rows, dropped_rows = clean_rows(raw_rows, min_required=args.min_required)
    report = build_report(
        input_path=args.input,
        output_path=args.output,
        raw_rows=raw_rows,
        cleaned_rows=cleaned_rows,
        dropped_rows=dropped_rows,
    )

    print_report(report)
    if args.dry_run:
        return

    write_csv(args.output, cleaned_rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved_clean_csv: {args.output}")
    print(f"saved_report: {args.report}")


def clean_rows(
    rows: list[dict[str, str]],
    *,
    min_required: int,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    by_key: dict[str, dict[str, str]] = {}
    dropped = []

    for index, row in enumerate(rows, start=1):
        cleaned = clean_row(row)
        required_score = sum(
            1
            for field in ["product_name", "brand", "category", "product_url", "concerns"]
            if cleaned.get(field)
        )
        if required_score < min_required:
            dropped.append(
                {
                    "row": index,
                    "reason": "too many required fields missing",
                    "required_score": required_score,
                    "product_name": cleaned.get("product_name", ""),
                    "brand": cleaned.get("brand", ""),
                    "product_url": cleaned.get("product_url", ""),
                }
            )
            continue

        key = dedupe_key(cleaned)
        if key in by_key:
            by_key[key] = merge_product(by_key[key], cleaned)
        else:
            by_key[key] = cleaned

    return sorted(by_key.values(), key=lambda item: (item.get("brand", ""), item.get("product_name", ""))), dropped


def clean_row(row: dict[str, str]) -> dict[str, str]:
    cleaned = {field: clean_value(row.get(field, "")) for field in FIELDS}
    cleaned["product_name"] = clean_product_name(cleaned["product_name"])
    cleaned["brand"] = clean_brand(cleaned["brand"])
    cleaned["category"] = normalize_category(cleaned["category"], cleaned["product_name"], cleaned["product_url"])
    cleaned["price"] = normalize_price(cleaned["price"])
    cleaned["ingredients"] = normalize_ingredients(cleaned["ingredients"])
    cleaned["skin_type"] = normalize_ordered_labels(cleaned["skin_type"], SKIN_TYPE_ORDER)
    cleaned["concerns"] = normalize_ordered_labels(cleaned["concerns"], CONCERN_ORDER)
    cleaned["allergen_flag"] = normalize_sentence(cleaned["allergen_flag"])
    cleaned["pregnancy_safe"] = normalize_sentence(cleaned["pregnancy_safe"])
    cleaned["ingredient_functions"] = normalize_list(cleaned["ingredient_functions"])
    cleaned["ingredient_warnings"] = normalize_warning_list(cleaned["ingredient_warnings"])
    cleaned["bpom_id"] = normalize_bpom(cleaned["bpom_id"])
    cleaned["product_url"] = normalize_url(cleaned["product_url"])
    cleaned["rating"] = normalize_rating(cleaned["rating"])
    cleaned["review_count"] = normalize_integer(cleaned["review_count"])
    cleaned["source"] = cleaned["source"] or "Sociolla"
    cleaned["scraped_at"] = normalize_timestamp(cleaned["scraped_at"])
    return cleaned


def clean_product_name(value: str) -> str:
    value = re.sub(r"^Jual\s+", "", value, flags=re.I)
    value = re.sub(r"\s*\|\s*Sociolla.*$", "", value, flags=re.I)
    value = re.sub(r"\s+", " ", value).strip(" -")
    return value


def clean_brand(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" -")
    return value


def normalize_category(category: str, product_name: str, url: str) -> str:
    haystack = f"{category} {product_name} {url}".casefold()
    for token, normalized in CATEGORY_MAP.items():
        if token in haystack:
            return normalized
    return category.casefold() or "skincare"


def normalize_price(value: str) -> str:
    cleaned = clean_value(value)
    if not cleaned:
        return ""
    digits = re.sub(r"[^\d]", "", cleaned)
    if not digits:
        return cleaned
    return f"Rp{int(digits):,}".replace(",", ".")


def normalize_ingredients(value: str) -> str:
    if has_web_artifact(value):
        return ""
    parts = split_list(value)
    normalized = []
    for part in parts:
        item = re.sub(r"\s+", " ", part).strip(" .;")
        if not item or has_web_artifact(item):
            continue
        if len(item) > 120:
            continue
        normalized.append(item)
    return ", ".join(unique_preserve_order(normalized))


def normalize_ordered_labels(value: str, order: list[str]) -> str:
    labels = split_list(value)
    canonical = {label.casefold(): label for label in order}
    found = []
    for label in labels:
        key = label.casefold()
        if key in canonical:
            found.append(canonical[key])
        else:
            found.append(label)
    ordered = [label for label in order if label in found]
    extras = [label for label in found if label not in ordered]
    return ", ".join(unique_preserve_order([*ordered, *extras]))


def normalize_list(value: str) -> str:
    if has_web_artifact(value):
        return ""
    return ", ".join(unique_preserve_order(split_list(value)))


def normalize_warning_list(value: str) -> str:
    if has_web_artifact(value):
        return ""
    parts = []
    for segment in re.split(r";|\n", value):
        cleaned = clean_value(segment)
        if cleaned and not has_web_artifact(cleaned):
            parts.append(cleaned)
    return "; ".join(unique_preserve_order(parts))


def normalize_sentence(value: str) -> str:
    if has_web_artifact(value):
        return ""
    return re.sub(r"\s+", " ", value).strip(" ;.")


def normalize_bpom(value: str) -> str:
    match = re.search(r"\bN[A-Z]\d{9,}\b", value.upper())
    return match.group(0) if match else ""


def normalize_url(value: str) -> str:
    cleaned = clean_value(value)
    if not cleaned:
        return ""
    return cleaned.split("#", 1)[0]


def normalize_rating(value: str) -> str:
    cleaned = clean_value(value).replace(",", ".")
    try:
        rating = float(cleaned)
    except ValueError:
        return ""
    if rating < 0 or rating > 5:
        return ""
    return f"{rating:.1f}"


def normalize_integer(value: str) -> str:
    cleaned = clean_value(value)
    if not cleaned:
        return ""
    cleaned = cleaned.replace(",", ".").casefold()
    try:
        if cleaned.endswith("k"):
            return str(int(float(cleaned[:-1]) * 1000))
        digits = re.sub(r"[^\d]", "", cleaned)
        return str(int(digits)) if digits else ""
    except ValueError:
        return ""


def normalize_timestamp(value: str) -> str:
    cleaned = clean_value(value)
    if not cleaned:
        return datetime.now(timezone.utc).isoformat()
    return cleaned


def merge_product(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    merged = dict(left)
    for field in FIELDS:
        current = merged.get(field, "")
        incoming = right.get(field, "")
        if not current and incoming:
            merged[field] = incoming
        elif field in {"ingredients", "concerns", "skin_type", "ingredient_functions"} and incoming:
            merged[field] = normalize_list(f"{current}, {incoming}")
        elif field == "ingredient_warnings" and incoming:
            merged[field] = normalize_warning_list(f"{current}; {incoming}")
        elif field == "rating" and incoming:
            merged[field] = incoming
        elif field == "review_count":
            merged[field] = max_numeric_string(current, incoming)
    return clean_row(merged)


def max_numeric_string(left: str, right: str) -> str:
    try:
        return str(max(int(left or 0), int(right or 0)))
    except ValueError:
        return right or left


def dedupe_key(row: dict[str, str]) -> str:
    brand = canonical(row.get("brand", ""))
    product = canonical(row.get("product_name", ""))
    if brand and product:
        return f"{brand}|{product}"
    return canonical(row.get("product_url", ""))


def build_report(
    *,
    input_path: Path,
    output_path: Path,
    raw_rows: list[dict[str, str]],
    cleaned_rows: list[dict[str, str]],
    dropped_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    required_fields = ["product_name", "brand", "category", "product_url", "concerns"]
    optional_fields = ["price", "ingredients", "rating", "review_count", "bpom_id"]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "output": str(output_path),
        "raw_rows": len(raw_rows),
        "cleaned_rows": len(cleaned_rows),
        "dropped_rows": len(dropped_rows),
        "duplicates_removed": max(0, len(raw_rows) - len(cleaned_rows) - len(dropped_rows)),
        "category_counts": Counter(row.get("category", "") for row in cleaned_rows),
        "required_field_missing_counts": missing_counts(cleaned_rows, required_fields),
        "optional_field_missing_counts": missing_counts(cleaned_rows, optional_fields),
        "dropped_examples": dropped_rows[:20],
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"raw_rows: {report['raw_rows']}")
    print(f"cleaned_rows: {report['cleaned_rows']}")
    print(f"dropped_rows: {report['dropped_rows']}")
    print(f"duplicates_removed: {report['duplicates_removed']}")
    print("category_counts:")
    for category, count in report["category_counts"].items():
        print(f"  {category}: {count}")
    print("required_field_missing_counts:")
    for field, count in report["required_field_missing_counts"].items():
        print(f"  {field}: {count}")


def missing_counts(rows: list[dict[str, str]], fields: list[str]) -> dict[str, int]:
    return {field: sum(1 for row in rows if not row.get(field)) for field in fields}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV is empty or invalid: {path}")
        return [
            {field: clean_value(row.get(field, "")) for field in FIELDS}
            for row in reader
            if any(clean_value(value) for value in row.values())
        ]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def split_list(value: str) -> list[str]:
    return [
        clean_value(part)
        for part in re.split(r",|;|\n", value)
        if clean_value(part)
    ]


def unique_preserve_order(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        cleaned = clean_value(value)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        result.append(cleaned)
        seen.add(key)
    return result


def canonical(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_value(value).casefold()).strip()


def clean_value(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def has_web_artifact(value: str) -> bool:
    if not value:
        return False
    if len(value) > 3000:
        return True
    return any(re.search(pattern, value, flags=re.I) for pattern in WEB_ARTIFACT_PATTERNS)


if __name__ == "__main__":
    main()
