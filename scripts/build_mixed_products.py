import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clean_sociolla_products import FIELDS, clean_row, write_csv


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL = BASE_DIR / "data" / "Indonesian Skincare Sample Dataset" / "product.csv"
DEFAULT_CLAIMS = BASE_DIR / "data" / "Indonesian Skincare Sample Dataset" / "product_claim_category.csv"
DEFAULT_INGREDIENTS = BASE_DIR / "data" / "Indonesian Skincare Sample Dataset" / "ingredients_category.csv"
DEFAULT_SOCIOLLA = BASE_DIR / "data" / "sociolla_products.csv"
DEFAULT_OUTPUT = BASE_DIR / "data" / "mixed_products.csv"
DEFAULT_REPORT = BASE_DIR / "reports" / "mixed_products_report.json"

CONCERN_KEYWORDS = {
    "Acne Care": ["acne", "jerawat", "blemish"],
    "Pore Care": ["pore", "pori", "komedo"],
    "Oil Control": ["oil", "sebum", "minyak"],
    "Soothing": ["soothing", "menenangkan", "redness", "kemerahan"],
    "Hydrating": ["hydrate", "hidrasi", "melembapkan", "moistur"],
    "Barrier Care": ["barrier", "ceramide", "panthenol"],
    "Brightening": ["bright", "cerah", "mencerahkan", "dark spot", "noda"],
    "Smoothing": ["smooth", "halus", "tekstur"],
    "Exfoliation": ["exfoliat", "peeling", "aha", "bha", "salicylic", "glycolic"],
    "Anti-Aging": ["anti-aging", "aging", "wrinkle", "kerutan", "elast"],
    "Cleansing": ["cleanse", "cleansing", "membersihkan", "pembersih"],
    "Sensitive-Safe": ["sensitive", "sensitif"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mixed KB CSV from original and scraped product data.")
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--ingredients", type=Path, default=DEFAULT_INGREDIENTS)
    parser.add_argument("--sociolla", type=Path, default=DEFAULT_SOCIOLLA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--target-per-category",
        type=int,
        default=100,
        help="Keep all original rows, then supplement each category from scraped rows up to this target. Use 0 for full merge.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_rows = read_csv(args.original)
    sociolla_rows = [clean_row(row) for row in read_csv(args.sociolla)]
    claim_map = load_claim_map(args.claims)
    ingredient_lookup = load_ingredient_lookup(args.ingredients)

    normalized_original = [
        normalize_original_row(row, claim_map=claim_map, ingredient_lookup=ingredient_lookup)
        for row in original_rows
    ]
    if args.target_per_category > 0:
        mixed_rows, duplicate_count, supplement_counts = build_target_balanced_rows(
            normalized_original,
            sociolla_rows,
            target=args.target_per_category,
        )
    else:
        mixed_rows, duplicate_count = merge_rows(normalized_original, sociolla_rows)
        supplement_counts = {}
    write_csv(args.output, mixed_rows)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(args.output),
        "target_per_category": args.target_per_category or None,
        "original_input_rows": len(original_rows),
        "sociolla_input_rows": len(sociolla_rows),
        "output_rows": len(mixed_rows),
        "merged_duplicates": duplicate_count,
        "supplement_counts": supplement_counts,
        "category_counts": dict(Counter(row["category"] for row in mixed_rows)),
        "source_counts": dict(Counter(row["source"] for row in mixed_rows)),
        "missing_counts": {
            field: sum(1 for row in mixed_rows if not row.get(field))
            for field in ["product_name", "brand", "category", "price", "ingredients", "concerns", "skin_type", "bpom_id"]
        },
        "duplicate_products": count_duplicate_products(mixed_rows),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))


def normalize_original_row(
    row: dict[str, str],
    *,
    claim_map: dict[str, str],
    ingredient_lookup: dict[str, dict[str, str]],
) -> dict[str, str]:
    product_name = clean_text(row.get("product_name", ""))
    brand = clean_text(row.get("brand", ""))
    ingredients = clean_text(row.get("ingredients_list", ""))
    claims = split_claims(row.get("description_product", ""))
    claim_categories = unique(claim_map.get(canonical(claim), claim) for claim in claims)
    concerns = unique([*claim_categories, *infer_concerns(" ".join([product_name, ingredients, *claims]))])
    ingredient_names = split_ingredients(ingredients)
    functions, warnings = ingredient_notes(ingredient_names, ingredient_lookup)

    normalized = {
        "product_name": product_name,
        "brand": brand,
        "category": clean_text(row.get("product_type", "")).casefold(),
        "price": format_price(row.get("discount_price", "") or row.get("normal_price", "")),
        "ingredients": ingredients,
        "skin_type": infer_skin_type(" ".join([product_name, ingredients, *claims, *concerns])),
        "concerns": ", ".join(concerns or ["Skin Conditioning"]),
        "allergen_flag": "yes: " + "; ".join(warnings[:8]) if warnings else "none detected from matched ingredient data",
        "pregnancy_safe": infer_pregnancy_safe(warnings),
        "ingredient_functions": ", ".join(functions),
        "ingredient_warnings": "; ".join(warnings),
        "bpom_id": clean_text(row.get("bpom_id", "")),
        "product_url": clean_text(row.get("product_url", "")),
        "rating": clean_text(row.get("rating", "")),
        "review_count": clean_text(row.get("review_count", "")),
        "source": "Original Indonesian Skincare Sample Dataset",
        "scraped_at": "",
    }
    return clean_row(normalized)


def merge_rows(original_rows: list[dict[str, str]], sociolla_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    by_key: dict[str, dict[str, str]] = {}
    duplicate_count = 0
    for row in [*sociolla_rows, *original_rows]:
        key = dedupe_key(row)
        if key in by_key:
            duplicate_count += 1
            by_key[key] = merge_product(by_key[key], row)
        else:
            by_key[key] = row
    rows = sorted(by_key.values(), key=lambda item: (item.get("category", ""), item.get("brand", ""), item.get("product_name", "")))
    return rows, duplicate_count


def build_target_balanced_rows(
    original_rows: list[dict[str, str]],
    sociolla_rows: list[dict[str, str]],
    *,
    target: int,
) -> tuple[list[dict[str, str]], int, dict[str, int]]:
    sociolla_by_key = {dedupe_key(row): row for row in sociolla_rows}
    mandatory_rows = []
    mandatory_keys = set()
    overlap_count = 0

    for original in original_rows:
        key = dedupe_key(original)
        mandatory_keys.add(key)
        if key in sociolla_by_key:
            mandatory_rows.append(merge_product(sociolla_by_key[key], original))
            overlap_count += 1
        else:
            mandatory_rows.append(original)

    mandatory_counts = Counter(row["category"] for row in mandatory_rows)
    overfull = {category: count for category, count in mandatory_counts.items() if count > target}
    if overfull:
        raise ValueError(f"Original dataset already exceeds target_per_category={target}: {overfull}")

    selected = list(mandatory_rows)
    selected_counts = Counter(row["category"] for row in selected)
    supplement_counts: Counter[str] = Counter()
    scraped_candidates = [
        row for row in sociolla_rows
        if dedupe_key(row) not in mandatory_keys
    ]
    scraped_candidates.sort(key=supplement_rank, reverse=True)

    for row in scraped_candidates:
        category = row["category"]
        if selected_counts[category] >= target:
            continue
        selected.append(row)
        selected_counts[category] += 1
        supplement_counts[category] += 1

    shortfalls = {
        category: target - selected_counts[category]
        for category in sorted(set(selected_counts) | {"facial wash", "serum", "toner"})
        if selected_counts[category] < target
    }
    if shortfalls:
        raise ValueError(f"Not enough scraped products to reach target_per_category={target}: {shortfalls}")

    selected.sort(key=lambda item: (item.get("category", ""), source_order(item), item.get("brand", ""), item.get("product_name", "")))
    return selected, overlap_count, dict(supplement_counts)


def supplement_rank(row: dict[str, str]) -> tuple[int, float, int, str]:
    completeness_fields = [
        "ingredients",
        "bpom_id",
        "ingredient_functions",
        "ingredient_warnings",
        "concerns",
        "skin_type",
        "rating",
        "review_count",
    ]
    completeness = sum(1 for field in completeness_fields if row.get(field))
    try:
        rating = float(row.get("rating", "") or 0)
    except ValueError:
        rating = 0.0
    try:
        review_count = int(row.get("review_count", "") or 0)
    except ValueError:
        review_count = 0
    return completeness, rating, review_count, row.get("product_name", "")


def source_order(row: dict[str, str]) -> int:
    source = row.get("source", "")
    return 0 if "Original Indonesian Skincare Sample Dataset" in source else 1


def merge_product(existing: dict[str, str], incoming: dict[str, str]) -> dict[str, str]:
    merged = dict(existing)
    incoming_is_original = incoming.get("source") == "Original Indonesian Skincare Sample Dataset"
    for field in FIELDS:
        current = merged.get(field, "")
        value = incoming.get(field, "")
        if field == "source" and value and value not in current:
            merged[field] = f"{current} + {value}" if current else value
        elif field in {"ingredients", "bpom_id", "ingredient_functions", "ingredient_warnings"} and incoming_is_original and value:
            merged[field] = value
        elif not current and value:
            merged[field] = value
        elif field in {"concerns", "skin_type"} and value:
            merged[field] = ", ".join(unique([*split_list(current), *split_list(value)]))
    return clean_row(merged)


def load_claim_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rows = read_csv(path)
    return {
        canonical(row.get("description_product", "")): clean_text(row.get("claim_category", ""))
        for row in rows
        if row.get("description_product") and row.get("claim_category")
    }


def load_ingredient_lookup(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    lookup = {}
    for row in read_csv(path):
        name = clean_text(row.get("ingredient_name", ""))
        if name:
            lookup[canonical(name)] = {key.lower(): clean_text(value) for key, value in row.items()}
    return lookup


def ingredient_notes(names: list[str], lookup: dict[str, dict[str, str]]) -> tuple[list[str], list[str]]:
    functions = []
    warnings = []
    for name in names:
        row = lookup.get(canonical(name))
        if not row:
            continue
        functions.extend([row.get("function1", ""), row.get("function2", "")])
        warning_values = unique([row.get("warning1", ""), row.get("warning2", "")])
        if warning_values:
            warnings.append(f"{name}: {', '.join(warning_values)}")
    return unique(functions), unique(warnings)


def infer_concerns(text: str) -> list[str]:
    lower = text.casefold()
    return [label for label, tokens in CONCERN_KEYWORDS.items() if any(token in lower for token in tokens)]


def infer_skin_type(text: str) -> str:
    lower = text.casefold()
    labels = []
    if any(token in lower for token in ["dry", "kering", "dehydrated", "hidrasi", "moistur", "melembapkan"]):
        labels.append("dry/dehydrated")
    if any(token in lower for token in ["oily", "berminyak", "acne", "jerawat", "sebum", "komedo", "pore", "pori"]):
        labels.append("oily/acne-prone")
    if any(token in lower for token in ["sensitive", "sensitif", "redness", "kemerahan", "irritation"]):
        labels.append("sensitive")
    return ", ".join(unique(labels or ["all/unspecified"]))


def infer_pregnancy_safe(warnings: list[str]) -> str:
    avoid = [warning.split(":", 1)[0] for warning in warnings if "pregnancy" in warning.casefold()]
    if avoid:
        return "avoid during pregnancy: " + ", ".join(unique(avoid))
    return "unknown"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [{key: clean_text(value) for key, value in row.items()} for row in csv.DictReader(file)]


def format_price(value: str) -> str:
    digits = re.sub(r"[^\d]", "", clean_text(value))
    return f"Rp{int(digits):,}".replace(",", ".") if digits else ""


def split_claims(value: str) -> list[str]:
    return unique(part.strip(" .") for part in re.split(r",|\n|;", clean_text(value)) if part.strip(" ."))


def split_ingredients(value: str) -> list[str]:
    return unique(part.strip(" .") for part in re.split(r",|;", clean_text(value)) if part.strip(" ."))


def split_list(value: str) -> list[str]:
    return [part.strip() for part in clean_text(value).split(",") if part.strip()]


def unique(values: Any) -> list[str]:
    result = []
    seen = set()
    for value in values:
        cleaned = clean_text(value)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        result.append(cleaned)
        seen.add(key)
    return result


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def canonical(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_text(value).casefold()).strip()


def dedupe_key(row: dict[str, str]) -> str:
    return f"{canonical(row.get('brand', ''))}|{canonical(row.get('product_name', ''))}"


def count_duplicate_products(rows: list[dict[str, str]]) -> int:
    counts = Counter(dedupe_key(row) for row in rows)
    return sum(count - 1 for count in counts.values() if count > 1)


if __name__ == "__main__":
    main()
