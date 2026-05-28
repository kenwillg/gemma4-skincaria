import argparse
import csv
import html
import json
import re
import subprocess
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT = BASE_DIR / "data" / "sociolla_products.csv"
DEFAULT_RAW_OUT = BASE_DIR / "data" / "raw" / "sociolla_products_raw.jsonl"
DEFAULT_INGREDIENT_CSV = BASE_DIR / "data" / "Indonesian Skincare Sample Dataset" / "ingredients_category.csv"
PRODUCT_SITEMAP = "https://www.sociolla.com/sitemap/product.xml"

NORMALIZED_FIELDS = [
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

CATEGORY_ALIASES = {
    "face-serum": "serum",
    "190-face-serum": "serum",
    "serum": "serum",
    "toner": "toner",
    "face-toner": "toner",
    "facial-toner": "toner",
    "face-wash": "facial wash",
    "facial-wash": "facial wash",
    "2277-face-wash": "facial wash",
    "cleanser": "facial wash",
}

CONCERN_KEYWORDS = {
    "Acne Care": ["acne", "jerawat", "blemish", "p.acnes", "acne-prone"],
    "Pore Care": ["pore", "pori", "komedo", "blackhead", "whitehead"],
    "Oil Control": ["oil", "sebum", "minyak", "berminyak"],
    "Soothing": ["soothe", "calm", "menenangkan", "redness", "kemerahan", "inflamasi", "irritation"],
    "Hydrating": ["hydrate", "hydrating", "hidrasi", "melembapkan", "moistur", "hyaluronic"],
    "Barrier Care": ["barrier", "skin barrier", "ceramide", "panthenol"],
    "Brightening": ["bright", "cerah", "mencerahkan", "dark spot", "noda", "hyperpigmentation"],
    "Smoothing": ["smooth", "halus", "tekstur", "texture"],
    "Exfoliation": ["exfoliat", "peeling", "aha", "bha", "salicylic", "glycolic", "lactic acid"],
    "Anti-Aging": ["anti-aging", "aging", "wrinkle", "kerutan", "fine line", "firmness", "elast"],
    "Cleansing": ["cleanse", "cleansing", "membersihkan", "pembersih", "kotoran", "makeup"],
    "Sensitive-Safe": ["sensitive", "sensitif", "hypoallergenic", "fragrance free"],
}

INGREDIENT_WARNING_TOKENS = [
    "fragrance",
    "parfum",
    "alcohol",
    "denat",
    "essential oil",
    "menthol",
    "peppermint",
    "citrus",
    "limonene",
    "linalool",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Sociolla skincare products into the normalized Skincaria KB CSV schema."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--raw-out", type=Path, default=DEFAULT_RAW_OUT)
    parser.add_argument("--ingredient-csv", type=Path, default=DEFAULT_INGREDIENT_CSV)
    parser.add_argument("--from-sitemap", action="store_true", help="Discover product URLs from Sociolla sitemap.")
    parser.add_argument("--sitemap-url", default=PRODUCT_SITEMAP)
    parser.add_argument("--seed-url", action="append", default=[], help="Specific product URL to scrape. Can repeat.")
    parser.add_argument("--seed-file", type=Path, default=None, help="Text file with one product URL per line.")
    parser.add_argument(
        "--category-keyword",
        action="append",
        default=[],
        help="URL keyword to keep from sitemap, e.g. face-serum, face-wash, toner. Can repeat.",
    )
    parser.add_argument("--max-products", type=int, default=100)
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--append", action="store_true", help="Append/merge with existing output CSV.")
    parser.add_argument("--no-clean", action="store_true", help="Skip post-scrape cleaning/validation.")
    parser.add_argument("--dry-run", action="store_true", help="Discover URLs but do not fetch product pages.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    category_keywords = args.category_keyword or ["face-serum", "serum", "face-wash", "2277-face-wash", "toner"]
    ingredient_lookup = load_ingredient_lookup(args.ingredient_csv)
    seed_urls = collect_seed_urls(args)

    with httpx.Client(
        timeout=args.timeout,
        follow_redirects=True,
        headers={
            "User-Agent": "SkincariaResearchBot/0.1 (+educational research; contact: local project)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    ) as client:
        if args.from_sitemap:
            discovered = discover_product_urls(
                client=client,
                sitemap_url=args.sitemap_url,
                category_keywords=category_keywords,
                limit=max(args.max_products * 4, args.max_products),
            )
            seed_urls.extend(discovered)

        urls = unique_preserve_order(normalize_product_url(url) for url in seed_urls if url.strip())
        urls = [url for url in urls if keep_category_url(url, category_keywords)]
        existing_products = read_existing_products(args.out) if args.append else []
        if existing_products:
            existing_urls = {normalize_product_url(row.get("product_url", "")) for row in existing_products if row.get("product_url")}
            before_skip = len(urls)
            urls = [url for url in urls if normalize_product_url(url) not in existing_urls]
            print(f"skipped_existing_urls: {before_skip - len(urls)}")
        urls = urls[: args.max_products]

        print(f"candidate_urls: {len(urls)}")
        if args.dry_run:
            for url in urls:
                print(url)
            return

        products = []
        raw_rows = []
        for index, url in enumerate(urls, start=1):
            print(f"[{index}/{len(urls)}] {url}", flush=True)
            try:
                raw = fetch_text(client, url)
                product = parse_product_page(url=url, raw_html=raw, ingredient_lookup=ingredient_lookup)
            except Exception as exc:
                print(f"  skipped: {exc}")
                continue
            if not product.get("product_name") or not product.get("brand"):
                print("  skipped: missing product name or brand")
                continue
            products.append(product)
            raw_rows.append(
                {
                    "url": url,
                    "scraped_at": product["scraped_at"],
                    "product": product,
                }
            )
            print(f"  ok: {product['brand']} - {product['product_name']}")
            if index < len(urls):
                time.sleep(max(args.delay, 0.0))

    merged = merge_products(existing_products if args.append else [], products)
    write_products(args.out, merged)
    append_raw(args.raw_out, raw_rows)
    if not args.no_clean:
        run_cleaner(args.out)
    print(f"saved_products: {len(merged)} -> {args.out}")
    print(f"saved_raw_rows: {len(raw_rows)} -> {args.raw_out}")
    print("rebuild KB after scraping if needed:")
    print("  Remove-Item -Recurse -Force chroma_db")
    print("  .\\.venv\\Scripts\\python.exe main.py")


def run_cleaner(path: Path) -> None:
    cleaner = BASE_DIR / "scripts" / "clean_sociolla_products.py"
    if not cleaner.exists():
        return
    subprocess.run(
        [
            sys.executable,
            str(cleaner),
            "--input",
            str(path),
            "--output",
            str(path),
        ],
        check=True,
    )


def collect_seed_urls(args: argparse.Namespace) -> list[str]:
    urls = list(args.seed_url)
    if args.seed_file and args.seed_file.exists():
        urls.extend(
            line.strip()
            for line in args.seed_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    return urls


def discover_product_urls(
    *,
    client: httpx.Client,
    sitemap_url: str,
    category_keywords: list[str],
    limit: int,
) -> list[str]:
    seen = set()
    queue = [sitemap_url]
    product_urls = []

    while queue and len(product_urls) < limit:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        xml_text = fetch_text(client, url)
        root = ET.fromstring(xml_text)
        namespace = ""
        if root.tag.startswith("{"):
            namespace = root.tag.split("}", 1)[0] + "}"

        for loc in root.findall(f".//{namespace}loc"):
            value = (loc.text or "").strip()
            if not value:
                continue
            lower = value.lower()
            if lower.endswith(".xml"):
                queue.append(value)
            elif keep_category_url(value, category_keywords):
                product_urls.append(value)
                if len(product_urls) >= limit:
                    break

    return unique_preserve_order(product_urls)


def fetch_text(client: httpx.Client, url: str) -> str:
    response = client.get(url)
    response.raise_for_status()
    return response.text


def normalize_product_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url.strip())
    if not parsed.scheme:
        parsed = urllib.parse.urlparse("https://" + url.strip())
    path = parsed.path
    if "/amp/" not in path:
        path = "/amp" + path if path.startswith("/") else "/amp/" + path
    query = parsed.query
    if "screen=mobile" not in query:
        query = query + ("&" if query else "") + "screen=mobile"
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, "", query, ""))


def keep_category_url(url: str, category_keywords: list[str]) -> bool:
    lower = url.lower()
    return any(keyword.lower() in lower for keyword in category_keywords)


def parse_product_page(
    *,
    url: str,
    raw_html: str,
    ingredient_lookup: dict[str, dict[str, str]],
) -> dict[str, str]:
    text = HtmlTextExtractor.extract(raw_html)
    lines = clean_lines(text.splitlines())
    joined = "\n".join(lines)
    meta = extract_meta(raw_html)
    jsonld = extract_jsonld(raw_html)

    product_name = first_nonempty(
        find_jsonld_value(jsonld, ["name"]),
        clean_product_title(meta.get("og:title", "")),
        first_heading(raw_html),
        infer_product_name_from_lines(lines),
    )
    brand = first_nonempty(
        find_jsonld_brand(jsonld),
        infer_brand_from_lines(lines, product_name),
    )
    category = infer_category(url, joined)
    normal_price, discount_price = infer_prices(joined, jsonld)
    rating, review_count = infer_rating(joined, jsonld)
    description = extract_section(joined, ["DESCRIPTION", "DESKRIPSI"], ["HOW TO USE", "CARA PENGGUNAAN", "AVERAGE USER RATING"])
    ingredients = extract_ingredients(joined)
    bpom_id = first_regex(joined, r"\bN[A-Z]\d{9,}\b")
    concerns = infer_concerns(" ".join([description, product_name, ingredients]))
    skin_type = infer_skin_type(" ".join([description, product_name, ingredients]))
    pregnancy_safe = infer_pregnancy_safe(description)
    ingredient_names = parse_ingredient_list(ingredients)
    ingredient_functions, ingredient_warnings = ingredient_notes(ingredient_names, ingredient_lookup)
    allergen_flag = infer_allergen_flag(ingredients, ingredient_warnings, description)

    return {
        "product_name": product_name,
        "brand": brand,
        "category": category,
        "price": format_price(discount_price or normal_price),
        "ingredients": ingredients,
        "skin_type": skin_type,
        "concerns": ", ".join(concerns),
        "allergen_flag": allergen_flag,
        "pregnancy_safe": pregnancy_safe,
        "ingredient_functions": ", ".join(ingredient_functions),
        "ingredient_warnings": "; ".join(ingredient_warnings),
        "bpom_id": bpom_id,
        "product_url": url,
        "rating": rating,
        "review_count": review_count,
        "source": "Sociolla AMP page",
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


class HtmlTextExtractor(HTMLParser):
    BLOCK_TAGS = {"p", "div", "br", "li", "h1", "h2", "h3", "section", "article", "tr"}
    IGNORED_TAGS = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored_depth = 0

    @classmethod
    def extract(cls, raw_html: str) -> str:
        parser = cls()
        parser.feed(raw_html)
        return html.unescape("".join(parser.parts))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_name = tag.lower()
        if tag_name in self.IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if tag_name in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_name = tag.lower()
        if tag_name in self.IGNORED_TAGS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if tag_name in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        cleaned = re.sub(r"\s+", " ", data).strip()
        if cleaned:
            self.parts.append(cleaned + " ")


def extract_meta(raw_html: str) -> dict[str, str]:
    result: dict[str, str] = {}
    pattern = re.compile(r"<meta\s+([^>]+)>", flags=re.I)
    for match in pattern.finditer(raw_html):
        attrs = dict(re.findall(r'([\w:-]+)=["\']([^"\']*)["\']', match.group(1)))
        key = attrs.get("property") or attrs.get("name")
        content = attrs.get("content")
        if key and content:
            result[key] = html.unescape(content.strip())
    return result


def extract_jsonld(raw_html: str) -> list[Any]:
    blocks = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        raw_html,
        flags=re.I | re.S,
    )
    payloads = []
    for block in blocks:
        cleaned = html.unescape(block.strip())
        try:
            payloads.append(json.loads(cleaned))
        except json.JSONDecodeError:
            continue
    return payloads


def find_jsonld_value(payloads: list[Any], keys: list[str]) -> str:
    for payload in walk_json(payloads):
        if isinstance(payload, dict):
            if payload.get("@type") == "Product" or "offers" in payload:
                for key in keys:
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
    return ""


def find_jsonld_brand(payloads: list[Any]) -> str:
    for payload in walk_json(payloads):
        if isinstance(payload, dict):
            brand = payload.get("brand")
            if isinstance(brand, str) and brand.strip():
                return brand.strip()
            if isinstance(brand, dict):
                name = brand.get("name")
                if isinstance(name, str) and name.strip():
                    return name.strip()
    return ""


def walk_json(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_json(child)


def clean_lines(lines: list[str]) -> list[str]:
    result = []
    for line in lines:
        cleaned = re.sub(r"\s+", " ", html.unescape(line)).strip()
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


def clean_product_title(value: str) -> str:
    value = re.sub(r"^Jual\s+(Skin Care|Skincare)?\s*", "", value, flags=re.I).strip()
    value = re.sub(r"\s*\|\s*Sociolla.*$", "", value, flags=re.I).strip()
    return value


def first_heading(raw_html: str) -> str:
    for tag in ["h1", "h2"]:
        match = re.search(fr"<{tag}[^>]*>(.*?)</{tag}>", raw_html, flags=re.I | re.S)
        if match:
            return clean_html_fragment(match.group(1))
    return ""


def infer_product_name_from_lines(lines: list[str]) -> str:
    candidates = [
        line
        for line in lines[:40]
        if not line.startswith("Rp ")
        and not re.match(r"^\d+%$", line)
        and line.upper() not in {"DESCRIPTION", "HOW TO USE", "SIZE:", "QUANTITY", "ADD TO WISHLIST"}
        and len(line) >= 4
    ]
    return max(candidates, key=len, default="")


def infer_brand_from_lines(lines: list[str], product_name: str) -> str:
    if product_name in lines:
        idx = lines.index(product_name)
        window = lines[max(0, idx - 8) : idx + 3]
        for item in reversed(window):
            if item != product_name and is_probable_brand(item):
                return item
    for item in lines[:30]:
        if is_probable_brand(item):
            return item
    return ""


def is_probable_brand(value: str) -> bool:
    if len(value) > 40 or len(value) < 2:
        return False
    if value.startswith("Rp ") or value.upper() in {"DESCRIPTION", "SIZE:", "QUANTITY", "ADD TO WISHLIST"}:
        return False
    if re.search(r"\d", value):
        return False
    return bool(re.match(r"^[A-Za-z0-9&.' -]+$", value))


def infer_category(url: str, text: str) -> str:
    lower = (url + " " + text[:300]).lower()
    for token, category in CATEGORY_ALIASES.items():
        if token in lower:
            return category
    return "skincare"


def infer_prices(text: str, jsonld: list[Any]) -> tuple[str, str]:
    json_price = ""
    for payload in walk_json(jsonld):
        if isinstance(payload, dict):
            offers = payload.get("offers")
            if isinstance(offers, dict) and offers.get("price"):
                json_price = str(offers.get("price"))
                break
    prices = re.findall(r"Rp\s*[\d.]+", text)
    if len(prices) >= 2:
        return prices[0], prices[1]
    if len(prices) == 1:
        return prices[0], ""
    return json_price, ""


def infer_rating(text: str, jsonld: list[Any]) -> tuple[str, str]:
    for payload in walk_json(jsonld):
        if isinstance(payload, dict):
            rating = payload.get("aggregateRating")
            if isinstance(rating, dict):
                return str(rating.get("ratingValue", "") or ""), str(rating.get("reviewCount", "") or "")
    match = re.search(r"\b([1-5][.,]\d)\s*\(([\d.,kK]+)\)", text)
    if match:
        return match.group(1).replace(",", "."), normalize_count(match.group(2))
    return "", ""


def extract_section(text: str, starts: list[str], stops: list[str]) -> str:
    upper = text.upper()
    start_idx = -1
    for start in starts:
        idx = upper.find(start.upper())
        if idx >= 0:
            start_idx = idx + len(start)
            break
    if start_idx < 0:
        return ""
    stop_idx = len(text)
    for stop in stops:
        idx = upper.find(stop.upper(), start_idx)
        if idx >= 0:
            stop_idx = min(stop_idx, idx)
    return clean_multiline(text[start_idx:stop_idx])


def extract_ingredients(text: str) -> str:
    patterns = [
        r"Komposisi Lengkap\s*:?\s*(.*?)(?:Cara Penggunaan|HOW TO USE|AVERAGE USER RATING|$)",
        r"Ingredients\s*:?\s*(.*?)(?:HOW TO USE|Cara Penggunaan|AVERAGE USER RATING|$)",
        r"Full Ingredients\s*:?\s*(.*?)(?:HOW TO USE|Cara Penggunaan|AVERAGE USER RATING|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            value = clean_multiline(match.group(1))
            if "," in value or len(value.split()) >= 4:
                return value
    return ""


def infer_concerns(text: str) -> list[str]:
    lower = text.casefold()
    concerns = []
    for label, tokens in CONCERN_KEYWORDS.items():
        if any(token.casefold() in lower for token in tokens):
            concerns.append(label)
    return concerns or ["Skin Conditioning"]


def infer_skin_type(text: str) -> str:
    lower = text.casefold()
    labels = []
    if any(token in lower for token in ["dry", "kering", "dehydrated", "hidrasi", "moistur"]):
        labels.append("dry/dehydrated")
    if any(token in lower for token in ["oily", "berminyak", "acne", "jerawat", "sebum", "komedo"]):
        labels.append("oily/acne-prone")
    if any(token in lower for token in ["sensitive", "sensitif", "redness", "kemerahan", "irritation"]):
        labels.append("sensitive")
    if "all skin type" in lower or "semua jenis kulit" in lower or "all skin types" in lower:
        labels.append("all/unspecified")
    return ", ".join(unique_preserve_order(labels or ["all/unspecified"]))


def infer_pregnancy_safe(text: str) -> str:
    lower = text.casefold()
    if any(token in lower for token in ["pregnancy safe", "aman digunakan ibu hamil", "ibu hamil"]):
        return "claimed pregnancy safe on source page"
    if "pregnancy" in lower or "hamil" in lower:
        return "pregnancy mentioned; verify manually"
    return "unknown"


def infer_allergen_flag(ingredients: str, ingredient_warnings: list[str], description: str) -> str:
    lower = f"{ingredients} {description}".casefold()
    flags = []
    if any(token in lower for token in INGREDIENT_WARNING_TOKENS):
        flags.append("potential irritant/allergen token found")
    flags.extend(ingredient_warnings[:5])
    return "yes: " + "; ".join(unique_preserve_order(flags)) if flags else "none detected from scraped data"


def load_ingredient_lookup(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    lookup = {}
    for row in rows:
        name = clean_value(row.get("ingredient_name", ""))
        if not name:
            continue
        for key in ingredient_keys(name):
            lookup[key] = {clean_key(k): clean_value(v) for k, v in row.items()}
    return lookup


def ingredient_notes(
    ingredient_names: list[str],
    lookup: dict[str, dict[str, str]],
) -> tuple[list[str], list[str]]:
    functions = []
    warnings = []
    for ingredient in ingredient_names:
        row = None
        for key in ingredient_keys(ingredient):
            if key in lookup:
                row = lookup[key]
                break
        if not row:
            continue
        functions.extend([row.get("function1", ""), row.get("function2", "")])
        row_warnings = unique_preserve_order([row.get("warning1", ""), row.get("warning2", "")])
        if row_warnings:
            warnings.append(f"{ingredient}: {', '.join(row_warnings)}")
    return unique_preserve_order([item for item in functions if item]), unique_preserve_order(warnings)


def parse_ingredient_list(value: str) -> list[str]:
    if not value:
        return []
    return unique_preserve_order(
        clean_value(part)
        for part in re.split(r",|;|\n", value)
        if clean_value(part) and len(clean_value(part)) <= 80
    )


def ingredient_keys(value: str) -> list[str]:
    cleaned = clean_value(value)
    without_parentheses = re.sub(r"\s*\([^)]*\)", "", cleaned).strip()
    return unique_preserve_order([canonical(cleaned), canonical(without_parentheses)])


def canonical(value: str) -> str:
    return re.sub(r"\s+", " ", clean_value(value)).casefold()


def read_existing_products(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [{key: clean_value(value) for key, value in row.items()} for row in csv.DictReader(file)]


def merge_products(existing: list[dict[str, str]], new: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for row in [*existing, *new]:
        key = f"{row.get('brand', '').casefold()}|{row.get('product_name', '').casefold()}"
        if not key.strip("|"):
            continue
        previous = rows.get(key, {})
        merged = {**previous}
        for field in NORMALIZED_FIELDS:
            value = clean_value(row.get(field, ""))
            if value or field not in merged:
                merged[field] = value
        rows[key] = merged
    return list(rows.values())


def write_products(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=NORMALIZED_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in NORMALIZED_FIELDS})


def append_raw(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_html_fragment(value: str) -> str:
    return clean_value(re.sub(r"<[^>]+>", " ", html.unescape(value)))


def clean_multiline(value: str) -> str:
    lines = clean_lines(value.splitlines())
    return " ".join(lines)


def clean_key(value: str | None) -> str:
    return clean_value(value).lower()


def clean_value(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def first_nonempty(*values: str) -> str:
    for value in values:
        cleaned = clean_value(value)
        if cleaned:
            return cleaned
    return ""


def first_regex(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.I)
    return match.group(0) if match else ""


def format_price(value: str) -> str:
    value = clean_value(value)
    if not value:
        return ""
    if value.startswith("Rp"):
        return value
    try:
        number = int(float(value))
    except ValueError:
        return value
    return f"Rp{number:,}".replace(",", ".")


def normalize_count(value: str) -> str:
    cleaned = value.replace(",", ".").casefold()
    if cleaned.endswith("k"):
        try:
            return str(int(float(cleaned[:-1]) * 1000))
        except ValueError:
            return value
    return cleaned.replace(".", "")


def unique_preserve_order(values: Any) -> list[str]:
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


if __name__ == "__main__":
    main()
