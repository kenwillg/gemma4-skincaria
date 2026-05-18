import argparse
import csv
import hashlib
import json
import re
import time
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx


DEFAULT_SUBREDDIT = "SkincareAddiction"
DEFAULT_TAGS = [
    "Product Request",
    "Product Question",
    "B&A",
    "Before&After",
    "Before & After",
    "Routine Help",
    "Routine help",
    "routine help",
    "Acne",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class ImagePost:
    post_id: str
    title: str
    flair: str
    created_utc: int
    permalink: str
    source_url: str
    image_url: str
    image_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download image posts from a subreddit for dataset curation."
    )
    parser.add_argument("--subreddit", default=DEFAULT_SUBREDDIT)
    parser.add_argument("--out-dir", default="data/reddit_skincare_images")
    parser.add_argument("--tags", nargs="*", default=DEFAULT_TAGS)
    parser.add_argument(
        "--listing",
        choices=["new", "hot", "top"],
        default="new",
        help="Subreddit listing to crawl before local filtering.",
    )
    parser.add_argument(
        "--time",
        choices=["hour", "day", "week", "month", "year", "all"],
        default="all",
        help="Only used with --listing top.",
    )
    parser.add_argument("--max-posts", type=int, default=500)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=1.2)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-nsfw",
        action="store_true",
        help="By default over_18 posts are skipped.",
    )
    parser.add_argument(
        "--user-agent",
        default="skincaria-academic-dataset-script/0.1",
        help="Use a descriptive User-Agent for Reddit requests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        image_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = out_dir / "metadata.csv"
    metadata_jsonl = out_dir / "metadata.jsonl"
    seen_posts = load_seen_posts(metadata_csv)

    headers = {"User-Agent": args.user_agent}
    stats = {"seen": 0, "matched_posts": 0, "downloaded_images": 0, "skipped": 0}

    with httpx.Client(headers=headers, follow_redirects=True, timeout=args.timeout) as client:
        rows: list[dict[str, Any]] = []
        try:
            posts = iter_posts(client, args)
            for post in posts:
                process_post(post, args, client, image_dir, rows, seen_posts, stats)
        except httpx.HTTPError as exc:
            raise SystemExit(
                "Reddit request failed. Check your connection, firewall/VPN, or try a larger "
                f"--timeout value. Detail: {exc}"
            ) from exc

        if rows and not args.dry_run:
            append_metadata(metadata_csv, metadata_jsonl, rows)

    print(json.dumps(stats, indent=2))
    if args.dry_run:
        print("Dry run only. No files were downloaded.")
    else:
        print(f"Images: {image_dir}")
        print(f"Metadata CSV: {metadata_csv}")
        print(f"Metadata JSONL: {metadata_jsonl}")


def process_post(
    post: dict[str, Any],
    args: argparse.Namespace,
    client: httpx.Client,
    image_dir: Path,
    rows: list[dict[str, Any]],
    seen_posts: set[str],
    stats: dict[str, int],
) -> None:
    data = post.get("data", {})
    stats["seen"] += 1
    post_id = str(data.get("id") or "")
    if not post_id or post_id in seen_posts:
        return
    if data.get("over_18") and not args.include_nsfw:
        stats["skipped"] += 1
        return
    if not matches_tags(data, args.tags):
        return

    image_urls = extract_image_urls(data)
    if not image_urls:
        return

    stats["matched_posts"] += 1
    for index, image_url in enumerate(image_urls):
        image_post = ImagePost(
            post_id=post_id,
            title=str(data.get("title") or ""),
            flair=str(data.get("link_flair_text") or ""),
            created_utc=int(data.get("created_utc") or 0),
            permalink="https://www.reddit.com" + str(data.get("permalink") or ""),
            source_url=str(data.get("url_overridden_by_dest") or data.get("url") or ""),
            image_url=image_url,
            image_index=index,
        )
        row = image_post_to_row(image_post)
        if args.dry_run:
            print(json.dumps(row, ensure_ascii=False))
            rows.append(row)
            continue

        path = download_image(client, image_url, image_dir, post_id, index)
        if path is None:
            stats["skipped"] += 1
            continue
        row["local_path"] = str(path)
        rows.append(row)
        stats["downloaded_images"] += 1
        time.sleep(args.sleep)

    seen_posts.add(post_id)


def iter_posts(client: httpx.Client, args: argparse.Namespace):
    after = None
    yielded = 0
    while yielded < args.max_posts:
        params: dict[str, Any] = {"limit": min(args.limit, args.max_posts - yielded)}
        if after:
            params["after"] = after
        if args.listing == "top":
            params["t"] = args.time

        url = f"https://www.reddit.com/r/{args.subreddit}/{args.listing}.json"
        response = client.get(url, params=params)
        if response.status_code == 429:
            time.sleep(10)
            continue
        response.raise_for_status()
        payload = response.json()
        listing = payload.get("data", {})
        children = listing.get("children", [])
        if not children:
            return

        for child in children:
            yielded += 1
            yield child
            if yielded >= args.max_posts:
                return

        after = listing.get("after")
        if not after:
            return
        time.sleep(args.sleep)


def matches_tags(post: dict[str, Any], tags: list[str]) -> bool:
    title = normalize(str(post.get("title") or ""))
    flair = normalize(str(post.get("link_flair_text") or ""))
    for tag in tags:
        normalized = normalize(tag)
        if not normalized:
            continue
        bracketed = f"[{normalized}]"
        if bracketed in title or normalized == flair or normalized in flair:
            return True
    return False


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def extract_image_urls(post: dict[str, Any]) -> list[str]:
    urls: list[str] = []

    if post.get("is_gallery") and isinstance(post.get("media_metadata"), dict):
        metadata = post["media_metadata"]
        gallery_items = post.get("gallery_data", {}).get("items", [])
        media_ids = [item.get("media_id") for item in gallery_items if item.get("media_id")]
        if not media_ids:
            media_ids = list(metadata.keys())
        for media_id in media_ids:
            item = metadata.get(media_id) or {}
            source = item.get("s") or {}
            url = source.get("u") or source.get("gif")
            if url:
                urls.append(clean_url(url))

    direct_url = str(post.get("url_overridden_by_dest") or post.get("url") or "")
    if is_direct_image_url(direct_url):
        urls.append(clean_url(direct_url))

    preview = post.get("preview", {})
    images = preview.get("images", []) if isinstance(preview, dict) else []
    if images:
        source = images[0].get("source", {})
        preview_url = source.get("url")
        if preview_url:
            urls.append(clean_url(preview_url))

    return dedupe_preserve_order([url for url in urls if is_probable_image_url(url)])


def clean_url(url: str) -> str:
    return unescape(url).replace("&amp;", "&")


def is_direct_image_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return Path(path).suffix in IMAGE_EXTENSIONS


def is_probable_image_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if is_direct_image_url(url):
        return True
    return "preview.redd.it" in parsed.netloc or "i.redd.it" in parsed.netloc


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def download_image(
    client: httpx.Client,
    url: str,
    image_dir: Path,
    post_id: str,
    index: int,
) -> Path | None:
    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"skip download {url}: {exc}")
        return None

    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
    extension = extension_from_content_type(content_type) or Path(urlparse(url).path).suffix
    if extension.lower() not in IMAGE_EXTENSIONS:
        print(f"skip non-image {url}: {content_type}")
        return None

    digest = hashlib.sha256(response.content).hexdigest()[:12]
    path = image_dir / f"{post_id}_{index}_{digest}{extension.lower()}"
    path.write_bytes(response.content)
    return path


def extension_from_content_type(content_type: str) -> str | None:
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(content_type)


def image_post_to_row(post: ImagePost) -> dict[str, Any]:
    return {
        "post_id": post.post_id,
        "title": post.title,
        "flair": post.flair,
        "created_utc": post.created_utc,
        "permalink": post.permalink,
        "source_url": post.source_url,
        "image_url": post.image_url,
        "image_index": post.image_index,
        "local_path": "",
        "label_acne_severity": "",
        "label_lesion_type": "",
        "label_redness": "",
        "label_main_area": "",
        "label_notes": "",
    }


def append_metadata(csv_path: Path, jsonl_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys())
    csv_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
        writer.writerows(rows)

    with jsonl_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_seen_posts(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return {row.get("post_id", "") for row in csv.DictReader(handle) if row.get("post_id")}


if __name__ == "__main__":
    main()
