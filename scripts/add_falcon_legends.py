import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = [
    (239, 68, 68),
    (14, 165, 233),
    (34, 197, 94),
    (245, 158, 11),
    (168, 85, 247),
    (236, 72, 153),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add report legends to Falcon overlay images using a _summary.json file."
    )
    parser.add_argument("--summary", type=Path, required=True, help="Path to Falcon _summary.json.")
    parser.add_argument(
        "--suffix",
        default="_legend",
        help="Suffix added before the image extension for legend outputs.",
    )
    parser.add_argument("--max-rows", type=int, default=18, help="Maximum legend rows per image.")
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def legend_rows(prompt: str, predictions: list[dict], max_rows: int) -> list[tuple[int | None, str]]:
    rows: list[tuple[int | None, str]] = []
    for prediction in predictions[:max_rows]:
        index = int(prediction.get("index", len(rows) + 1))
        area = prediction.get("mask_area_pixels", 0)
        rows.append((index, f"{index}: {prompt} ({area:,} px)"))

    remaining = len(predictions) - max_rows
    if remaining > 0:
        rows.append((None, f"+{remaining} more {prompt} predictions"))
    if not rows:
        rows.append((None, f"No {prompt} predictions"))
    return rows


def add_legend(
    overlay_path: Path,
    output_path: Path,
    *,
    prompt: str,
    predictions: list[dict],
    max_rows: int,
) -> None:
    image = Image.open(overlay_path).convert("RGB")
    title_font = load_font(18)
    body_font = load_font(14)
    small_font = load_font(12)

    scratch = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(scratch)
    rows = legend_rows(prompt, predictions, max_rows)

    title = f"Falcon prompt: {prompt}"
    subtitle = "Color = prediction instance"
    row_texts = [row[1] for row in rows]
    text_widths = [
        text_size(draw, title, title_font)[0],
        text_size(draw, subtitle, small_font)[0],
        *[text_size(draw, text, body_font)[0] for text in row_texts],
    ]
    panel_width = min(max(max(text_widths) + 62, 260), 420)
    row_height = 24
    panel_height = max(image.height, 76 + row_height * len(rows))

    canvas = Image.new("RGB", (image.width + panel_width, panel_height), (248, 250, 252))
    canvas.paste(image, (0, 0))

    legend = ImageDraw.Draw(canvas)
    panel_x = image.width
    legend.rectangle((panel_x, 0, canvas.width, panel_height), fill=(248, 250, 252))
    legend.line((panel_x, 0, panel_x, panel_height), fill=(203, 213, 225), width=1)

    x = panel_x + 18
    y = 18
    legend.text((x, y), title, fill=(15, 23, 42), font=title_font)
    y += 26
    legend.text((x, y), subtitle, fill=(71, 85, 105), font=small_font)
    y += 30

    for index, text in rows:
        if index is None:
            color = (100, 116, 139)
        else:
            color = COLORS[(index - 1) % len(COLORS)]
        legend.rounded_rectangle((x, y + 3, x + 16, y + 19), radius=3, fill=color)
        legend.text((x + 28, y), text, fill=(15, 23, 42), font=body_font)
        y += row_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def iter_summary_images(summary: dict, summary_path: Path) -> list[dict]:
    if "images" in summary:
        return summary["images"]

    image_path = Path(summary.get("image", ""))
    prompt = summary.get("prompt", "prompt")
    overlay = summary_path.with_name(f"{image_path.stem}_{prompt.replace(' ', '-')}_overlay.jpg")
    return [
        {
            "image": str(image_path),
            "overlay": str(overlay),
            "prediction_count": summary.get("prediction_count", 0),
            "predictions": summary.get("predictions", []),
        }
    ]


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    prompt = summary.get("prompt", "unknown")
    converted = 0

    for item in iter_summary_images(summary, args.summary):
        overlay_path = Path(item["overlay"])
        if not overlay_path.exists():
            print(f"missing overlay: {overlay_path}")
            continue

        output_path = overlay_path.with_name(f"{overlay_path.stem}{args.suffix}{overlay_path.suffix}")
        add_legend(
            overlay_path,
            output_path,
            prompt=prompt,
            predictions=item.get("predictions", []),
            max_rows=args.max_rows,
        )
        converted += 1
        print(f"{overlay_path} -> {output_path}")

    print(f"converted: {converted}")


if __name__ == "__main__":
    main()
