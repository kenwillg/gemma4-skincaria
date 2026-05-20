import asyncio
import base64
import binascii
import io
import json
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from ollama_client import OllamaClient


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = (
    BASE_DIR
    / "runs"
    / "detect"
    / "skincaria-detect"
    / "yolo11n-img960"
    / "weights"
    / "best.pt"
)

PALETTE = {
    "Acne": "#ef4444",
    "Blackheads": "#111827",
    "Dark-Spots": "#92400e",
    "Dry-Skin": "#f59e0b",
    "Enlarged-Pores": "#0ea5e9",
    "Eyebags": "#8b5cf6",
    "Oily-Skin": "#22c55e",
    "Skin-Redness": "#ec4899",
    "Whiteheads": "#e5e7eb",
    "Wrinkles": "#64748b",
}


PLANNER_PROMPT = """You are the Skincaria agent planner.
Return JSON only. Do not recommend products. Do not use a knowledge base.
Use this schema:
{
  "summary": "one short sentence",
  "route": "static|agentic",
  "actions": [{"tool": "PLAN|DETECT|REVIEW|ANSWER", "reason": "short reason"}],
  "known_limits": ["short limitation"]
}
Keep reasons concise. Do not reveal hidden chain-of-thought."""


REVIEW_PROMPT = """You are Skincaria's visual review module.
Use the user's concern, YOLO11n detection summary, and attached images to produce a careful skin-observation answer.
Do not recommend products yet. Do not mention a knowledge base. Do not diagnose disease.
Important:
- YOLO detections are machine detections, not exhaustive truth.
- The first attached image, when present, is the original user image.
- The second attached image, when present, is the YOLO-annotated image.
- If you visually notice a likely condition that YOLO did not detect, include it as "VLM cross-check: possible ...", not as a YOLO detection.
- Separate detector-backed observations from VLM-only observations in plain language.
Return JSON only:
{
  "observations": ["short visible observation"],
  "uncertainties": ["short uncertainty or caveat"],
  "final_answer": "concise answer in Indonesian or English matching the user"
}
Prefer calibrated language: "terlihat", "terdeteksi", "kemungkinan", "perlu dicek ulang"."""


class AgenticYoloPipeline:
    def __init__(
        self,
        *,
        ollama: OllamaClient,
        weights_path: Path | None = None,
    ) -> None:
        self.ollama = ollama
        self.weights_path = Path(os.getenv("SKINCARIA_YOLO_WEIGHTS", weights_path or DEFAULT_WEIGHTS))
        self.conf = float(os.getenv("SKINCARIA_YOLO_CONF", "0.12"))
        self.iou = float(os.getenv("SKINCARIA_YOLO_IOU", "0.55"))
        self.imgsz = int(os.getenv("SKINCARIA_YOLO_IMGSZ", "960"))
        self.max_det = int(os.getenv("SKINCARIA_YOLO_MAX_DET", "80"))
        self._model: Any | None = None

    async def plan(self, *, concern: str, has_image: bool) -> dict[str, Any]:
        user_prompt = json.dumps(
            {
                "user_concern": concern or "",
                "image_available": has_image,
                "available_tools": ["YOLO11n object detection", "Gemma visual review", "final answer"],
                "excluded_tools": ["Falcon Perception", "product knowledge base"],
            },
            ensure_ascii=False,
        )
        raw = await self.ollama.chat(
            [
                {"role": "system", "content": PLANNER_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            json_mode=True,
            temperature=0.1,
        )
        return parse_json_object(raw) or self.default_plan(has_image)

    async def detect(self, image_base64: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._detect_sync, image_base64)

    async def review(
        self,
        *,
        concern: str,
        detection_summary: dict[str, Any] | None,
        original_image_base64: str | None = None,
    ) -> dict[str, Any]:
        user_message: dict[str, Any] = {
            "role": "user",
            "content": json.dumps(
                {
                    "user_concern": concern or "",
                    "detector": "YOLO11n Skincaria weights, image size 960",
                    "detection_summary": compact_detection_summary(detection_summary),
                    "image_note": (
                        "Attached images are ordered as: original user image first, "
                        "YOLO-annotated frame second. Use the original image for visual cross-checks."
                    ),
                },
                ensure_ascii=False,
            ),
        }
        images: list[str] = []
        if original_image_base64:
            images.append(strip_data_url(original_image_base64))
        annotated = detection_summary.get("annotated_image") if detection_summary else None
        if annotated:
            images.append(annotated)
        if images:
            user_message["images"] = images

        raw = await self.ollama.chat(
            [
                {"role": "system", "content": REVIEW_PROMPT},
                user_message,
            ],
            json_mode=True,
            temperature=0.15,
        )
        parsed = parse_json_object(raw)
        if parsed:
            return parsed
        return {
            "observations": [],
            "uncertainties": ["Gemma did not return valid JSON, so this fallback is conservative."],
            "final_answer": "YOLO detection finished, but Gemma's review response could not be parsed. Please inspect the detections and try again.",
        }

    def default_plan(self, has_image: bool) -> dict[str, Any]:
        actions = [{"tool": "PLAN", "reason": "Classify the request and choose the analysis path."}]
        if has_image:
            actions.append({"tool": "DETECT", "reason": "Run YOLO11n 960 on the face image."})
        actions.extend(
            [
                {"tool": "REVIEW", "reason": "Ask Gemma to summarize detections safely."},
                {"tool": "ANSWER", "reason": "Return observations without product retrieval."},
            ]
        )
        return {
            "summary": "Use the agentic image-analysis path without Falcon or knowledge-base retrieval.",
            "route": "agentic",
            "actions": actions,
            "known_limits": ["YOLO labels are experimental and should be treated as visual signals."],
        }

    def _detect_sync(self, image_base64: str) -> dict[str, Any]:
        image = decode_base64_image(image_base64)
        model = self._load_model()
        result = model.predict(
            image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            verbose=False,
        )[0]

        names = result.names
        detections: list[dict[str, Any]] = []
        boxes = result.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            classes = boxes.cls.cpu().tolist()
            for index, (box, score, cls_id) in enumerate(zip(xyxy, confs, classes), start=1):
                class_name = str(names.get(int(cls_id), int(cls_id)))
                x1, y1, x2, y2 = [float(value) for value in box]
                detections.append(
                    {
                        "id": index,
                        "class": class_name,
                        "confidence": round(float(score), 4),
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "area_px": int(max(0.0, x2 - x1) * max(0.0, y2 - y1)),
                    }
                )

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        counts: dict[str, int] = {}
        for item in detections:
            counts[item["class"]] = counts.get(item["class"], 0) + 1

        return {
            "model": str(self.weights_path),
            "image_size": {"width": image.width, "height": image.height},
            "confidence_threshold": self.conf,
            "iou_threshold": self.iou,
            "imgsz": self.imgsz,
            "detections": detections,
            "counts": counts,
            "top_classes": sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5],
            "annotated_image": encode_image(draw_detections(image, detections)),
        }

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights_path}")
        from ultralytics import YOLO

        self._model = YOLO(str(self.weights_path))
        return self._model


def decode_base64_image(value: str) -> Image.Image:
    value = strip_data_url(value)
    try:
        raw = base64.b64decode(value, validate=False)
    except binascii.Error as exc:
        raise ValueError("Invalid image_base64 payload.") from exc
    with Image.open(io.BytesIO(raw)) as image:
        return image.convert("RGB")


def strip_data_url(value: str) -> str:
    if "," in value and value.strip().startswith("data:"):
        return value.split(",", 1)[1]
    return value


def encode_image(image: Image.Image) -> str:
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=88, optimize=True)
    return base64.b64encode(output.getvalue()).decode("ascii")


def draw_detections(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")
    font = ImageFont.load_default()
    line_width = max(2, round(min(image.size) / 260))

    for detection in detections[:50]:
        class_name = detection["class"]
        color = PALETTE.get(class_name, "#38bdf8")
        rgb = hex_to_rgb(color)
        x1, y1, x2, y2 = detection["bbox"]
        draw.rectangle((x1, y1, x2, y2), outline=rgb + (230,), width=line_width)
        draw.rectangle((x1, y1, x2, y2), fill=rgb + (28,))
        label = f"{class_name} {detection['confidence']:.2f}"
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        label_y = max(0, y1 - text_h - 6)
        draw.rectangle((x1, label_y, x1 + text_w + 8, label_y + text_h + 6), fill=rgb + (220,))
        text_fill = (17, 24, 39, 255) if class_name == "Whiteheads" else (255, 255, 255, 255)
        draw.text((x1 + 4, label_y + 3), label, fill=text_fill, font=font)

    return annotated


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def parse_json_object(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def compact_detection_summary(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {}
    return {
        "image_size": summary.get("image_size"),
        "confidence_threshold": summary.get("confidence_threshold"),
        "counts": summary.get("counts", {}),
        "top_classes": summary.get("top_classes", []),
        "detections": (summary.get("detections") or [])[:20],
    }
