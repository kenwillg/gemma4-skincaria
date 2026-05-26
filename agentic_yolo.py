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
DEFAULT_EFFICIENTNET_CHECKPOINT = (
    BASE_DIR
    / "runs"
    / "classify"
    / "efficientnetv2-b0-texture"
    / "best.pt"
)
DEFAULT_EFFICIENTNET_THRESHOLDS = (
    BASE_DIR
    / "runs"
    / "classify"
    / "efficientnetv2-b0-texture"
    / "thresholds.json"
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
Return JSON only. Plan visual analysis first, then product retrieval and recommendation when a usable concern or image is available.
Use this schema:
{
  "summary": "one short sentence",
  "route": "static|agentic",
  "actions": [{"tool": "PLAN|DETECT|CLASSIFY|REVIEW|ANSWER", "reason": "short reason"}],
  "known_limits": ["short limitation"]
}
Keep reasons concise. Do not reveal hidden chain-of-thought."""


REVIEW_PROMPT = """You are Skincaria's visual review module.
Use the user's concern, YOLO11n detection summary, EfficientNetV2 texture-classification summary, and attached images to produce a careful skin-observation answer.
Do not recommend products yet. Do not mention a knowledge base. Do not diagnose disease.
Important:
- YOLO detections are machine detections, not exhaustive truth.
- The first attached image, when present, is the original user image.
- The second attached image, when present, is the YOLO-annotated image.
- YOLO provides localized bounding-box evidence.
- EfficientNetV2 provides image-level texture evidence only; it does not localize boxes.
- If you visually notice a likely condition that the models did not detect, include it as "VLM cross-check: possible ...", not as a detector result.
- Separate YOLO-backed, EfficientNet-backed, and VLM-only observations in plain language.
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
        efficientnet_checkpoint: Path | None = None,
    ) -> None:
        self.ollama = ollama
        self.weights_path = Path(os.getenv("SKINCARIA_YOLO_WEIGHTS", weights_path or DEFAULT_WEIGHTS))
        self.efficientnet_checkpoint = Path(
            os.getenv(
                "SKINCARIA_EFFICIENTNET_CHECKPOINT",
                efficientnet_checkpoint or DEFAULT_EFFICIENTNET_CHECKPOINT,
            )
        )
        self.efficientnet_thresholds_path = Path(
            os.getenv("SKINCARIA_EFFICIENTNET_THRESHOLDS", DEFAULT_EFFICIENTNET_THRESHOLDS)
        )
        self.conf = float(os.getenv("SKINCARIA_YOLO_CONF", "0.12"))
        self.iou = float(os.getenv("SKINCARIA_YOLO_IOU", "0.55"))
        self.imgsz = int(os.getenv("SKINCARIA_YOLO_IMGSZ", "960"))
        self.max_det = int(os.getenv("SKINCARIA_YOLO_MAX_DET", "80"))
        self._model: Any | None = None
        self._efficientnet: EfficientNetTextureClassifier | None = None

    async def plan(self, *, concern: str, has_image: bool) -> dict[str, Any]:
        user_prompt = json.dumps(
            {
                "user_concern": concern or "",
                "image_available": has_image,
                "available_tools": [
                    "YOLO11n object detection",
                    "EfficientNetV2-B0 texture classification",
                    "Gemma visual review",
                    "product knowledge base retrieval",
                    "Gemma product recommendation",
                    "final answer",
                ],
                "excluded_tools": ["Falcon Perception"],
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

    async def assess_input(self, image_base64: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._assess_input_sync, image_base64)

    async def classify_texture(self, image_base64: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._classify_texture_sync, image_base64)

    async def review(
        self,
        *,
        concern: str,
        detection_summary: dict[str, Any] | None,
        texture_summary: dict[str, Any] | None = None,
        original_image_base64: str | None = None,
    ) -> dict[str, Any]:
        user_message: dict[str, Any] = {
            "role": "user",
            "content": json.dumps(
                {
                    "user_concern": concern or "",
                    "detector": "YOLO11n Skincaria weights, image size 960",
                    "detection_summary": compact_detection_summary(detection_summary),
                    "texture_classifier": "EfficientNetV2-B0 raw texture classifier with tuned per-class thresholds",
                    "texture_summary": compact_texture_summary(texture_summary),
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
        return fallback_visual_review(
            concern=concern,
            detection_summary=detection_summary,
            texture_summary=texture_summary,
        )

    def default_plan(self, has_image: bool) -> dict[str, Any]:
        actions = [{"tool": "PLAN", "reason": "Classify the request and choose the analysis path."}]
        if has_image:
            actions.append({"tool": "DETECT", "reason": "Run YOLO11n 960 on the face image."})
            actions.append(
                {
                    "tool": "CLASSIFY",
                    "reason": "Run EfficientNetV2-B0 for image-level texture labels.",
                }
            )
        actions.extend(
            [
                {"tool": "REVIEW", "reason": "Ask Gemma to summarize detections safely."},
                {"tool": "RETRIEVE", "reason": "Search the product knowledge base using visual evidence."},
                {"tool": "RECOMMEND", "reason": "Explain product matches with customer-specific reasons."},
                {"tool": "ANSWER", "reason": "Return observations and product recommendations."},
            ]
        )
        return {
            "summary": "Use the agentic image-analysis path, then retrieve matched products for recommendation.",
            "route": "agentic",
            "actions": actions,
            "known_limits": [
                "YOLO boxes and EfficientNet labels are experimental visual signals, not diagnoses."
            ],
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

    def _assess_input_sync(self, image_base64: str) -> dict[str, Any]:
        image = decode_base64_image(image_base64)
        return assess_input_image(image)

    def _classify_texture_sync(self, image_base64: str) -> dict[str, Any]:
        image = decode_base64_image(image_base64)
        classifier = self._load_efficientnet()
        return classifier.predict(image)

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights_path}")
        from ultralytics import YOLO

        self._model = YOLO(str(self.weights_path))
        return self._model

    def _load_efficientnet(self) -> "EfficientNetTextureClassifier":
        if self._efficientnet is not None:
            return self._efficientnet
        self._efficientnet = EfficientNetTextureClassifier(
            checkpoint_path=self.efficientnet_checkpoint,
            thresholds_path=self.efficientnet_thresholds_path,
        )
        return self._efficientnet


class EfficientNetTextureClassifier:
    def __init__(self, *, checkpoint_path: Path, thresholds_path: Path) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"EfficientNet checkpoint not found: {checkpoint_path}")
        import timm
        import torch
        from torchvision import transforms

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.config = checkpoint["config"]
        self.class_names = list(self.config["class_names"])
        self.imgsz = int(self.config.get("imgsz", 384))
        self.model = timm.create_model(
            self.config.get("model", "tf_efficientnetv2_b0"),
            pretrained=False,
            num_classes=len(self.class_names),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.skin_mask_enabled = os.getenv("SKINCARIA_EFFICIENTNET_SKIN_MASK", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.mask_background = os.getenv("SKINCARIA_EFFICIENTNET_MASK_BACKGROUND", "mean").strip().lower()
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.imgsz, self.imgsz)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.thresholds = self._load_thresholds(thresholds_path)

    def predict(self, image: Image.Image) -> dict[str, Any]:
        image = image.convert("RGB")
        model_image = image
        skin_mask = None
        preprocessing: dict[str, Any] = {
            "skin_mask_enabled": self.skin_mask_enabled,
            "input": "raw RGB image",
        }
        if self.skin_mask_enabled:
            model_image, skin_mask, skin_ratio = mask_non_skin_for_texture(
                image,
                background=self.mask_background,
            )
            preprocessing = {
                "skin_mask_enabled": True,
                "input": "likely-skin masked RGB image",
                "mask_background": self.mask_background,
                "skin_pixel_ratio": round(skin_ratio, 4),
                "note": "Non-skin pixels are replaced before EfficientNet inference; Grad-CAM is also gated to likely-skin pixels.",
            }

        tensor = self.transform(model_image).unsqueeze(0).to(self.device)
        logits, probs, activations, gradients, hook_handles = self._forward_for_gradcam(tensor)
        try:
            probs_list = probs.squeeze(0).detach().cpu().tolist()

            predictions = []
            for class_name, probability in zip(self.class_names, probs_list):
                threshold = float(self.thresholds.get(class_name, 0.5))
                predictions.append(
                    {
                        "class": class_name,
                        "probability": round(float(probability), 4),
                        "threshold": round(threshold, 4),
                        "active": float(probability) >= threshold,
                        "margin": round(float(probability) - threshold, 4),
                    }
                )
            predictions.sort(key=lambda item: item["probability"], reverse=True)
            active = [item for item in predictions if item["active"]]
            gradcam_target = active[0] if active else predictions[0]
            gradcam_index = self.class_names.index(gradcam_target["class"])
            gradcam = self._make_gradcam(
                image=image,
                logits=logits,
                target_index=gradcam_index,
                target_prediction=gradcam_target,
                activations=activations,
                gradients=gradients,
                skin_mask=skin_mask,
            )
        finally:
            for handle in hook_handles:
                handle.remove()
        return {
            "model": str(self.checkpoint_path),
            "task": "image-level texture multi-label classification",
            "imgsz": self.imgsz,
            "threshold_source": "tuned validation thresholds",
            "preprocessing": preprocessing,
            "predictions": predictions,
            "active_labels": active,
            "gradcam": gradcam,
        }

    def _forward_for_gradcam(self, tensor: Any) -> tuple[Any, Any, dict[str, Any], dict[str, Any], list[Any]]:
        activations: dict[str, Any] = {}
        gradients: dict[str, Any] = {}

        def forward_hook(_module: Any, _inputs: Any, output: Any) -> None:
            activations["value"] = output

        def backward_hook(_module: Any, _grad_input: Any, grad_output: Any) -> None:
            gradients["value"] = grad_output[0]

        forward_handle = self.model.conv_head.register_forward_hook(forward_hook)
        backward_handle = self.model.conv_head.register_full_backward_hook(backward_hook)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)
        probs = self.torch.sigmoid(logits)
        return logits, probs, activations, gradients, [forward_handle, backward_handle]

    def _make_gradcam(
        self,
        *,
        image: Image.Image,
        logits: Any,
        target_index: int,
        target_prediction: dict[str, Any],
        activations: dict[str, Any],
        gradients: dict[str, Any],
        skin_mask: Any | None,
    ) -> dict[str, Any]:
        import cv2
        import numpy as np

        self.model.zero_grad(set_to_none=True)
        logits[0, target_index].backward()
        activation = activations["value"].detach()
        gradient = gradients["value"].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = self.torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = self.torch.nn.functional.interpolate(
            cam,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().detach().cpu().numpy()
        if skin_mask is not None:
            mask = cv2.resize(
                skin_mask.astype(np.float32) / 255.0,
                (image.width, image.height),
                interpolation=cv2.INTER_LINEAR,
            )
            cam = cam * mask
        cam = cam - float(cam.min())
        max_value = float(cam.max())
        if max_value > 0:
            cam = cam / max_value

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        base = np.asarray(image).astype(np.float32)
        overlay = np.clip(base * 0.55 + heatmap.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
        hotspot_y, hotspot_x = np.unravel_index(int(np.argmax(cam)), cam.shape)
        overlay_image = Image.fromarray(overlay, mode="RGB")
        overlay_image = draw_gradcam_label(
            overlay_image,
            class_name=target_prediction["class"],
            probability=float(target_prediction["probability"]),
            threshold=float(target_prediction["threshold"]),
            active=bool(target_prediction["active"]),
            hotspot=(int(hotspot_x), int(hotspot_y)),
        )
        return {
            "class": target_prediction["class"],
            "probability": target_prediction["probability"],
            "threshold": target_prediction["threshold"],
            "active": target_prediction["active"],
            "hotspot": {"x": int(hotspot_x), "y": int(hotspot_y)},
            "overlay_image": encode_image(overlay_image),
            "method": "Grad-CAM on EfficientNetV2-B0 conv_head",
            "note": "Heatmap indicates regions that contributed to the selected image-level texture label.",
        }

    def _load_thresholds(self, thresholds_path: Path) -> dict[str, float]:
        if not thresholds_path.exists():
            return {class_name: 0.5 for class_name in self.class_names}
        payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
        thresholds = payload.get("thresholds", {})
        return {
            class_name: float(thresholds.get(class_name, 0.5))
            for class_name in self.class_names
        }


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


def draw_gradcam_label(
    image: Image.Image,
    *,
    class_name: str,
    probability: float,
    threshold: float,
    active: bool,
    hotspot: tuple[int, int],
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")
    font = ImageFont.load_default()
    status = "passed" if active else "below threshold"
    label = f"Grad-CAM: {class_name} {probability * 100:.1f}% / threshold {threshold * 100:.0f}% ({status})"
    text_box = draw.textbbox((0, 0), label, font=font)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    pad = 8
    x1 = 8
    y1 = 8
    x2 = min(annotated.width - 8, x1 + text_w + pad * 2)
    y2 = y1 + text_h + pad * 2
    fill = (31, 157, 114, 230) if active else (183, 121, 31, 230)
    draw.rectangle((x1, y1, x2, y2), fill=fill)
    draw.text((x1 + pad, y1 + pad), label, fill=(255, 255, 255, 255), font=font)

    hx, hy = hotspot
    radius = max(8, round(min(annotated.size) / 45))
    draw.ellipse((hx - radius, hy - radius, hx + radius, hy + radius), outline=(255, 255, 255, 240), width=3)
    draw.ellipse((hx - 3, hy - 3, hx + 3, hy + 3), fill=(255, 255, 255, 240))
    hotspot_label = "highest activation"
    hotspot_box = draw.textbbox((0, 0), hotspot_label, font=font)
    label_w = hotspot_box[2] - hotspot_box[0]
    label_h = hotspot_box[3] - hotspot_box[1]
    lx = min(max(8, hx + radius + 6), max(8, annotated.width - label_w - pad * 2 - 8))
    ly = min(max(8, hy - label_h // 2 - pad), max(8, annotated.height - label_h - pad * 2 - 8))
    draw.rectangle((lx, ly, lx + label_w + pad * 2, ly + label_h + pad * 2), fill=(17, 24, 39, 210))
    draw.text((lx + pad, ly + pad), hotspot_label, fill=(255, 255, 255, 255), font=font)
    return annotated


def assess_input_image(image: Image.Image) -> dict[str, Any]:
    import cv2
    import numpy as np

    rgb = np.asarray(image.convert("RGB"))
    skin_mask = likely_skin_mask(image)
    skin_ratio = float((skin_mask > 0).mean())

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
    face_bbox = None
    face_area_ratio = 0.0
    if len(faces) > 0:
        x, y, width, height = max(faces, key=lambda item: item[2] * item[3])
        face_bbox = [int(x), int(y), int(width), int(height)]
        face_area_ratio = float((width * height) / max(image.width * image.height, 1))

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    warnings = []
    if face_bbox is None:
        warnings.append("No frontal face was detected; results may be unreliable.")
    if skin_ratio < 0.12:
        warnings.append("Only a small likely-skin area was found; upload a clearer face/skin photo.")
    if blur_score < 35.0:
        warnings.append("The image appears blurry; retaking the photo may improve analysis.")

    acceptable = bool(face_bbox is not None or skin_ratio >= 0.18)
    if not acceptable:
        warnings.append("Input quality is below the recommended level for skin analysis.")

    mask_overlay = rgb.astype(np.float32).copy()
    green = np.zeros_like(mask_overlay)
    green[:, :, 1] = 255
    alpha = (skin_mask.astype(np.float32) / 255.0)[:, :, None] * 0.35
    mask_overlay = np.clip(mask_overlay * (1.0 - alpha) + green * alpha, 0, 255).astype(np.uint8)

    return {
        "image_size": {"width": image.width, "height": image.height},
        "face_detected": face_bbox is not None,
        "face_bbox": face_bbox,
        "face_area_ratio": round(face_area_ratio, 4),
        "skin_pixel_ratio": round(skin_ratio, 4),
        "blur_score": round(blur_score, 2),
        "acceptable": acceptable,
        "warnings": warnings,
        "skin_mask_overlay": encode_image(Image.fromarray(mask_overlay, mode="RGB")),
        "method": "OpenCV Haar face check + HSL likely-skin mask",
    }


def likely_skin_mask(image: Image.Image) -> Any:
    import cv2
    import numpy as np

    rgb = np.asarray(image.convert("RGB"))
    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
    h = hls[:, :, 0].astype(np.float32)
    l = hls[:, :, 1].astype(np.float32)
    s = hls[:, :, 2].astype(np.float32)
    ls_ratio = l / np.maximum(s, 1.0)
    skin_mask = (
        (s >= 35.0)
        & (ls_ratio > 0.45)
        & (ls_ratio < 3.2)
        & ((h <= 18.0) | (h >= 160.0))
    ).astype(np.uint8) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    return skin_mask


def mask_non_skin_for_texture(
    image: Image.Image,
    *,
    background: str,
) -> tuple[Image.Image, Any, float]:
    import cv2
    import numpy as np

    rgb = np.asarray(image.convert("RGB")).copy()
    skin_mask = likely_skin_mask(image)
    skin_ratio = float((skin_mask > 0).mean())
    alpha = cv2.GaussianBlur(skin_mask, (9, 9), 0).astype(np.float32) / 255.0
    alpha = alpha[:, :, None]

    if background == "black":
        bg = np.zeros_like(rgb, dtype=np.float32)
    else:
        bg = np.empty_like(rgb, dtype=np.float32)
        bg[:, :, 0] = 123.675
        bg[:, :, 1] = 116.28
        bg[:, :, 2] = 103.53

    masked = rgb.astype(np.float32) * alpha + bg * (1.0 - alpha)
    return Image.fromarray(np.clip(masked, 0, 255).astype(np.uint8), mode="RGB"), skin_mask, skin_ratio


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


def compact_texture_summary(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {}
    return {
        "task": summary.get("task"),
        "threshold_source": summary.get("threshold_source"),
        "preprocessing": summary.get("preprocessing"),
        "active_labels": summary.get("active_labels", []),
        "predictions": (summary.get("predictions") or [])[:10],
    }


def fallback_visual_review(
    *,
    concern: str,
    detection_summary: dict[str, Any] | None,
    texture_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    counts = (detection_summary or {}).get("counts") or {}
    detections = (detection_summary or {}).get("detections") or []
    active_labels = (texture_summary or {}).get("active_labels") or []

    observations = []
    if counts:
        count_text = ", ".join(
            f"{name}: {count}"
            for name, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
        )
        observations.append(f"YOLO-backed localized detections: {count_text}.")
    top_detections = [
        f"{item.get('class')} ({float(item.get('confidence', 0.0)) * 100:.1f}%)"
        for item in detections[:5]
        if item.get("class")
    ]
    if top_detections:
        observations.append(f"Highest-confidence visible findings: {', '.join(top_detections)}.")
    if active_labels:
        texture_text = ", ".join(
            f"{item.get('class')} ({float(item.get('probability', 0.0)) * 100:.1f}%)"
            for item in active_labels
            if item.get("class")
        )
        if texture_text:
            observations.append(f"EfficientNet image-level texture evidence: {texture_text}.")
    if concern:
        observations.append(f"User concern included: {concern}.")
    if not observations:
        observations.append("No strong model-backed skin condition signal was available from the current input.")

    return {
        "observations": observations,
        "uncertainties": [
            "Gemma visual review did not return valid JSON, so this summary was built from structured model outputs.",
            "YOLO and EfficientNet outputs are experimental support signals, not medical diagnoses.",
        ],
        "final_answer": "Visual analysis completed using structured YOLO and EfficientNet outputs. Product retrieval can continue from these model-backed observations.",
    }
