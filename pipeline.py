import asyncio
import base64
import io
import json
import re
from typing import Any, AsyncIterator

from PIL import Image

from kb_builder import SkincareKB
from ollama_client import OllamaClient


SKIN_LABEL_SCHEMA = {
    "skin_type": ["oily", "dry", "combination", "normal"],
    "acne_severity": ["none", "mild", "moderate", "severe"],
    "redness": ["none", "mild", "moderate", "severe"],
    "pore_condition": ["normal", "enlarged", "clogged"],
    "dehydration": ["none", "mild", "severe"],
}


IMAGE_ANALYSIS_PROMPT = """Analyze the visible facial skin condition from the image. Return JSON only with exactly this schema:
{
"skin_type": "oily|dry|combination|normal",
"acne_severity": "none|mild|moderate|severe",
"redness": "none|mild|moderate|severe",
"pore_condition": "normal|enlarged|clogged",
"dehydration": "none|mild|severe"
}
Do not include any explanation. Return JSON only."""


TEXT_ANALYSIS_PROMPT = """Infer skincare condition labels from the user's text description. Return JSON only with exactly this schema:
{
"skin_type": "oily|dry|combination|normal",
"acne_severity": "none|mild|moderate|severe",
"redness": "none|mild|moderate|severe",
"pore_condition": "normal|enlarged|clogged",
"dehydration": "none|mild|severe"
}
Do not include any explanation. Return JSON only."""


RECOMMENDATION_PROMPT = """Kamu adalah asisten rekomendasi skincare Indonesia yang ahli. Berdasarkan kondisi kulit pengguna dan produk yang tersedia, berikan rekomendasi yang jelas dan bermanfaat dalam Bahasa Indonesia. Sertakan alasan berbasis bahan aktif (INCI) untuk setiap produk yang direkomendasikan. Jika ada produk yang mengandung alergen atau tidak aman untuk ibu hamil, sebutkan dengan jelas."""


class ManualDescriptionRequired(Exception):
    pass


class SkincariaPipeline:
    def __init__(self, ollama: OllamaClient | None = None, kb: SkincareKB | None = None):
        self.ollama = ollama or OllamaClient()
        self.kb = kb or SkincareKB()

    async def run_analysis(self, *, image_base64: str | None, concern: str) -> dict[str, Any]:
        concern = (concern or "").strip()
        if not image_base64 and not concern:
            raise ManualDescriptionRequired("Mohon ambil foto wajah atau ceritakan kondisi kulitmu.")

        source = "text"
        if image_base64:
            try:
                skin_labels = await self._analyze_image(image_base64)
                source = "image"
            except ManualDescriptionRequired:
                if not concern:
                    raise
                skin_labels = await self._analyze_text(concern)
                source = "text_fallback"
        else:
            skin_labels = await self._analyze_text(concern)

        query = self.build_query(skin_labels, concern)
        products = await asyncio.to_thread(self.kb.query, query, 5)
        products_context = self.kb.format_context(products)

        return {
            "manual_input_required": False,
            "analysis_source": source,
            "skin_labels": skin_labels,
            "query": query,
            "products": self._preview_products(products),
            "products_context": products_context,
        }

    def build_query(self, skin_labels: dict[str, str], concern: str) -> str:
        return f"Skin condition: {json.dumps(skin_labels, ensure_ascii=False)}. User concern: {concern}"

    async def stream_recommendation(
        self,
        *,
        skin_labels: dict[str, str],
        concern: str,
        products_context: str,
    ) -> AsyncIterator[str]:
        user_prompt = "\n\n".join(
            [
                f"Kondisi kulit pengguna:\n{json.dumps(skin_labels, ensure_ascii=False, indent=2)}",
                f"Keluhan pengguna:\n{concern or 'Tidak ada keluhan tambahan.'}",
                f"Produk hasil retrieval:\n{products_context}",
                "Tulis rekomendasi dalam format Markdown ringkas: ringkasan kondisi, 3-5 produk rekomendasi, alasan bahan aktif, peringatan alergi/kehamilan bila ada, dan cara pakai singkat.",
            ]
        )
        messages = [
            {"role": "system", "content": RECOMMENDATION_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        async for token in self.ollama.stream_chat(messages, temperature=0.35):
            yield token

    async def _analyze_image(self, image_base64: str) -> dict[str, str]:
        user_content = "Analyze this face image for visible skincare condition labels."
        attempts = [
            IMAGE_ANALYSIS_PROMPT,
            IMAGE_ANALYSIS_PROMPT
            + "\nSTRICT MODE: Return a single valid JSON object only. No markdown fences. No prose. Use only allowed enum values.",
        ]

        for prompt in attempts:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content, "images": [strip_data_url(image_base64)]},
            ]
            raw = await self.ollama.chat(messages, json_mode=True, temperature=0.1)
            labels = parse_skin_labels(raw)
            if labels:
                return labels

        raise ManualDescriptionRequired(
            "Analisis foto belum berhasil. Mohon ceritakan kondisi kulitmu secara manual."
        )

    async def _analyze_text(self, concern: str) -> dict[str, str]:
        if not concern.strip():
            raise ManualDescriptionRequired("Mohon ceritakan kondisi kulitmu secara manual.")

        attempts = [
            TEXT_ANALYSIS_PROMPT,
            TEXT_ANALYSIS_PROMPT
            + "\nSTRICT MODE: Return a single valid JSON object only. No markdown fences. No prose. Use only allowed enum values.",
        ]
        for prompt in attempts:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": concern},
            ]
            raw = await self.ollama.chat(messages, json_mode=True, temperature=0.1)
            labels = parse_skin_labels(raw)
            if labels:
                return labels

        return heuristic_labels_from_text(concern)

    def _preview_products(self, products: list[dict[str, Any]]) -> list[dict[str, Any]]:
        previews = []
        for product in products:
            meta = product.get("metadata", {})
            previews.append(
                {
                    "product_name": meta.get("product_name", ""),
                    "brand": meta.get("brand", ""),
                    "category": meta.get("category", ""),
                    "price": meta.get("price", ""),
                    "skin_type": meta.get("skin_type", ""),
                    "concerns": meta.get("concerns", ""),
                    "allergen_flag": meta.get("allergen_flag", ""),
                    "pregnancy_safe": meta.get("pregnancy_safe", ""),
                    "similarity": product.get("similarity"),
                }
            )
        return previews


def parse_skin_labels(raw: str) -> dict[str, str] | None:
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

    if not isinstance(payload, dict):
        return None

    labels: dict[str, str] = {}
    for key, allowed_values in SKIN_LABEL_SCHEMA.items():
        value = str(payload.get(key, "")).strip().lower()
        if value not in allowed_values:
            return None
        labels[key] = value
    return labels


def heuristic_labels_from_text(text: str) -> dict[str, str]:
    lowered = text.lower()
    labels = {
        "skin_type": "normal",
        "acne_severity": "none",
        "redness": "none",
        "pore_condition": "normal",
        "dehydration": "none",
    }

    if any(word in lowered for word in ["berminyak", "oily", "kilang minyak"]):
        labels["skin_type"] = "oily"
    elif any(word in lowered for word in ["kering", "dry", "mengelupas"]):
        labels["skin_type"] = "dry"
    elif any(word in lowered for word in ["kombinasi", "combination", "t-zone", "t zone"]):
        labels["skin_type"] = "combination"

    if any(word in lowered for word in ["jerawat parah", "meradang banyak", "severe"]):
        labels["acne_severity"] = "severe"
    elif any(word in lowered for word in ["jerawat", "bruntusan", "komedo", "acne"]):
        labels["acne_severity"] = "moderate" if "banyak" in lowered else "mild"

    if any(word in lowered for word in ["kemerahan", "redness", "iritasi", "merah"]):
        labels["redness"] = "moderate" if "parah" in lowered else "mild"

    if any(word in lowered for word in ["pori", "komedo", "clogged", "tersumbat"]):
        labels["pore_condition"] = "clogged" if "komedo" in lowered or "tersumbat" in lowered else "enlarged"

    if any(word in lowered for word in ["dehidrasi", "ketarik", "kusam", "dehydrated"]):
        labels["dehydration"] = "severe" if "parah" in lowered else "mild"

    return labels


def strip_data_url(value: str) -> str:
    if "," in value and value.strip().startswith("data:"):
        return value.split(",", 1)[1]
    return value.strip()


def image_bytes_to_base64(raw: bytes) -> str:
    with Image.open(io.BytesIO(raw)) as image:
        image = image.convert("RGB")
        image.thumbnail((1024, 1024))
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=88, optimize=True)
    return base64.b64encode(output.getvalue()).decode("ascii")
