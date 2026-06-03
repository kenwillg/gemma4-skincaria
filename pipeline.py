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


RECOMMENDATION_PROMPT = """Kamu adalah asisten rekomendasi skincare Indonesia yang hangat, teliti, dan personal. Gunakan kondisi kulit, keluhan pengguna, dan produk yang tersedia untuk membuat rekomendasi yang terasa spesifik untuk orang tersebut, bukan saran generik. Prioritaskan kebutuhan yang paling mengganggu pengguna, jelaskan alasan berbasis bahan aktif (INCI), dan beri peringatan alergen/kehamilan bila ada. Untuk setiap produk, wajib jelaskan "kenapa cocok buat kamu" dari sudut pandang customer: hubungkan sinyal kulit/keluhan pengguna dengan klaim, kategori, dan bahan produk. Jangan mengklaim diagnosis medis."""


class ManualDescriptionRequired(Exception):
    pass


class SkincariaPipeline:
    def __init__(
        self,
        ollama: OllamaClient | None = None,
        kb: SkincareKB | None = None,
    ):
        self.ollama = ollama or OllamaClient()
        self.kb = kb or SkincareKB()

    async def run_analysis(
        self,
        *,
        image_base64: str | None,
        concern: str,
    ) -> dict[str, Any]:
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

    async def retrieve_products_from_perception(
        self,
        *,
        concern: str,
        detection_summary: dict[str, Any] | None,
        texture_summary: dict[str, Any] | None,
        review: dict[str, Any] | None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        query = self.build_perception_query(
            concern=concern,
            detection_summary=detection_summary,
            texture_summary=texture_summary,
            review=review,
        )
        products = await asyncio.to_thread(self.kb.query, query, top_k)
        previews = self._preview_products(products)
        return {
            "query": query,
            "products": previews,
            "products_context": self.kb.format_context(products),
            "recommendation_markdown": self.build_template_recommendation(
                concern=concern,
                detection_summary=detection_summary,
                texture_summary=texture_summary,
                review=review,
                products=previews,
            ),
            "fallback_recommendation": self.build_template_recommendation(
                concern=concern,
                detection_summary=detection_summary,
                texture_summary=texture_summary,
                review=review,
                products=previews,
            ),
        }

    def build_perception_query(
        self,
        *,
        concern: str,
        detection_summary: dict[str, Any] | None,
        texture_summary: dict[str, Any] | None,
        review: dict[str, Any] | None,
    ) -> str:
        counts = (detection_summary or {}).get("counts") or {}
        detections = (detection_summary or {}).get("detections") or []
        active_textures = (texture_summary or {}).get("active_labels") or []
        observations = (review or {}).get("observations") or []
        uncertainties = (review or {}).get("uncertainties") or []

        top_detection_terms = [
            f"{name} ({count})"
            for name, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
        ]
        detailed_terms = [
            f"{item.get('class', '')} confidence {item.get('confidence', '')}"
            for item in detections[:10]
            if item.get("class")
        ]
        texture_terms = [
            f"{item.get('class', '')} probability {item.get('probability', '')}"
            for item in active_textures
            if item.get("class")
        ]

        needs = infer_product_needs(counts=counts, active_textures=active_textures, concern=concern)
        return "\n".join(
            [
                "Retrieve skincare products for this customer's visible concerns.",
                f"Customer concern: {concern or 'not provided'}",
                f"Priority product needs: {', '.join(needs) if needs else 'gentle basic routine'}",
                f"YOLO localized detections: {', '.join(top_detection_terms) if top_detection_terms else 'none above threshold'}",
                f"Detection details: {', '.join(detailed_terms) if detailed_terms else 'none'}",
                f"EfficientNet image-level texture labels: {', '.join(texture_terms) if texture_terms else 'none above threshold'}",
                f"Visual review observations: {'; '.join(str(item) for item in observations) if observations else 'none'}",
                f"Uncertainties to respect: {'; '.join(str(item) for item in uncertainties) if uncertainties else 'none'}",
                "Prefer products matching acne, redness/sensitive skin, oily skin, enlarged pores, clogged pores, blackheads, whiteheads, hydration, barrier support, soothing, non-comedogenic, and gentle use when relevant.",
            ]
        )

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
                "Tulis rekomendasi dalam format Markdown ringkas dan personal: sapaan singkat, ringkasan kondisi, prioritas masalah utama, 3-5 produk rekomendasi, alasan bahan aktif, peringatan alergi/kehamilan bila ada, dan cara pakai pagi/malam yang realistis.",
            ]
        )
        messages = [
            {"role": "system", "content": RECOMMENDATION_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        async for token in self.ollama.stream_chat(messages, temperature=0.35):
            yield token

    def build_template_recommendation(
        self,
        *,
        concern: str,
        detection_summary: dict[str, Any] | None,
        texture_summary: dict[str, Any] | None,
        review: dict[str, Any] | None,
        products: list[dict[str, Any]],
    ) -> str:
        counts = (detection_summary or {}).get("counts") or {}
        active_textures = (texture_summary or {}).get("active_labels") or []
        needs = infer_product_needs(counts=counts, active_textures=active_textures, concern=concern)
        visible_terms = readable_visible_terms(counts=counts, active_textures=active_textures)
        observations = (review or {}).get("observations") or []
        basis = "; ".join([*visible_terms, *[str(item) for item in observations[:2]]])
        if not basis:
            basis = concern or "keluhan kulit yang kamu tulis"

        if not products:
            return (
                "### Rekomendasi Produk\n"
                "Belum ada produk yang cocok ditemukan dari knowledge base lokal untuk sinyal kulit ini.\n\n"
                "### Prioritas\n"
                f"Fokus kebutuhan: {', '.join(needs)}."
            )

        lines = [
            "### Rekomendasi Produk Untuk Kamu",
            f"Dari analisis visual dan keluhanmu, prioritasnya adalah: **{', '.join(needs)}**.",
            "Aku pilih produk di bawah dari knowledge base karena metadata produk paling nyambung dengan concern yang terlihat/ditulis.",
            "",
        ]
        for index, product in enumerate(products[:5], start=1):
            name = product.get("product_name") or "Produk tanpa nama"
            brand = product.get("brand") or "Brand tidak diketahui"
            category = product.get("category") or "kategori tidak tersedia"
            concerns = product.get("concerns") or "klaim/concern tidak tersedia"
            skin_type = product.get("skin_type") or "tipe kulit tidak tersedia"
            functions = product.get("ingredient_functions") or ""
            warning = product.get("allergen_flag") or "tidak ada catatan alergi dari data"
            ingredient_warning = product.get("ingredient_warnings") or ""
            pregnancy = product.get("pregnancy_safe") or "unknown"
            price = product.get("price") or "harga tidak tersedia"
            reason = customer_reason_for_product(
                needs=needs,
                basis=basis,
                product=product,
            )
            role = product_role(product)
            usage = usage_tip_for_product(product)
            caution = caution_for_product(
                product=product,
                warning=warning,
                ingredient_warning=ingredient_warning,
                pregnancy=pregnancy,
            )
            lines.extend(
                [
                    f"{index}. **{brand} - {name}**",
                    f"   - Peran di rutinitas: {role}",
                    f"   - Kenapa cocok buat kamu: {reason}",
                    f"   - Yang mendukung: {concerns}.",
                    f"   - Fungsi bahan yang tercatat: {functions or 'tidak ada fungsi bahan spesifik di metadata'}."
                    f" Cocok untuk: {skin_type}.",
                    f"   - Cara pakai: {usage}",
                    f"   - Catatan hati-hati: {caution}",
                    f"   - Harga: {price}.",
                    "",
                ]
            )

        lines.extend(
            [
                "### Urutan Pakai",
                "- Pagi: pilih salah satu facial wash yang lembut, lanjut moisturizer bila ada, lalu sunscreen.",
                "- Malam: facial wash, lalu serum/treatment yang paling relevan. Jika memakai peeling serum, jangan dipakai setiap malam.",
                "- Jangan mulai semua produk sekaligus. Mulai dari cleanser atau soothing product dulu, lalu tambah serum setelah kulit cocok.",
                "- Kalau kulit sedang perih/iritasi, prioritaskan produk soothing/barrier dan tunda exfoliating/peeling.",
            ]
        )
        return "\n".join(lines)

    async def stream_perception_recommendation(
        self,
        *,
        concern: str,
        detection_summary: dict[str, Any] | None,
        texture_summary: dict[str, Any] | None,
        review: dict[str, Any] | None,
        products_context: str,
    ) -> AsyncIterator[str]:
        perception_context = {
            "customer_concern": concern or "",
            "yolo_localized_counts": (detection_summary or {}).get("counts", {}),
            "yolo_top_detections": (detection_summary or {}).get("detections", [])[:12],
            "efficientnet_active_texture_labels": (texture_summary or {}).get("active_labels", []),
            "visual_review": review or {},
            "product_needs": infer_product_needs(
                counts=(detection_summary or {}).get("counts") or {},
                active_textures=(texture_summary or {}).get("active_labels") or [],
                concern=concern,
            ),
        }
        user_prompt = "\n\n".join(
            [
                "Konteks customer dan hasil analisis visual:\n"
                + json.dumps(perception_context, ensure_ascii=False, indent=2),
                f"Produk dari knowledge base:\n{products_context}",
                (
                    "Tulis rekomendasi dalam Bahasa Indonesia dengan Markdown ringkas. "
                    "Gunakan hanya produk dari knowledge base. Pilih 3-5 produk bila tersedia. "
                    "Untuk setiap produk tulis: nama produk, fungsi utama, 'Kenapa cocok buat kamu', "
                    "bahan/klaim yang mendukung, dan catatan hati-hati bila ada allergen warning, komedogenik, iritan, atau status kehamilan belum jelas. "
                    "Buat alasan dari POV customer, misalnya karena area pipi terlihat kemerahan atau ada deteksi acne/whiteheads, bukan alasan generik. "
                    "Akhiri dengan urutan pakai pagi/malam yang realistis. Jangan diagnosis medis dan jangan menjanjikan hasil pasti."
                ),
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
                    "ingredients": meta.get("ingredients", ""),
                    "ingredient_functions": meta.get("ingredient_functions", ""),
                    "ingredient_warnings": meta.get("ingredient_warnings", ""),
                    "product_url": meta.get("product_url", ""),
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


def infer_product_needs(
    *,
    counts: dict[str, Any],
    active_textures: list[dict[str, Any]],
    concern: str,
) -> list[str]:
    haystack = " ".join(
        [
            concern,
            " ".join(str(key) for key, value in counts.items() if value),
            " ".join(str(item.get("class", "")) for item in active_textures),
        ]
    ).casefold()
    needs = []
    if any(token in haystack for token in ["acne", "jerawat", "whiteheads", "blackheads", "komedo"]):
        needs.extend(["acne care", "clogged pores", "non-comedogenic"])
    if any(token in haystack for token in ["skin-redness", "redness", "kemerahan", "irritation", "iritasi"]):
        needs.extend(["soothing", "sensitive skin", "barrier support"])
    if any(token in haystack for token in ["oily-skin", "oily", "berminyak", "oil"]):
        needs.extend(["oil control", "lightweight hydration"])
    if any(token in haystack for token in ["enlarged-pores", "pore", "pori"]):
        needs.extend(["pore care", "texture smoothing"])
    if any(token in haystack for token in ["dry-skin", "dry", "kering", "dehydrated", "dehidrasi"]):
        needs.extend(["hydration", "barrier support"])
    if any(token in haystack for token in ["wrinkles", "wrinkle", "fine line", "garis halus"]):
        needs.extend(["anti-aging", "hydration"])
    if not needs:
        needs.extend(["gentle cleanser", "moisturizer", "sunscreen"])
    return unique_preserve_order(needs)


def readable_visible_terms(
    *,
    counts: dict[str, Any],
    active_textures: list[dict[str, Any]],
) -> list[str]:
    terms = []
    for name, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        if count:
            terms.append(f"terdeteksi {name} pada {count} area")
    for item in active_textures:
        class_name = item.get("class")
        probability = item.get("probability")
        if class_name:
            terms.append(f"tekstur image-level {class_name} aktif ({probability})")
    return terms[:8]


def customer_reason_for_product(
    *,
    needs: list[str],
    basis: str,
    product: dict[str, Any],
) -> str:
    product_text = " ".join(
        str(product.get(key, ""))
        for key in [
            "category",
            "skin_type",
            "concerns",
            "ingredient_functions",
            "allergen_flag",
            "pregnancy_safe",
        ]
    ).casefold()
    matched = [need for need in needs if any(token in product_text for token in need.casefold().split())]
    if not matched:
        matched = needs[:2]
    concern_text = product.get("concerns") or product.get("category") or "klaim produk ini"
    return (
        f"karena input kamu menunjukkan {basis}. Produk ini relevan untuk "
        f"{', '.join(matched)} dan metadata produknya menyebut {concern_text}."
    )


def product_role(product: dict[str, Any]) -> str:
    text = f"{product.get('category', '')} {product.get('product_name', '')} {product.get('concerns', '')}".casefold()
    if any(token in text for token in ["facial wash", "cleanser", "cleansing", "foam"]):
        return "pembersih wajah untuk mengangkat minyak, debu, sunscreen, dan kotoran dari pori."
    if any(token in text for token in ["peeling", "exfoliation", "exfoliating"]):
        return "exfoliating treatment mingguan untuk tekstur, pori tersumbat, minyak berlebih, dan kusam."
    if any(token in text for token in ["serum", "blemish", "pore"]):
        return "serum treatment untuk blemish, pori, tekstur, dan tampilan kulit yang tidak merata."
    if any(token in text for token in ["toner", "pad", "soothing", "cooling"]):
        return "toner/toner pad untuk hidrasi ringan, calming, dan rasa sejuk pada kulit sensitif."
    return "produk pendukung rutinitas sesuai concern yang tertera di knowledge base."


def usage_tip_for_product(product: dict[str, Any]) -> str:
    text = f"{product.get('category', '')} {product.get('product_name', '')} {product.get('concerns', '')}".casefold()
    if any(token in text for token in ["facial wash", "cleanser", "cleansing", "foam"]):
        return "pakai pagi dan malam; pilih satu cleanser saja agar kulit tidak terlalu kering."
    if any(token in text for token in ["peeling", "exfoliation", "exfoliating"]):
        return "pakai malam 1-2 kali seminggu dulu; jangan digabung dengan exfoliant lain di malam yang sama."
    if "serum" in text:
        return "pakai setelah cuci muka, mulai 2-3 kali seminggu lalu naikkan frekuensi jika kulit nyaman."
    if any(token in text for token in ["toner", "pad"]):
        return "pakai setelah cuci muka saat kulit terasa panas, kemerahan, atau butuh calming."
    return "pakai bertahap dan lakukan patch test sebelum rutin."


def caution_for_product(
    *,
    product: dict[str, Any],
    warning: str,
    ingredient_warning: str,
    pregnancy: str,
) -> str:
    text = f"{product.get('product_name', '')} {product.get('concerns', '')}".casefold()
    notes = []
    if any(token in text for token in ["peeling", "exfoliation", "exfoliating"]):
        notes.append("karena ini exfoliating/peeling, mulai pelan dan wajib sunscreen pagi.")
    if warning and not warning.startswith("none detected"):
        notes.append(warning)
    if ingredient_warning:
        notes.append(ingredient_warning)
    if pregnancy and pregnancy != "unknown":
        notes.append(f"status kehamilan: {pregnancy}")
    elif pregnancy == "unknown":
        notes.append("status kehamilan belum pasti dari metadata.")
    if not notes:
        notes.append("tidak ada warning spesifik dari metadata, tetap patch test.")
    return " ".join(notes)


def unique_preserve_order(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        result.append(value)
        seen.add(key)
    return result


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
