import ast
import csv
import hashlib
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "skincare_kb"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class SkincareKB:
    def __init__(
        self,
        data_dir: Path | str = DEFAULT_DATA_DIR,
        persist_dir: Path | str = DEFAULT_CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self._embedding_model: SentenceTransformer | None = None
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

    @property
    def embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            print(f"Memuat embedding model: {EMBEDDING_MODEL}")
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedding_model

    def collection_exists(self) -> bool:
        names = []
        for collection in self.client.list_collections():
            names.append(getattr(collection, "name", str(collection)))
        return self.collection_name in names

    def get_collection(self):
        return self.client.get_collection(self.collection_name)

    def get_or_create_collection(self):
        return self.client.get_or_create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        if not self.collection_exists():
            return 0
        return self.get_collection().count()

    def ensure_built(self) -> int:
        if self.count() > 0:
            print(f"ChromaDB siap: {self.count()} dokumen dalam collection {self.collection_name}.")
            return self.count()
        return self.build()

    def build(self) -> int:
        products = self._load_products()
        if not products:
            raise RuntimeError("Tidak ada produk yang bisa dimasukkan ke knowledge base.")

        if self.collection_exists():
            self.client.delete_collection(self.collection_name)

        collection = self.get_or_create_collection()
        print(f"Membangun ChromaDB collection {self.collection_name} dari {len(products)} produk...")

        batch_size = 64
        for start in range(0, len(products), batch_size):
            batch = products[start : start + batch_size]
            docs = [item["document"] for item in batch]
            embeddings = self._embed_texts(docs)
            collection.add(
                ids=[item["id"] for item in batch],
                documents=docs,
                metadatas=[item["metadata"] for item in batch],
                embeddings=embeddings,
            )
            end = min(start + batch_size, len(products))
            print(f"KB progress: {end}/{len(products)} dokumen")

        print(f"Knowledge base selesai: {collection.count()} dokumen tersimpan.")
        return collection.count()

    def query(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self.count() == 0:
            return []

        collection = self.get_collection()
        query_embedding = self._embed_texts([query_text])[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        products: list[dict[str, Any]] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for index, document in enumerate(documents):
            distance = distances[index] if index < len(distances) else None
            products.append(
                {
                    "id": ids[index] if index < len(ids) else "",
                    "document": document,
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                    "distance": distance,
                    "similarity": None if distance is None else max(0.0, 1.0 - float(distance)),
                }
            )
        return products

    def format_context(self, products: list[dict[str, Any]]) -> str:
        if not products:
            return "Tidak ada produk yang ditemukan di knowledge base."

        blocks = []
        for index, product in enumerate(products, start=1):
            meta = product.get("metadata", {})
            blocks.append(
                "\n".join(
                    [
                        f"Produk {index}:",
                        f"Brand: {meta.get('brand', '')}",
                        f"Product: {meta.get('product_name', '')}",
                        f"Category: {meta.get('category', '')}",
                        f"Price: {meta.get('price', '')}",
                        f"Skin type: {meta.get('skin_type', '')}",
                        f"Concerns: {meta.get('concerns', '')}",
                        f"Allergen flag: {meta.get('allergen_flag', '')}",
                        f"Pregnancy safe: {meta.get('pregnancy_safe', '')}",
                        f"Ingredient functions: {meta.get('ingredient_functions', '')}",
                        f"Ingredient warnings: {meta.get('ingredient_warnings', '')}",
                        f"BPOM: {meta.get('bpom_id', '')}",
                        f"Ingredients: {meta.get('ingredients', '')}",
                    ]
                )
            )
        return "\n\n".join(blocks)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def _load_products(self) -> list[dict[str, Any]]:
        sociolla = self.data_dir / "sociolla_products.csv"
        inci = self.data_dir / "inci_products.csv"
        if sociolla.exists() and inci.exists():
            return self._load_normalized_products([sociolla, inci])

        product_csv = self.data_dir / "Indonesian Skincare Sample Dataset" / "product.csv"
        claim_csv = self.data_dir / "Indonesian Skincare Sample Dataset" / "product_claim_category.csv"
        ingredient_csv = self.data_dir / "Indonesian Skincare Sample Dataset" / "ingredients_category.csv"
        inci_csv = self.data_dir / "Skin care product ingredients - INCI List" / "ingredientsList.csv"
        current_dataset_files = [product_csv, claim_csv, ingredient_csv, inci_csv]
        if all(path.exists() for path in current_dataset_files):
            return self._load_current_dataset_products(
                product_csv=product_csv,
                claim_csv=claim_csv,
                ingredient_csv=ingredient_csv,
                inci_csv=inci_csv,
            )

        expected = [sociolla, inci, *current_dataset_files]
        missing = [path for path in expected if not path.exists()]
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"File CSV knowledge base belum lengkap:\n{missing_list}")

    def _load_normalized_products(self, paths: list[Path]) -> list[dict[str, Any]]:
        rows_by_key: dict[str, dict[str, str]] = {}
        for path in paths:
            print(f"Membaca CSV: {path}")
            for row in self._read_csv(path):
                key = self._dedupe_key(row)
                existing = rows_by_key.get(key, {})
                merged = {**existing}
                for field, value in row.items():
                    if value != "":
                        merged[field] = value
                    elif field not in merged:
                        merged[field] = ""
                rows_by_key[key] = merged

        products = []
        for row in rows_by_key.values():
            normalized = self._normalize_row(row)
            document = self._document_from_row(normalized)
            products.append(
                {
                    "id": self._stable_id(normalized),
                    "document": document,
                    "metadata": normalized,
                }
            )
        return products

    def _load_current_dataset_products(
        self,
        *,
        product_csv: Path,
        claim_csv: Path,
        ingredient_csv: Path,
        inci_csv: Path,
    ) -> list[dict[str, Any]]:
        print(f"Membaca CSV produk: {product_csv}")
        product_rows = self._read_csv(product_csv)
        claim_map = self._load_claim_map(claim_csv)
        ingredient_lookup = self._load_ingredient_lookup(ingredient_csv)
        inci_lookup = self._load_inci_lookup(inci_csv)

        products = []
        for row in product_rows:
            normalized = self._normalize_current_dataset_row(
                row,
                claim_map=claim_map,
                ingredient_lookup=ingredient_lookup,
                inci_lookup=inci_lookup,
            )
            document = self._document_from_row(normalized)
            products.append(
                {
                    "id": self._stable_id(normalized),
                    "document": document,
                    "metadata": normalized,
                }
            )
        return products

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError(f"CSV kosong atau tidak valid: {path}")
            return [
                {self._clean_key(key): self._clean_value(value) for key, value in row.items()}
                for row in reader
                if any((value or "").strip() for value in row.values())
            ]

    def _normalize_row(self, row: dict[str, str]) -> dict[str, str]:
        fields = [
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
        ]
        normalized = {field: self._clean_value(row.get(field, "")) for field in fields}
        return normalized

    def _normalize_current_dataset_row(
        self,
        row: dict[str, str],
        *,
        claim_map: dict[str, str],
        ingredient_lookup: dict[str, dict[str, str]],
        inci_lookup: dict[str, dict[str, str]],
    ) -> dict[str, str]:
        product_name = self._clean_value(row.get("product_name", ""))
        brand = self._clean_value(row.get("brand", ""))
        if not product_name or not brand:
            raise ValueError("Setiap produk harus memiliki product_name dan brand.")

        ingredients = self._clean_value(row.get("ingredients_list", ""))
        ingredient_names = self._parse_ingredient_list(ingredients)
        claim_texts = self._split_list_text(row.get("description_product", ""))
        claim_categories = self._unique_values(
            claim_map.get(self._canonical_text(claim), claim)
            for claim in claim_texts
        )

        ingredient_functions = self._ingredient_functions(ingredient_names, ingredient_lookup)
        ingredient_warnings = self._ingredient_warnings(ingredient_names, ingredient_lookup)
        inci_notes = self._inci_notes(ingredient_names, inci_lookup)

        normalized = {
            "product_name": product_name,
            "brand": brand,
            "category": self._clean_value(row.get("product_type", "")),
            "price": self._format_price(
                normal_price=row.get("normal_price", ""),
                discount_price=row.get("discount_price", ""),
            ),
            "ingredients": ingredients,
            "skin_type": self._derive_skin_type(claim_texts, claim_categories, inci_notes),
            "concerns": ", ".join(self._unique_values([*claim_categories, *claim_texts])),
            "allergen_flag": self._derive_allergen_flag(ingredient_warnings, inci_notes),
            "pregnancy_safe": self._derive_pregnancy_safe(inci_notes),
            "ingredient_functions": ", ".join(ingredient_functions),
            "ingredient_warnings": "; ".join(ingredient_warnings),
            "bpom_id": self._clean_value(row.get("bpom_id", "")),
            "product_url": self._clean_value(row.get("product_url", "")),
            "rating": self._clean_value(row.get("rating", "")),
            "review_count": self._clean_value(row.get("review_count", "")),
        }
        return normalized

    def _document_from_row(self, row: dict[str, str]) -> str:
        return (
            f"Brand: {row['brand']}. "
            f"Product: {row['product_name']}. "
            f"Category: {row['category']}. "
            f"Price: {row['price']}. "
            f"Skin type: {row['skin_type']}. "
            f"Concerns: {row['concerns']}. "
            f"Ingredient functions: {row.get('ingredient_functions', '')}. "
            f"Ingredient warnings: {row.get('ingredient_warnings', '')}. "
            f"Allergen flag: {row.get('allergen_flag', '')}. "
            f"Pregnancy safe: {row.get('pregnancy_safe', '')}. "
            f"Ingredients: {row['ingredients']}."
        )

    def _dedupe_key(self, row: dict[str, str]) -> str:
        product = self._clean_value(row.get("product_name", "")).lower()
        brand = self._clean_value(row.get("brand", "")).lower()
        if not product or not brand:
            raise ValueError("Setiap produk harus memiliki product_name dan brand.")
        return f"{product}|{brand}"

    def _stable_id(self, row: dict[str, str]) -> str:
        raw = f"{row['brand']}|{row['product_name']}".lower().encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def _clean_key(self, value: str | None) -> str:
        return (value or "").strip().lower()

    def _clean_value(self, value: Any) -> str:
        return str(value or "").strip()

    def _load_claim_map(self, path: Path) -> dict[str, str]:
        print(f"Membaca CSV klaim: {path}")
        rows = self._read_csv(path)
        claim_map: dict[str, str] = {}
        for row in rows:
            description = self._canonical_text(row.get("description_product", ""))
            category = self._clean_value(row.get("claim_category", ""))
            if description and category:
                claim_map[description] = category
        return claim_map

    def _load_ingredient_lookup(self, path: Path) -> dict[str, dict[str, str]]:
        print(f"Membaca CSV kategori bahan: {path}")
        rows = self._read_csv(path)
        lookup: dict[str, dict[str, str]] = {}
        for row in rows:
            name = self._clean_value(row.get("ingredient_name", ""))
            if not name:
                continue
            for key in self._ingredient_keys(name):
                lookup[key] = row
        return lookup

    def _load_inci_lookup(self, path: Path) -> dict[str, dict[str, str]]:
        print(f"Membaca CSV INCI: {path}")
        rows = self._read_csv(path)
        lookup: dict[str, dict[str, str]] = {}
        for row in rows:
            name = self._clean_value(row.get("name", ""))
            if not name:
                continue
            for key in self._ingredient_keys(name):
                lookup[key] = row
        return lookup

    def _parse_ingredient_list(self, value: str) -> list[str]:
        return self._split_list_text(value)

    def _split_list_text(self, value: str | None) -> list[str]:
        return self._unique_values(
            part.strip()
            for part in self._clean_value(value).split(",")
            if part.strip()
        )

    def _ingredient_keys(self, value: str) -> list[str]:
        cleaned = self._clean_value(value)
        without_parentheses = re.sub(r"\s*\([^)]*\)", "", cleaned).strip()
        keys = [
            self._canonical_text(cleaned),
            self._canonical_text(without_parentheses),
        ]
        return [key for key in self._unique_values(keys) if key]

    def _canonical_text(self, value: str | None) -> str:
        cleaned = re.sub(r"\s+", " ", self._clean_value(value)).strip()
        return cleaned.casefold()

    def _unique_values(self, values: Any) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = self._clean_value(value)
            if not cleaned:
                continue
            key = cleaned.casefold()
            if key not in seen:
                unique.append(cleaned)
                seen.add(key)
        return unique

    def _lookup_ingredient(
        self,
        ingredient_name: str,
        lookup: dict[str, dict[str, str]],
    ) -> dict[str, str] | None:
        for key in self._ingredient_keys(ingredient_name):
            if key in lookup:
                return lookup[key]
        return None

    def _ingredient_functions(
        self,
        ingredient_names: list[str],
        ingredient_lookup: dict[str, dict[str, str]],
    ) -> list[str]:
        functions = []
        for ingredient in ingredient_names:
            row = self._lookup_ingredient(ingredient, ingredient_lookup)
            if not row:
                continue
            functions.extend([row.get("function1", ""), row.get("function2", "")])
        return self._unique_values(functions)

    def _ingredient_warnings(
        self,
        ingredient_names: list[str],
        ingredient_lookup: dict[str, dict[str, str]],
    ) -> list[str]:
        warnings = []
        for ingredient in ingredient_names:
            row = self._lookup_ingredient(ingredient, ingredient_lookup)
            if not row:
                continue
            ingredient_warnings = self._unique_values([row.get("warning1", ""), row.get("warning2", "")])
            if ingredient_warnings:
                warnings.append(f"{ingredient}: {', '.join(ingredient_warnings)}")
        return warnings

    def _inci_notes(
        self,
        ingredient_names: list[str],
        inci_lookup: dict[str, dict[str, str]],
    ) -> list[dict[str, str]]:
        notes = []
        for ingredient in ingredient_names:
            row = self._lookup_ingredient(ingredient, inci_lookup)
            if not row:
                continue
            notes.append(
                {
                    "ingredient": ingredient,
                    "good_for": ", ".join(self._parse_python_listish(row.get("who_is_it_good_for", ""))),
                    "avoid": ", ".join(self._parse_python_listish(row.get("who_should_avoid", ""))),
                }
            )
        return notes

    def _parse_python_listish(self, value: str) -> list[str]:
        cleaned = self._clean_value(value)
        if not cleaned:
            return []
        try:
            parsed = ast.literal_eval(cleaned)
        except (ValueError, SyntaxError):
            return self._split_list_text(cleaned)
        if not isinstance(parsed, list):
            return self._split_list_text(cleaned)
        return self._unique_values(str(item).strip() for item in parsed if str(item).strip())

    def _derive_skin_type(
        self,
        claim_texts: list[str],
        claim_categories: list[str],
        inci_notes: list[dict[str, str]],
    ) -> str:
        haystack = " ".join(
            [
                *claim_texts,
                *claim_categories,
                *(note.get("good_for", "") for note in inci_notes),
            ]
        ).casefold()
        labels = []
        if any(token in haystack for token in ["dry", "dehydrated", "hidrasi", "melembapkan", "hydrat"]):
            labels.append("dry/dehydrated")
        if any(token in haystack for token in ["acne", "jerawat", "blackhead", "komedo", "pore", "pori", "oil"]):
            labels.append("oily/acne-prone")
        if any(token in haystack for token in ["sensitive", "sensitif", "redness", "kemerahan", "soothing"]):
            labels.append("sensitive")
        if not labels:
            labels.append("all/unspecified")
        return ", ".join(self._unique_values(labels))

    def _derive_allergen_flag(
        self,
        ingredient_warnings: list[str],
        inci_notes: list[dict[str, str]],
    ) -> str:
        allergy_related = [
            warning
            for warning in ingredient_warnings
            if any(token in warning.casefold() for token in ["allergen", "irritant", "drying", "comedogenic"])
        ]
        avoid_related = [
            f"{note['ingredient']}: {note['avoid']}"
            for note in inci_notes
            if note.get("avoid")
            and note.get("avoid", "").casefold() not in {"related allergy"}
        ]
        flags = self._unique_values([*allergy_related, *avoid_related])
        if flags:
            return "yes: " + "; ".join(flags[:8])
        return "none detected from matched ingredient data"

    def _derive_pregnancy_safe(self, inci_notes: list[dict[str, str]]) -> str:
        avoid = [
            note["ingredient"]
            for note in inci_notes
            if "pregnancy" in note.get("avoid", "").casefold()
        ]
        if avoid:
            return "avoid during pregnancy: " + ", ".join(self._unique_values(avoid))

        pregnancy_friendly = [
            note["ingredient"]
            for note in inci_notes
            if "pregnancy" in note.get("good_for", "").casefold()
        ]
        if pregnancy_friendly:
            return "unknown overall; matched pregnancy-friendly ingredients: " + ", ".join(
                self._unique_values(pregnancy_friendly[:8])
            )
        return "unknown"

    def _format_price(self, *, normal_price: str, discount_price: str) -> str:
        discount = self._clean_value(discount_price)
        normal = self._clean_value(normal_price)
        selected = discount if discount and discount != "0" else normal
        if not selected:
            return ""
        try:
            amount = int(float(selected))
        except ValueError:
            return selected
        return f"Rp{amount:,}".replace(",", ".")
