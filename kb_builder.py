import csv
import hashlib
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
        missing = [path for path in [sociolla, inci] if not path.exists()]
        if missing:
            missing_list = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(f"File CSV knowledge base belum ditemukan:\n{missing_list}")

        rows_by_key: dict[str, dict[str, str]] = {}
        for path in [sociolla, inci]:
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
        ]
        normalized = {field: self._clean_value(row.get(field, "")) for field in fields}
        return normalized

    def _document_from_row(self, row: dict[str, str]) -> str:
        return (
            f"Brand: {row['brand']}. "
            f"Product: {row['product_name']}. "
            f"Category: {row['category']}. "
            f"Price: {row['price']}. "
            f"Skin type: {row['skin_type']}. "
            f"Concerns: {row['concerns']}. "
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
