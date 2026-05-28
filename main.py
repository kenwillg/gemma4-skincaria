import asyncio
import base64
import binascii
import io
import json
import os
import socket
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from ollama_client import OllamaClient


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_HTML = STATIC_DIR / "index.html"
IMAGE_TEST_HTML = STATIC_DIR / "image_test.html"
FACE_MODEL_IMAGE = BASE_DIR / "data" / "face-model.jpg"
EVALUATION_DIR = BASE_DIR / "data" / "evaluation"
LLM_JUDGE_CASES_JSONL = EVALUATION_DIR / "llm_judge_cases.jsonl"
MODEL_NAME = os.getenv("SKINCARIA_MODEL", "gemma4:e4b")
HOST = "0.0.0.0"
PORT = 8000


app = FastAPI(title="Skincaria", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(INDEX_HTML)


@app.get("/app")
async def skincare_app():
    return FileResponse(INDEX_HTML)


@app.get("/image-test")
async def image_test():
    return FileResponse(IMAGE_TEST_HTML)


@app.get("/assets/face-model")
async def face_model_asset():
    if not FACE_MODEL_IMAGE.exists():
        raise HTTPException(status_code=404, detail="face-model.jpg not found in data/.")
    return FileResponse(FACE_MODEL_IMAGE)


@app.get("/health")
async def health():
    client = get_ollama_client()
    kb = get_kb()

    try:
        models = await client.list_models()
        ollama_running = True
        model_available = MODEL_NAME in models
    except httpx.HTTPError:
        ollama_running = False
        model_available = False

    return {
        "ollama_running": ollama_running,
        "model": MODEL_NAME,
        "model_available": model_available,
        "kb_collection": "skincare_kb",
        "kb_document_count": kb.count(),
    }


@app.post("/evaluation/cases")
async def save_evaluation_case(request: Request):
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Expected a JSON object.")

    user_concern = str(payload.get("user_concern") or "").strip()
    assistant_response = str(payload.get("assistant_response") or "").strip()
    retrieved_products = payload.get("retrieved_products") or []
    visual_evidence = payload.get("visual_evidence") or {}

    if not user_concern and not visual_evidence:
        raise HTTPException(status_code=400, detail="Evaluation case needs a concern or visual evidence.")
    if not assistant_response:
        raise HTTPException(status_code=400, detail="Evaluation case needs an assistant response.")
    if not isinstance(retrieved_products, list):
        raise HTTPException(status_code=400, detail="retrieved_products must be a list.")
    if not isinstance(visual_evidence, dict):
        raise HTTPException(status_code=400, detail="visual_evidence must be an object.")

    row = {
        "id": payload.get("id") or f"case_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "user_concern": user_concern,
        "visual_evidence": visual_evidence,
        "retrieved_products": retrieved_products,
        "assistant_response": assistant_response,
        "metadata": {
            "source": "skincaria_frontend",
            "raw_image_saved": False,
            "note": "Raw face image is intentionally not stored in the LLM judge case.",
        },
    }
    await asyncio.to_thread(append_jsonl_row, LLM_JUDGE_CASES_JSONL, row)
    return {
        "saved": True,
        "id": row["id"],
        "path": str(LLM_JUDGE_CASES_JSONL.relative_to(BASE_DIR)),
    }


@app.post("/analyze")
async def analyze(
    concern: Annotated[str, Form()] = "",
    image: Annotated[UploadFile | None, File()] = None,
    image_base64: Annotated[str | None, Form()] = None,
):
    from pipeline import ManualDescriptionRequired

    pipeline = get_pipeline()
    encoded_image = await extract_image_base64(image, image_base64)

    try:
        result = await pipeline.run_analysis(
            image_base64=encoded_image,
            concern=concern,
        )
        return JSONResponse(result)
    except ManualDescriptionRequired as exc:
        return JSONResponse(
            {
                "manual_input_required": True,
                "message": str(exc),
                "skin_labels": None,
                "products": [],
                "products_context": "",
            },
            status_code=200,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"Ollama tidak merespons: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"Inferensi model gagal: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/image-caption")
async def image_caption(
    prompt: Annotated[str, Form()] = "Describe this image in detail.",
    image: Annotated[UploadFile | None, File()] = None,
    image_base64: Annotated[str | None, Form()] = None,
):
    encoded_image = await extract_image_base64(image, image_base64)
    if not encoded_image:
        raise HTTPException(status_code=400, detail="Upload image atau kirim image_base64.")

    try:
        client = get_ollama_client()
        messages = [
            {
                "role": "system",
                "content": "You are a precise visual captioning model. Describe only what is visible.",
            },
            {
                "role": "user",
                "content": prompt.strip() or "Describe this image in detail.",
                "images": [encoded_image],
            },
        ]
        caption = await client.chat(messages, temperature=0.1)
        return {
            "model": client.model,
            "prompt": prompt,
            "caption": caption,
        }
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"Ollama tidak merespons: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"Inferensi gambar gagal: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.websocket("/ws/recommend")
async def recommend(websocket: WebSocket):
    await websocket.accept()
    pipeline = get_pipeline()
    try:
        payload = await websocket.receive_json()
        skin_labels = payload.get("skin_labels") or {}
        concern = payload.get("concern") or ""
        products_context = payload.get("products_context") or ""

        await websocket.send_json({"type": "stage", "stage": "Membuat rekomendasi..."})
        async for token in pipeline.stream_recommendation(
            skin_labels=skin_labels,
            concern=concern,
            products_context=products_context,
        ):
            await websocket.send_json({"type": "token", "content": token})
        await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close()


@app.websocket("/ws/agentic")
async def agentic_analysis(websocket: WebSocket):
    await websocket.accept()
    agent = get_agentic_pipeline()
    try:
        payload = await websocket.receive_json()
        concern = (payload.get("concern") or "").strip()
        image_base64 = payload.get("image_base64")

        if not image_base64 and not concern:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "Please turn video on, upload a photo, or tell Skincaria what changed.",
                }
            )
            await websocket.close()
            return

        await websocket.send_json(
            {
                "type": "stage",
                "stage": "plan",
                "message": "Gemma is routing the request and choosing tools.",
            }
        )
        plan = await agent.plan(concern=concern, has_image=bool(image_base64))
        await websocket.send_json({"type": "plan", "plan": plan})

        quality = None
        detections = None
        texture = None
        if image_base64:
            await websocket.send_json(
                {
                    "type": "stage",
                    "stage": "quality",
                    "message": "Checking whether the image contains enough face/skin signal.",
                }
            )
            quality = await agent.assess_input(image_base64)
            await websocket.send_json({"type": "quality", "quality": quality})

            await websocket.send_json(
                {
                    "type": "stage",
                    "stage": "detect",
                    "message": "YOLO11n 960 is detecting visible skin-condition boxes.",
                }
            )
            detections = await agent.detect(image_base64)
            await websocket.send_json({"type": "detections", "detections": detections})

            await websocket.send_json(
                {
                    "type": "stage",
                    "stage": "classify",
                    "message": "EfficientNetV2-B0 is classifying image-level texture labels.",
                }
            )
            texture = await agent.classify_texture(image_base64)
            await websocket.send_json({"type": "texture", "texture": texture})

        await websocket.send_json(
            {
                "type": "stage",
                "stage": "answer",
                "message": "Gemma is reviewing the detections and preparing the answer.",
            }
        )
        review = await agent.review(
            concern=concern,
            detection_summary=detections,
            texture_summary=texture,
            original_image_base64=image_base64,
        )
        await websocket.send_json({"type": "review", "review": review})

        recommendation_pipeline = get_pipeline()
        await websocket.send_json(
            {
                "type": "stage",
                "stage": "retrieve",
                "message": "Searching the product knowledge base for matched skincare options.",
            }
        )
        retrieval = await recommendation_pipeline.retrieve_products_from_perception(
            concern=concern,
            detection_summary=detections,
            texture_summary=texture,
            review=review,
            top_k=5,
        )
        await websocket.send_json({"type": "products", "retrieval": retrieval})

        await websocket.send_json(
            {
                "type": "stage",
                "stage": "recommend",
                "message": "Preparing grounded product recommendations with customer-specific reasons.",
            }
        )
        await websocket.send_json(
            {
                "type": "recommendation_token",
                "content": retrieval.get("recommendation_markdown")
                or "Tidak ada rekomendasi produk yang bisa dibuat dari knowledge base lokal.",
            }
        )
        await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close()


# FUTURE: Voice input via Faster Whisper
# from faster_whisper import WhisperModel
# model = WhisperModel("small", device="cuda", compute_type="float16")
# segments, _ = model.transcribe(audio_bytes)
# transcript = " ".join([s.text for s in segments])
# To activate: install faster-whisper, uncomment this route,
# and enable the mic button in index.html
# @app.post("/transcribe")
# async def transcribe(audio: UploadFile = File(...)):
#     audio_bytes = await audio.read()
#     segments, _ = model.transcribe(audio_bytes)
#     transcript = " ".join([s.text for s in segments])
#     return {"transcript": transcript}


def get_ollama_client() -> OllamaClient:
    client = getattr(app.state, "ollama_client", None)
    if client is None:
        client = OllamaClient(model=MODEL_NAME)
        app.state.ollama_client = client
    return client


def get_kb() -> Any:
    from kb_builder import DEFAULT_CHROMA_DIR, DEFAULT_DATA_DIR, SkincareKB

    kb = getattr(app.state, "kb", None)
    if kb is None:
        kb = SkincareKB(data_dir=DEFAULT_DATA_DIR, persist_dir=DEFAULT_CHROMA_DIR)
        app.state.kb = kb
    return kb


def get_pipeline() -> Any:
    from pipeline import SkincariaPipeline

    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        pipeline = SkincariaPipeline(ollama=get_ollama_client(), kb=get_kb())
        app.state.pipeline = pipeline
    return pipeline


def get_agentic_pipeline() -> Any:
    from agentic_yolo import AgenticYoloPipeline

    pipeline = getattr(app.state, "agentic_pipeline", None)
    if pipeline is None:
        pipeline = AgenticYoloPipeline(ollama=get_ollama_client())
        app.state.agentic_pipeline = pipeline
    return pipeline


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


async def extract_image_base64(
    image: UploadFile | None,
    image_base64: str | None,
) -> str | None:
    if image_base64:
        return strip_data_url(image_base64)

    if image is None:
        return None

    raw = await image.read()
    if not raw:
        return None

    try:
        text = raw.decode("utf-8").strip()
        if text.startswith("data:"):
            return strip_data_url(text)
        if looks_like_base64(text):
            return text
    except UnicodeDecodeError:
        pass

    return await asyncio.to_thread(image_bytes_to_base64, raw)


def looks_like_base64(value: str) -> bool:
    if len(value) < 128:
        return False
    try:
        base64.b64decode(value, validate=True)
        return True
    except binascii.Error:
        return False


def image_bytes_to_base64(raw: bytes) -> str:
    with Image.open(io.BytesIO(raw)) as image:
        image = image.convert("RGB")
        image.thumbnail((1024, 1024))
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=88, optimize=True)
    return base64.b64encode(output.getvalue()).decode("ascii")


def strip_data_url(value: str) -> str:
    if "," in value and value.strip().startswith("data:"):
        return value.split(",", 1)[1]
    return value.strip()


async def startup_checks() -> bool:
    client = get_ollama_client()
    print("Memeriksa Ollama di http://localhost:11434 ...")
    try:
        models = await client.list_models()
    except httpx.HTTPError as exc:
        print()
        print("Ollama belum berjalan atau tidak bisa diakses.")
        print("Jalankan Ollama terlebih dahulu, lalu ulangi:")
        print("  ollama serve")
        print("  python main.py")
        print(f"Detail: {exc}")
        return False

    if MODEL_NAME not in models:
        try:
            await client.pull_model()
        except Exception as exc:
            print()
            print(f"Gagal mengunduh model {MODEL_NAME} via Ollama.")
            print("Pastikan nama model benar dan koneksi Ollama tersedia.")
            print(f"Detail: {exc}")
            return False
    else:
        print(f"Model {MODEL_NAME} sudah tersedia.")

    kb = get_kb()
    try:
        await asyncio.to_thread(kb.ensure_built)
    except FileNotFoundError as exc:
        print()
        print(str(exc))
        print()
        print("Pastikan dataset CSV tersedia di folder data/ sesuai daftar di atas.")
        print("Lalu jalankan ulang: python main.py")
        return False
    except Exception as exc:
        print()
        print("Gagal membangun knowledge base ChromaDB.")
        print(f"Detail: {exc}")
        return False

    get_pipeline()
    return True


def get_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


def main() -> None:
    if os.getenv("SKINCARIA_STARTUP_CHECKS", "0") == "1":
        ready = asyncio.run(startup_checks())
        if not ready:
            sys.exit(1)

    lan_ip = get_lan_ip()
    print()
    print(f"Skincaria agent loop: http://localhost:{PORT}")
    print(f"Skincaria agent loop di HP: http://{lan_ip}:{PORT}")
    print(f"Image caption tester: http://localhost:{PORT}/image-test")
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
