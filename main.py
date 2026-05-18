import asyncio
import base64
import binascii
import io
import os
import socket
import sys
from pathlib import Path
from typing import Annotated, Any

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from ollama_client import OllamaClient


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_HTML = STATIC_DIR / "index.html"
IMAGE_TEST_HTML = STATIC_DIR / "image_test.html"
MODEL_NAME = os.getenv("SKINCARIA_MODEL", "gemma4:e4b")
HOST = "0.0.0.0"
PORT = 8000


app = FastAPI(title="Skincaria", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(IMAGE_TEST_HTML)


@app.get("/app")
async def skincare_app():
    return FileResponse(INDEX_HTML)


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
    print(f"Image caption tester: http://localhost:{PORT}")
    print(f"Image caption tester di HP: http://{lan_ip}:{PORT}")
    print(f"Skincaria app lama: http://localhost:{PORT}/app")
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
