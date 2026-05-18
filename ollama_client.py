import json
import os
import sys
from typing import Any, AsyncIterator

import httpx


class OllamaClient:
    def __init__(self, model: str | None = None, base_url: str = "http://localhost:11434"):
        self.model = model or os.getenv("SKINCARIA_MODEL", "gemma4:e4b")
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0)
        self.num_ctx = int(os.getenv("SKINCARIA_NUM_CTX", "2048"))

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            response = await client.get("/api/tags")
            response.raise_for_status()
            payload = response.json()

        names: list[str] = []
        for item in payload.get("models", []):
            name = item.get("name") or item.get("model")
            if name:
                names.append(name)
        return names

    async def is_running(self) -> bool:
        try:
            await self.list_models()
            return True
        except httpx.HTTPError:
            return False

    async def has_model(self) -> bool:
        models = await self.list_models()
        return self.model in models

    async def pull_model(self) -> None:
        print(f"Model {self.model} belum tersedia. Mengunduh dari Ollama...")
        async with httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(None)) as client:
            async with client.stream(
                "POST",
                "/api/pull",
                json={"name": self.model, "stream": True},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    payload = json.loads(line)
                    if payload.get("error"):
                        raise RuntimeError(payload["error"])
                    self._print_pull_progress(payload)
                    if payload.get("status") == "success":
                        break
        print()
        print(f"Model {self.model} siap digunakan.")

    def _print_pull_progress(self, payload: dict[str, Any]) -> None:
        status = payload.get("status", "pulling")
        completed = payload.get("completed")
        total = payload.get("total")

        if completed and total:
            width = 30
            ratio = min(max(completed / total, 0.0), 1.0)
            filled = int(width * ratio)
            bar = "#" * filled + "-" * (width - filled)
            completed_mb = completed / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            message = f"\r{status:<24} [{bar}] {ratio * 100:5.1f}% {completed_mb:,.1f}/{total_mb:,.1f} MB"
        else:
            message = f"\r{status:<90}"

        sys.stdout.write(message)
        sys.stdout.flush()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},
        }
        if json_mode:
            body["format"] = "json"

        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            response = await client.post("/api/chat", json=body)
            self._raise_for_status(response)
            payload = response.json()

        return payload.get("message", {}).get("content", "")

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.35,
    ) -> AsyncIterator[str]:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},
        }

        async with httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(None)) as client:
            async with client.stream("POST", "/api/chat", json=body) as response:
                await self._raise_stream_for_status(response)
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    payload = json.loads(line)
                    if payload.get("error"):
                        raise RuntimeError(payload["error"])
                    token = payload.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if payload.get("done"):
                        break

    def _raise_for_status(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            message = self._response_error(response)
            raise RuntimeError(
                f"{response.status_code} dari Ollama untuk model {self.model}: {message}"
            ) from exc

    def _response_error(self, response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text.strip() or response.reason_phrase
        return str(payload.get("error") or payload).strip()

    async def _raise_stream_for_status(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raw = await response.aread()
            try:
                payload = json.loads(raw)
                message = str(payload.get("error") or payload).strip()
            except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                message = raw.decode("utf-8", errors="replace").strip() or response.reason_phrase
            raise RuntimeError(
                f"{response.status_code} dari Ollama untuk model {self.model}: {message}"
            ) from exc
