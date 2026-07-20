"""Local OCR UI server with secure Sarvam TTS and live STT proxying."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
import mimetypes
from urllib.request import Request, urlopen

import websockets
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/gzip", ".gz")

class PWAStaticFiles(StaticFiles):
    def file_response(self, path: str, stat_result: os.stat_result, scope, status_code: int = 200):
        response = super().file_response(path, stat_result, scope, status_code)
        if path.endswith(".gz"):
            response.media_type = "application/gzip"
            response.headers["content-type"] = "application/gzip"
        # Prevent caching of sw.js and manifest.json to allow instant updates
        if path.endswith("sw.js") or path.endswith("manifest.json"):
            response.headers["cache-control"] = "no-cache, no-store, must-revalidate"
            response.headers["pragma"] = "no-cache"
            response.headers["expires"] = "0"
        return response

ROOT = Path(__file__).resolve().parent
SARVAM_STT_URL = (
    "wss://api.sarvam.ai/speech-to-text/ws?language-code=ml-IN"
    "&model=saaras%3Av3&mode=transcribe&sample_rate=16000&input_audio_codec=pcm_s16le"
    "&high_vad_sensitivity=true"
)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger = logging.getLogger("akshara.live_reading")


def load_local_env() -> None:
    """Load uncommitted local settings without requiring a third-party package."""
    env_file = ROOT / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


load_local_env()

import definitions
definitions.init_db()


def api_key(kind: str) -> str:
    key = os.environ.get(f"SARVAM_{kind}_API_KEY")
    if not key:
        raise HTTPException(503, f"SARVAM_{kind}_API_KEY is not configured on the server.")
    return key


@app.get("/api/health")
async def health() -> JSONResponse:
    """Allow the browser to explain missing local configuration clearly."""
    return JSONResponse({
        "sarvam_tts_configured": bool(os.environ.get("SARVAM_TTS_API_KEY")),
        "sarvam_stt_configured": bool(os.environ.get("SARVAM_STT_API_KEY")),
    })


@app.get("/api/define")
async def define_word(word: str) -> JSONResponse:
    w = word.strip()
    if not w:
        raise HTTPException(400, "Word cannot be empty.")
    try:
        result = await asyncio.to_thread(definitions.lookup_word, w)
        return JSONResponse(result)
    except Exception as error:
        logger.exception(f"Definition lookup failed for '{w}'")
        raise HTTPException(500, "Could not load word definition.")


@app.post("/api/compare_reading")
async def compare_reading(payload: dict[str, str]) -> JSONResponse:
    expected = str(payload.get("expected", "")).strip()
    spoken = str(payload.get("spoken", "")).strip()
    try:
        import backend.malayalam_sandhi
        result = await asyncio.to_thread(backend.malayalam_sandhi.align_reading_sandhi, expected, spoken)
        return JSONResponse(result)
    except Exception as error:
        logger.exception("Reading comparison failed.")
        raise HTTPException(500, "Reading comparison failed.")


def request_tts(text: str, key: str) -> str:
    body = json.dumps({
        "text": text,
        "target_language_code": "ml-IN",
        "model": "bulbul:v3",
        "speaker": "shubh",
    }).encode("utf-8")
    request = Request(
        "https://api.sarvam.ai/text-to-speech",
        data=body,
        headers={"api-subscription-key": key, "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=45) as response:
        result = json.loads(response.read())
    audio = result.get("audios", [None])[0]
    if not audio:
        raise ValueError("Sarvam returned no audio.")
    return audio


@app.post("/api/tts")
async def text_to_speech(payload: dict[str, str]) -> JSONResponse:
    text = str(payload.get("text", "")).strip()
    if not text:
        raise HTTPException(400, "Text cannot be empty.")
    if len(text) > 2500:
        raise HTTPException(400, "Text must be 2,500 characters or fewer.")
    try:
        audio = await asyncio.to_thread(request_tts, text, api_key("TTS"))
    except HTTPError as error:
        raise HTTPException(502, f"Sarvam request failed ({error.code}).") from error
    except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(502, "Sarvam speech generation failed.") from error
    return JSONResponse({"audio": audio})


@app.websocket("/ws/reading")
async def live_reading(client: WebSocket) -> None:
    await client.accept()
    key = os.environ.get("SARVAM_STT_API_KEY")
    if not key:
        await client.send_json({"type": "error", "error": "SARVAM_STT_API_KEY is not configured on the server."})
        await client.close(code=1011)
        return
    try:
        async with websockets.connect(
            SARVAM_STT_URL,
            additional_headers={"api-subscription-key": key},
            open_timeout=15,
            close_timeout=5,
        ) as sarvam:
            async def forward_transcripts() -> None:
                async for message in sarvam:
                    await client.send_text(message)

            forwarder = asyncio.create_task(forward_transcripts())
            try:
                while True:
                    message = json.loads(await client.receive_text())
                    if message.get("type") == "audio" and isinstance(message.get("data"), str):
                        await sarvam.send(json.dumps({
                            "audio": {"data": message["data"], "sample_rate": "16000", "encoding": "audio/wav"}
                        }))
                    elif message.get("type") == "flush":
                        await sarvam.send(json.dumps({"type": "flush"}))
            finally:
                forwarder.cancel()
                with __import__("contextlib").suppress(asyncio.CancelledError):
                    await forwarder
    except WebSocketDisconnect:
        return
    except Exception:
        logger.exception("Sarvam live transcription proxy failed")
        if client.client_state.name == "CONNECTED":
            await client.send_json({"type": "error", "error": "Live transcription connection failed."})
            await client.close(code=1011)


app.mount("/", PWAStaticFiles(directory=ROOT, html=True), name="static")
