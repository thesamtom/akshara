"""Local OCR UI server with secure Sarvam TTS and live STT proxying."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
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


from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/gzip", ".gz")
def clean_text_with_openai(text: str) -> str:
    """Clean OCR text using OpenAI GPT-4o-mini to remove gibberish and artifacts."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return text

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    system_prompt = (
        "You are an expert editor specializing in refining Malayalam OCR outputs. "
        "Your task is to clean up the provided Malayalam text by removing visual gibberish, artifacts, noise characters, "
        "and fixing obvious OCR typos/broken letterforms.\n\n"
        "Ensure that:\n"
        "1. The output contains ONLY the cleaned Malayalam text. Do not add any introduction, explanation, comments, or formatting wrappers.\n"
        "2. Format the output as a single continuous sentence/line. Remove all line breaks and join the words with spaces to form a single continuous line of text.\n"
        "3. Strictly preserve the original meaning and words of the source text.\n"
        "4. Do not translate the text.\n"
        "5. Remove any non-Malayalam characters that are clearly OCR noise or artifacts, but keep any legitimate Malayalam words/punctuation."
    )
    body = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": 0.1,
    }).encode("utf-8")
    
    try:
        request = Request(
            url,
            data=body,
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        cleaned = result["choices"][0]["message"]["content"].strip()
        if cleaned:
            return cleaned
    except Exception as error:
        logger.warning(f"OpenAI transcription cleanup failed: {error}. Falling back to raw text.")
    return text


@app.get("/api/health")
async def health() -> JSONResponse:
    """Allow the browser to explain missing local configuration clearly."""
    return JSONResponse({
        "sarvam_tts_configured": bool(os.environ.get("SARVAM_TTS_API_KEY")),
        "sarvam_stt_configured": bool(os.environ.get("SARVAM_STT_API_KEY")),
        "gemini_ocr_configured": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
        "openai_cleanup_configured": bool(os.environ.get("OPENAI_API_KEY")),
    })


@app.post("/api/ocr")
async def process_ocr_image(file: UploadFile = File(...)) -> JSONResponse:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise HTTPException(503, "GEMINI_API_KEY is not configured on the server. Add GEMINI_API_KEY to your .env file.")

    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Uploaded image file is empty.")

    def run_gemini_ocr(image_bytes: bytes) -> dict:
        import io
        from PIL import Image, ImageOps
        import cv2
        import numpy as np
        import ocr.pipeline
        from ocr.errors import OCRConfigurationError, OCRProcessingError, NoTextDetectedError

        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Invalid image file format: {e}")

        try:
            res = ocr.pipeline.process_image(img, engine="gemini")
            return res.to_dict()
        except NoTextDetectedError:
            return {"raw_text": "", "paragraphs": [], "lines": [], "warnings": ["No text detected in image."]}

    try:
        result = await asyncio.to_thread(run_gemini_ocr, contents)
        if result.get("raw_text") and os.environ.get("OPENAI_API_KEY"):
            cleaned_text = await asyncio.to_thread(clean_text_with_openai, result["raw_text"])
            if cleaned_text != result["raw_text"]:
                result["raw_text"] = cleaned_text
                from ocr.engines.base import OCRLine, OCRParagraph
                raw_lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
                lines_objs = [OCRLine(text=line, confidence=0.99) for line in raw_lines]
                paragraph_obj = OCRParagraph(text=cleaned_text, lines=lines_objs)
                result["lines"] = [line.to_dict() for line in lines_objs]
                result["paragraphs"] = [paragraph_obj.to_dict()]
        return JSONResponse(result)
    except ValueError as err:
        raise HTTPException(400, str(err))
    except Exception as error:
        logger.exception("Gemini OCR extraction failed.")
        raise HTTPException(502, f"Gemini OCR processing failed: {error}")


@app.post("/api/clean")
async def clean_text_endpoint(payload: dict[str, str]) -> JSONResponse:
    text = payload.get("text", "").strip()
    if not text:
        return JSONResponse({"cleaned_text": ""})
    if not os.environ.get("OPENAI_API_KEY"):
        return JSONResponse({"cleaned_text": text})
    try:
        cleaned_text = await asyncio.to_thread(clean_text_with_openai, text)
        return JSONResponse({"cleaned_text": cleaned_text})
    except Exception as error:
        logger.warning(f"Failed to clean text: {error}")
        return JSONResponse({"cleaned_text": text})


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


@app.post("/api/tutor_feedback")
async def tutor_feedback(payload: dict) -> JSONResponse:
    total_words = payload.get("totalWords", 0)
    correct_words = payload.get("correctWords", 0)
    partial_words = payload.get("partialWords", 0)
    incorrect_words = payload.get("incorrectWords", 0)
    accuracy = payload.get("accuracy", 0)
    reading_time = payload.get("readingTime", "")
    average_speed = payload.get("averageSpeed", "")
    incorrect_word_list = payload.get("incorrectWordList", [])
    partial_word_list = payload.get("partialWordList", [])
    observations = payload.get("observations", [])

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        feedback_text = (
            f"Great effort today! You read with {accuracy}% accuracy in {reading_time}. "
            f"You got {correct_words} words correct. Keep practicing to improve further!"
        )
    else:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_key}",
        }
        
        system_prompt = (
            "You are an experienced and encouraging reading tutor for children with dyslexia.\n"
            "Your job is to analyze the student's reading report and provide supportive, personalized feedback.\n"
            "Your feedback should:\n"
            "- Start with positive encouragement.\n"
            "- Mention what the learner did well.\n"
            "- Explain the most common pronunciation mistakes.\n"
            "- Identify any reading patterns observed.\n"
            "- Suggest practical techniques to improve.\n"
            "- Recommend which types of words should be practiced.\n"
            "- End with a short motivational message.\n\n"
            "Keep the response:\n"
            "- Friendly, supportive, positive, and easy to understand.\n"
            "- Around 120–180 words.\n"
            "- Never discourage or criticize the learner.\n\n"
            "Avoid phrases like 'You failed', 'You performed poorly', 'Wrong pronunciation'.\n"
            "Instead use encouraging language such as 'Let's practice this together.', 'You're improving.', 'With a little more practice...', 'Great effort today.'"
        )
        
        user_prompt = json.dumps({
            "totalWords": total_words,
            "correctWords": correct_words,
            "partialWords": partial_words,
            "incorrectWords": incorrect_words,
            "accuracy": accuracy,
            "readingTime": reading_time,
            "averageSpeed": average_speed,
            "incorrectWordList": incorrect_word_list,
            "partialWordList": partial_word_list,
            "observations": observations
        }, ensure_ascii=False)
        
        body = json.dumps({
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
        }).encode("utf-8")
        
        try:
            request = Request(
                url,
                data=body,
                headers=headers,
                method="POST",
            )
            with urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
            feedback_text = result["choices"][0]["message"]["content"].strip()
        except Exception as error:
            logger.warning(f"Failed to generate tutor feedback from OpenAI: {error}")
            feedback_text = (
                f"Great effort today! You read with {accuracy}% accuracy in {reading_time}. "
                f"You got {correct_words} words correct. Keep practicing to improve further!"
            )
            
    audio_base64 = ""
    sarvam_key = os.environ.get("SARVAM_TTS_API_KEY")
    if sarvam_key:
        try:
            audio_base64 = await asyncio.to_thread(request_tts, feedback_text, sarvam_key)
        except Exception as error:
            logger.warning(f"Failed to generate TTS for tutor feedback: {error}")

    return JSONResponse({
        "feedback_text": feedback_text,
        "audio_base64": audio_base64
    })


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
            open_timeout=10,
            close_timeout=3,
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
    except (socket.gaierror, OSError, TimeoutError, websockets.exceptions.WebSocketException) as error:
        logger.warning(f"Sarvam STT network lookup/connection failed: {error}")
        if client.client_state.name == "CONNECTED":
            await client.send_json({"type": "error", "error": "Unable to reach Sarvam STT service. Please check connection."})
            await client.close(code=1011)
    except Exception:
        logger.exception("Sarvam live transcription proxy failed")
        if client.client_state.name == "CONNECTED":
            await client.send_json({"type": "error", "error": "Live transcription connection failed."})
            await client.close(code=1011)


app.mount("/", PWAStaticFiles(directory=ROOT, html=True), name="static")
