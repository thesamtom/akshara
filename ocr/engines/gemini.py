"""Gemini 2.5 Flash Cloud Vision OCR Engine."""
from __future__ import annotations

import io
import os
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .base import OCREngine, OCRLine, OCRParagraph, OCRResult
from ..errors import OCRConfigurationError, OCRProcessingError, NoTextDetectedError

PROMPT = (
    "Act as an expert Malayalam OCR engine specialized in deciphering degraded, blurry, "
    "or low-contrast print text. Carefully analyze the visual letterforms and transcribe "
    "the exact Malayalam text from this image accurately. Preserve line breaks and paragraph structure. "
    "Do not translate or invent non-existent words."
)


def _load_local_env() -> None:
    from pathlib import Path
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


class GeminiEngine(OCREngine):
    """Gemini 2.5 Flash cloud vision OCR engine with automatic 429 rate-limit handling."""

    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash", client: Any = None):
        if api_key is None and "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in os.environ:
            _load_local_env()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model = model

        if client is not None:
            self.client = client
        else:
            if not self.api_key:
                raise OCRConfigurationError("GEMINI_API_KEY is not configured in environment variables.")
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as error:
                raise OCRConfigurationError(f"Failed to initialize google-genai client: {error}") from error

    def extract(self, image: Any) -> OCRResult:
        if not self.api_key and getattr(self, "client", None) is None:
            raise OCRConfigurationError("GEMINI_API_KEY is not configured in environment variables.")

        try:
            if isinstance(image, (str, os.PathLike)):
                pil_image = Image.open(image)
            elif isinstance(image, np.ndarray):
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 and image.shape[2] == 3 else image
                pil_image = Image.fromarray(rgb)
            elif isinstance(image, Image.Image):
                pil_image = image
            elif isinstance(image, bytes):
                pil_image = Image.open(io.BytesIO(image))
            else:
                raise OCRProcessingError(f"Unsupported image input type: {type(image)}")

            buf = io.BytesIO()
            pil_image.save(buf, format="JPEG")
            img_bytes = buf.getvalue()
        except Exception as error:
            if isinstance(error, (OCRConfigurationError, OCRProcessingError, NoTextDetectedError)):
                raise
            raise OCRProcessingError(f"Malformed or unsupported image input: {error}") from error

        try:
            from google.genai import types
            contents = [types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"), PROMPT]
            gen_config = types.GenerateContentConfig(temperature=0.1)
        except Exception:
            contents = [pil_image, PROMPT]
            gen_config = None

        max_attempts = 1
        last_exception = None
        response_text = ""

        candidate_models = [self.model]
        for fallback in ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        for model_name in candidate_models:
            for attempt in range(1, max_attempts + 1):
                try:
                    kwargs = {"model": model_name, "contents": contents}
                    if gen_config is not None:
                        kwargs["config"] = gen_config
                    response = self.client.models.generate_content(**kwargs)
                    response_text = (getattr(response, "text", "") or "").strip()
                    if response_text:
                        break
                except Exception as error:
                    last_exception = error
                    err_msg = str(error).lower()
                    if "429" in err_msg or "resource_exhausted" in err_msg or "rate limit" in err_msg:
                        # Smart rate-limit retry delay
                        wait_sec = 2.0 * (2 ** (attempt - 1))
                        time.sleep(wait_sec)
                    elif attempt < max_attempts:
                        time.sleep(0.5)
            if response_text:
                break

        if not response_text and last_exception:
            raise OCRProcessingError(f"Gemini OCR API request failed: {last_exception}")

        if not response_text:
            raise NoTextDetectedError("Gemini OCR detected no text in this image.")

        raw_lines = [line.strip() for line in response_text.splitlines() if line.strip()]
        lines = [OCRLine(text=line, confidence=0.99) for line in raw_lines]
        paragraphs = [OCRParagraph(text=response_text, lines=lines)]

        return OCRResult(
            raw_text=response_text,
            lines=lines,
            paragraphs=paragraphs,
            engine_used=self.name,
            is_draft=False
        )
