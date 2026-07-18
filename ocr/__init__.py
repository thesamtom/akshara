"""Structured Malayalam OCR for Akshara."""

from .engines.base import OCRLine, OCRParagraph, OCRResult
from .pipeline import process_image

__all__ = ["OCRLine", "OCRParagraph", "OCRResult", "process_image"]
