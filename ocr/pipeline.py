"""Public orchestration API; intentionally independent of FastAPI."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .engines.base import OCREngine, OCRResult
from .engines.tesseract import TesseractEngine
from .engines.gemini import GeminiEngine
from .errors import OCRConfigurationError
from .postprocess import finalize_result
from .preprocess import PreprocessConfig, load_image, preprocess_image


def _build_engine(name: str, options: Mapping[str, object]) -> OCREngine:
    if name == "gemini":
        api_key = options.get("api_key") if isinstance(options.get("api_key"), str) else None
        client = options.get("client")
        return GeminiEngine(api_key=api_key, client=client)
    if name == "tesseract":
        return TesseractEngine(
            tesseract_cmd=options.get("tesseract_cmd") if isinstance(options.get("tesseract_cmd"), str) else None,
            tessdata_dir=options.get("tessdata_dir") if isinstance(options.get("tessdata_dir"), str) else None,
        )
    raise OCRConfigurationError(f"Unknown OCR engine '{name}'. Supported engines: 'gemini', 'tesseract'.")


import numpy as np

def process_image(image_path: str | Path | np.ndarray, engine: str = "gemini", **options: object) -> OCRResult:
    """Convert an image to a reviewable OCR draft. Raises typed OCR errors on failure."""
    config = options.get("preprocess_config")
    if config is not None and not isinstance(config, PreprocessConfig):
        raise OCRConfigurationError("preprocess_config must be a PreprocessConfig instance")
    
    if config is None:
        # Multimodal LLM models like Gemini work much better with color/grayscale photos containing original gradients and shadows.
        # Binarization destroys shadow borders and causes OCR failures on mobile photographs.
        config = PreprocessConfig(adaptive_binarize=(engine != "gemini"))

    source = load_image(image_path, config)
    prepared = preprocess_image(source, config)
    backend = _build_engine(engine, options)
    result = backend.extract(prepared)
    corrections = options.get("corrections")
    if corrections is not None and not isinstance(corrections, Mapping):
        raise OCRConfigurationError("corrections must be a mapping of reviewed substitutions")
    threshold = options.get("low_confidence_threshold", 0.75)
    if not isinstance(threshold, (int, float)):
        raise OCRConfigurationError("low_confidence_threshold must be numeric")
    return finalize_result(result, corrections=corrections, low_confidence_threshold=float(threshold))
