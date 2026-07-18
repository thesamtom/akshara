"""Public orchestration API; intentionally independent of FastAPI."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .engines.base import OCREngine, OCRResult
from .engines.tesseract import TesseractEngine
from .errors import OCRConfigurationError
from .postprocess import finalize_result
from .preprocess import PreprocessConfig, load_image, preprocess_image


def _build_engine(name: str, options: Mapping[str, object]) -> OCREngine:
    if name == "tesseract":
        return TesseractEngine(
            tesseract_cmd=options.get("tesseract_cmd") if isinstance(options.get("tesseract_cmd"), str) else None,
            tessdata_dir=options.get("tessdata_dir") if isinstance(options.get("tessdata_dir"), str) else None,
        )
    raise OCRConfigurationError(f"Unknown OCR engine '{name}'. Tesseract is the only supported engine.")


def process_image(image_path: str | Path, engine: str = "tesseract", **options: object) -> OCRResult:
    """Convert an image to a reviewable OCR draft. Raises typed OCR errors on failure."""
    config = options.get("preprocess_config")
    if config is not None and not isinstance(config, PreprocessConfig):
        raise OCRConfigurationError("preprocess_config must be a PreprocessConfig instance")
    source = load_image(image_path, config or PreprocessConfig())
    prepared = preprocess_image(source, config or PreprocessConfig())
    backend = _build_engine(engine, options)
    result = backend.extract(prepared)
    corrections = options.get("corrections")
    if corrections is not None and not isinstance(corrections, Mapping):
        raise OCRConfigurationError("corrections must be a mapping of reviewed substitutions")
    threshold = options.get("low_confidence_threshold", 0.75)
    if not isinstance(threshold, (int, float)):
        raise OCRConfigurationError("low_confidence_threshold must be numeric")
    return finalize_result(result, corrections=corrections, low_confidence_threshold=float(threshold))
