from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from ocr.engines.base import OCRLine, OCRResult
from ocr.errors import NoTextDetectedError, OCRProcessingError
from ocr.postprocess import finalize_result
from ocr.preprocess import PreprocessConfig, load_image, preprocess_image


def test_preprocess_handles_rotated_noisy_page() -> None:
    page = np.full((500, 900, 3), 255, dtype=np.uint8)
    cv2.putText(page, "Malayalam OCR", (80, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    rotation = cv2.getRotationMatrix2D((450, 250), 8, 1)
    page = cv2.warpAffine(page, rotation, (900, 500), borderValue=(255, 255, 255))
    noise = np.random.default_rng(1).normal(0, 12, page.shape).astype(np.int16)
    noisy = np.clip(page.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    result = preprocess_image(noisy, PreprocessConfig(min_dimension=500))
    assert result.shape == noisy.shape
    assert result.dtype == np.uint8
    assert len(np.unique(result)) <= 2  # adaptive binarization output


def test_unreadable_file_has_clear_error(tmp_path) -> None:
    bad = tmp_path / "not-an-image.png"
    bad.write_text("not an image", encoding="utf-8")
    with pytest.raises(OCRProcessingError, match="unreadable"):
        load_image(bad)


def test_unsupported_file_has_clear_error(tmp_path) -> None:
    unsupported = tmp_path / "page.pdf"
    unsupported.write_bytes(b"%PDF")
    with pytest.raises(OCRProcessingError, match="Unsupported"):
        load_image(unsupported)


def test_result_is_normalized_structured_json_and_flags_low_confidence() -> None:
    # Decomposed e + acute is a simple NFC normalization sentinel independent of OCR fonts.
    result = OCRResult(
        raw_text="e\u0301",
        lines=[OCRLine("e\u0301", confidence=0.40, bbox=(0, 0, 10, 10))],
        paragraphs=[],
        engine_used="fake",
    )
    finalized = finalize_result(result)
    payload = finalized.to_dict()
    assert payload["raw_text"] == "é"
    assert payload["lines"][0]["low_confidence"] is True
    assert "Low confidence on line 1" in payload["warnings"][0]
    assert json.loads(json.dumps(payload)) == payload


def test_no_text_error_is_typed() -> None:
    with pytest.raises(NoTextDetectedError):
        raise NoTextDetectedError("No text")
