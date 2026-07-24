from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from ocr.engines.base import OCRLine, OCRResult
from ocr.engines.gemini import GeminiEngine
from ocr.errors import NoTextDetectedError, OCRConfigurationError, OCRProcessingError
from ocr.pipeline import process_image
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


def test_gemini_engine_extract_success() -> None:
    mock_response = MagicMock()
    mock_payload = {
        "data": [
            {
                "text_detections": [
                    {
                        "text_prediction": {
                            "text": "മലയാളം വായന",
                            "confidence": 0.95
                        },
                        "bounding_box": {
                            "points": [
                                {"x": 0.1, "y": 0.1},
                                {"x": 0.5, "y": 0.1},
                                {"x": 0.5, "y": 0.2},
                                {"x": 0.1, "y": 0.2}
                            ]
                        }
                    }
                ]
            }
        ]
    }
    mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")
    
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        engine = GeminiEngine(api_key="fake-key")
        fake_img = np.full((100, 100, 3), 255, dtype=np.uint8)
        res = engine.extract(fake_img)
        
        assert res.raw_text == "മലയാളം വായന"
        assert res.engine_used == "gemini"
        assert len(res.lines) == 1
        assert res.lines[0].bbox == (10, 10, 50, 20)
        mock_urlopen.assert_called_once()


def test_gemini_engine_missing_api_key(monkeypatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("ocr.engines.gemini._load_local_env", lambda: None)
    with pytest.raises(OCRConfigurationError, match="NVIDIA_API_KEY"):
        GeminiEngine()


def test_gemini_engine_retry_on_failure() -> None:
    import urllib.error
    mock_response = MagicMock()
    mock_payload = {
        "data": [
            {
                "text_detections": [
                    {
                        "text_prediction": {
                            "text": "വിജയം",
                            "confidence": 0.98
                        }
                    }
                ]
            }
        ]
    }
    mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")

    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen") as mock_urlopen:
        # Raise errors twice, then succeed
        mock_urlopen.side_effect = [
            urllib.error.HTTPError("http://foo", 429, "Too Many Requests", {}, None),
            urllib.error.URLError("Timeout"),
            mock_context
        ]
        
        with patch("time.sleep") as mock_sleep:  # Mock sleep to run tests instantly
            engine = GeminiEngine(api_key="fake-key")
            fake_img = np.full((100, 100, 3), 255, dtype=np.uint8)
            res = engine.extract(fake_img)
            
            assert res.raw_text == "വിജയം"
            assert mock_urlopen.call_count == 3


def test_gemini_engine_no_text_detected() -> None:
    mock_response = MagicMock()
    mock_payload = {
        "data": [
            {
                "text_detections": []
            }
        ]
    }
    mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        engine = GeminiEngine(api_key="fake-key")
        fake_img = np.full((100, 100, 3), 255, dtype=np.uint8)
        with pytest.raises(NoTextDetectedError):
            engine.extract(fake_img)


def test_pipeline_disables_binarization_for_gemini() -> None:
    # Create a mock color image
    fake_img = np.full((1500, 1500, 3), 200, dtype=np.uint8)
    cv2.circle(fake_img, (750, 750), 100, (50, 100, 150), -1)

    with patch("ocr.pipeline._build_engine") as mock_build:
        mock_engine = MagicMock()
        mock_engine.extract.return_value = OCRResult(
            raw_text="ടെസ്റ്റ്",
            lines=[],
            paragraphs=[],
            engine_used="gemini"
        )
        mock_build.return_value = mock_engine
        
        process_image(fake_img, engine="gemini")
        
        # Check that the image passed to extract has color/grayscale shades (not binarized)
        mock_build.assert_called_once()
        prepared_img = mock_engine.extract.call_args[0][0]
        # Binarized image only has 0 and 255. Colored/optimized will have multiple distinct color values.
        assert len(np.unique(prepared_img)) > 2
