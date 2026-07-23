from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from server import app

client = TestClient(app)

def test_tutor_feedback_endpoint_fallback_without_keys() -> None:
    # Test fallback when API keys are not in environment
    with patch.dict("os.environ", {"OPENAI_API_KEY": "", "SARVAM_TTS_API_KEY": ""}):
        payload = {
            "totalWords": 10,
            "correctWords": 8,
            "partialWords": 1,
            "incorrectWords": 1,
            "accuracy": 85,
            "readingTime": "12s",
            "averageSpeed": "50 WPM",
            "incorrectWordList": ["test"],
            "partialWordList": ["word"],
            "observations": ["Some issues"]
        }
        response = client.post("/api/tutor_feedback", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "feedback_text" in data
        assert "85%" in data["feedback_text"]
        assert data["audio_base64"] == ""


def test_tutor_feedback_endpoint_success() -> None:
    # Test successful flow when keys are present and mock API calls
    mock_openai_response = MagicMock()
    mock_openai_response.__enter__.return_value = mock_openai_response
    mock_openai_response.read.return_value = json.dumps({
        "choices": [{
            "message": {
                "content": "Encouraging feedback from tutor."
            }
        }]
    }).encode("utf-8")

    mock_sarvam_response = MagicMock()
    mock_sarvam_response.__enter__.return_value = mock_sarvam_response
    mock_sarvam_response.read.return_value = json.dumps({
        "audios": ["dGVzdF9hdWRpb19iYXNlNjQ="]
    }).encode("utf-8")

    with patch("server.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = [mock_openai_response, mock_sarvam_response]
        
        with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-openai-key", "SARVAM_TTS_API_KEY": "fake-sarvam-key"}):
            payload = {
                "totalWords": 10,
                "correctWords": 8,
                "partialWords": 1,
                "incorrectWords": 1,
                "accuracy": 85,
                "readingTime": "12s",
                "averageSpeed": "50 WPM",
                "incorrectWordList": ["test"],
                "partialWordList": ["word"],
                "observations": ["Some issues"]
            }
            response = client.post("/api/tutor_feedback", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["feedback_text"] == "Encouraging feedback from tutor."
            assert data["audio_base64"] == "dGVzdF9hdWRpb19iYXNlNjQ="
