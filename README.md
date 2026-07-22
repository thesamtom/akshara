# Akshara OCR

Standalone, review-first Malayalam OCR powered by Gemini 2.5 Flash with client-side offline fallback.

## Install

```bash
pip install -r requirements.txt
```

To run the test suite, run:
```bash
python -m pytest
```

## Run CLI OCR

```bash
python -m ocr.cli page.jpg --engine gemini
```

To write extracted JSON and draft text directly:

```bash
python -m ocr.cli page.jpg --engine gemini --output output.json --text-output output.txt
```

## Browser App with Gemini OCR & Sarvam TTS/STT

Copy `.env.example` to `.env` and configure your API keys:

```powershell
Copy-Item .env.example .env
# Edit .env and set GEMINI_API_KEY, SARVAM_STT_API_KEY, and SARVAM_TTS_API_KEY.
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`, convert an image (uses Gemini 2.5 Flash when online, automatically falling back to client-side Tesseract WASM when offline), tap words for Sandhi-aware dictionary definitions, listen to Sarvam TTS audio, or check your reading live with Sarvam Speech-to-Text.
