# Akshara OCR

Standalone, review-first Malayalam OCR using Tesseract offline.

## Install

```bash
pip install -r requirements.txt
```

To run the test suite, install `pip install -r requirements-dev.txt` and run `python -m pytest -q`.

Install Tesseract. The module uses its bundled Malayalam (`mal`) language data automatically.

## Run

```bash
python -m ocr.cli page.jpg --engine tesseract
```

To avoid Windows PowerShell pipeline encoding issues, have the CLI write files directly:

```bash
python -m ocr.cli page.jpg --engine tesseract --output output.json --text-output output.txt
```

The module processes Malayalam with its bundled `mal` language data.

## Browser app with Sarvam TTS

The browser app converts Malayalam images locally and reads extracted text through Sarvam TTS. Copy `.env.example` to `.env`, add newly generated keys once, and never put them in `index.html` or commit them.

```powershell
Copy-Item .env.example .env
# Edit .env and set SARVAM_STT_API_KEY and SARVAM_TTS_API_KEY.
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`, convert an image, then select **Play text**. The local server sends Malayalam text to Sarvam using `ml-IN` and returns the resulting audio to the page.

Select **Start Reading** to grant microphone access. The page streams 16 kHz microphone chunks through the local server to Sarvam speech-to-text with `ml-IN`, so word checking updates while the learner reads. Select **Stop Reading** to flush the final transcript. Words that have been passed and do not match are marked red.

For the API contract, engine decision, requirements, risks, and acceptance criteria, see [ocr/PRD.md](ocr/PRD.md).
