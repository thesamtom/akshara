from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import OCRProcessingError
from .pipeline import process_image


def main() -> int:
    # PowerShell sessions can default to cp1252, which cannot print Malayalam.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Produce a structured, reviewable Malayalam OCR draft.")
    parser.add_argument("image_path")
    parser.add_argument("--engine", choices=("gemini", "tesseract"), default="gemini")
    parser.add_argument("--tesseract-cmd")
    parser.add_argument("--tessdata-dir", help="Directory containing Malayalam mal.traineddata (normally detected automatically).")
    parser.add_argument("--output", type=Path, help="Write the complete structured result as UTF-8 JSON.")
    parser.add_argument("--text-output", type=Path, help="Write only the extracted draft text as UTF-8 plain text.")
    args = parser.parse_args()
    try:
        result = process_image(
            args.image_path,
            engine=args.engine,
            tesseract_cmd=args.tesseract_cmd,
            tessdata_dir=args.tessdata_dir,
        )
    except OCRProcessingError as error:
        parser.error(str(error))
    payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.text_output:
        args.text_output.write_text(result.raw_text + "\n", encoding="utf-8")
    if not args.output:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
