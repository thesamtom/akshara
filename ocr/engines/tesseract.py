from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2

from .base import OCREngine, OCRLine, OCRResult
from ..errors import OCRConfigurationError, OCRProcessingError, NoTextDetectedError


def _tesseract_safe_path(path: Path) -> str:
    """Return an unquoted Windows short path for Tesseract's command parser."""
    resolved = str(path.resolve())
    if os.name != "nt" or " " not in resolved:
        return resolved
    try:
        import ctypes

        required = ctypes.windll.kernel32.GetShortPathNameW(resolved, None, 0)
        if required:
            buffer = ctypes.create_unicode_buffer(required)
            ctypes.windll.kernel32.GetShortPathNameW(resolved, buffer, required)
            return buffer.value
    except (AttributeError, OSError):
        pass
    return resolved


class TesseractEngine(OCREngine):
    """Free offline fallback. Requires Tesseract plus Malayalam `mal.traineddata`."""

    name = "tesseract"

    def __init__(
        self,
        tesseract_cmd: str | None = None,
        psm: int = 6,
        tessdata_dir: str | None = None,
    ):
        try:
            import pytesseract
        except ImportError as error:
            raise OCRConfigurationError("Tesseract backend requires `pytesseract`; install requirements.txt.") from error
        self.pytesseract = pytesseract
        discovered_command = shutil.which("tesseract")
        windows_default = Path(os.environ.get("ProgramFiles", r"C:\\Program Files")) / "Tesseract-OCR" / "tesseract.exe"
        if tesseract_cmd:
            self.pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        elif discovered_command:
            self.pytesseract.pytesseract.tesseract_cmd = discovered_command
        elif windows_default.is_file():
            self.pytesseract.pytesseract.tesseract_cmd = str(windows_default)
        else:
            raise OCRConfigurationError("Tesseract executable was not found; install it or pass tesseract_cmd.")
        bundled_tessdata = Path(__file__).resolve().parents[1] / "tessdata"
        selected_tessdata = Path(tessdata_dir) if tessdata_dir else bundled_tessdata
        self.tessdata_dir = selected_tessdata if (selected_tessdata / "mal.traineddata").is_file() else None
        self.language, self.psm = "mal", psm

    def extract(self, image) -> OCRResult:
        try:
            config = f'--psm {self.psm}'
            if self.tessdata_dir:
                config += f" --tessdata-dir {_tesseract_safe_path(self.tessdata_dir)}"
            data = self.pytesseract.image_to_data(image, lang=self.language, config=config, output_type=self.pytesseract.Output.DICT)
        except self.pytesseract.TesseractError as error:
            raise OCRConfigurationError(
                f"Tesseract failed. Confirm Malayalam traineddata (`{self.language}`) is installed: {error}"
            ) from error
        except OSError as error:
            raise OCRProcessingError(f"Tesseract could not run: {error}") from error

        lines: list[OCRLine] = []
        groups: dict[tuple[int, int, int], list[int]] = {}
        for index, text in enumerate(data["text"]):
            key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
            if text.strip():
                groups.setdefault(key, []).append(index)
        for indices in groups.values():
            words = [data["text"][index].strip() for index in indices]
            confidences = [float(data["conf"][index]) for index in indices if float(data["conf"][index]) >= 0]
            x1 = min(data["left"][index] for index in indices)
            y1 = min(data["top"][index] for index in indices)
            x2 = max(data["left"][index] + data["width"][index] for index in indices)
            y2 = max(data["top"][index] + data["height"][index] for index in indices)
            lines.append(OCRLine(" ".join(words), sum(confidences) / (100 * len(confidences)) if confidences else None, (x1, y1, x2, y2)))
        lines.sort(key=lambda line: (line.bbox[1], line.bbox[0]) if line.bbox else (0, 0))
        if not lines:
            raise NoTextDetectedError("Tesseract detected no text in this image.")
        return OCRResult(raw_text="\n".join(line.text for line in lines), lines=lines, paragraphs=[], engine_used=self.name)
