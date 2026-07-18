"""Backend-neutral OCR types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class OCRLine:
    text: str
    confidence: float | None = None
    bbox: tuple[int, int, int, int] | None = None
    low_confidence: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["bbox"] = list(self.bbox) if self.bbox else None
        return value


@dataclass
class OCRParagraph:
    text: str
    lines: list[OCRLine] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "lines": [line.to_dict() for line in self.lines]}


@dataclass
class OCRResult:
    raw_text: str
    lines: list[OCRLine]
    paragraphs: list[OCRParagraph]
    engine_used: str
    warnings: list[str] = field(default_factory=list)
    is_draft: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "paragraphs": [paragraph.to_dict() for paragraph in self.paragraphs],
            "lines": [line.to_dict() for line in self.lines],
            "engine_used": self.engine_used,
            "warnings": self.warnings,
            "is_draft": self.is_draft,
        }


class OCREngine(ABC):
    """An OCR provider receiving a preprocessed BGR OpenCV image."""

    name: str

    @abstractmethod
    def extract(self, image: Any) -> OCRResult:
        raise NotImplementedError
