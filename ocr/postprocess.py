"""Unicode-safe text cleanup and structural grouping."""
from __future__ import annotations

import unicodedata
from collections.abc import Mapping

from .engines.base import OCRLine, OCRParagraph, OCRResult

DEFAULT_LOW_CONFIDENCE = 0.75


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n").strip()


def apply_low_confidence_corrections(line: OCRLine, corrections: Mapping[str, str] | None = None) -> tuple[OCRLine, list[str]]:
    """Apply only explicitly supplied reviewed corrections; never guess Malayalam glyphs."""
    line.text = normalize_text(line.text)
    warnings: list[str] = []
    if line.low_confidence and corrections:
        for wrong, replacement in corrections.items():
            if wrong in line.text:
                line.text = line.text.replace(wrong, replacement)
                warnings.append("A reviewed low-confidence correction was applied; verify this line.")
    return line, warnings


def group_paragraphs(lines: list[OCRLine]) -> list[OCRParagraph]:
    if not lines:
        return []
    groups: list[list[OCRLine]] = [[]]
    previous: OCRLine | None = None
    for line in lines:
        starts_new = not line.text
        if previous and previous.bbox and line.bbox:
            previous_height = max(1, previous.bbox[3] - previous.bbox[1])
            vertical_gap = line.bbox[1] - previous.bbox[3]
            starts_new = starts_new or vertical_gap > previous_height * 1.8
        if starts_new and groups[-1]:
            groups.append([])
        if line.text:
            groups[-1].append(line)
        previous = line
    return [OCRParagraph(text="\n".join(item.text for item in group), lines=group) for group in groups if group]


def finalize_result(result: OCRResult, corrections: Mapping[str, str] | None = None, low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE) -> OCRResult:
    warnings = list(result.warnings)
    if result.lines and all(line.confidence is None for line in result.lines):
        warnings.append("The selected OCR engine did not provide confidence scores; review all text carefully.")
    final_lines: list[OCRLine] = []
    for index, line in enumerate(result.lines, start=1):
        line.low_confidence = line.confidence is not None and line.confidence < low_confidence_threshold
        line, changes = apply_low_confidence_corrections(line, corrections)
        final_lines.append(line)
        if line.low_confidence:
            confidence = f" ({line.confidence:.2f})" if line.confidence is not None else ""
            warnings.append(f"Low confidence on line {index}{confidence}; human review required.")
        warnings.extend(changes)
    result.lines = final_lines
    result.paragraphs = group_paragraphs(final_lines)
    result.raw_text = "\n\n".join(paragraph.text for paragraph in result.paragraphs) or normalize_text(result.raw_text)
    result.warnings = warnings
    return result
