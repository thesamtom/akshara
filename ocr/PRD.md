# Akshara Malayalam OCR Module — Product Requirements Document

## Overview

This module is the image-to-text entry point for Akshara. It turns a photographed or scanned Malayalam textbook/storybook page into reviewable, structured Malayalam Unicode text before downstream paragraph/word segmentation, reading support, and speech features run. Its output is explicitly a draft for a parent or teacher to correct.

## Goals and non-goals

**Goals**

- Accept a supported image file and return NFC-normalized Malayalam text with line and paragraph structure.
- Preserve line bounding boxes and per-line confidence where the OCR backend exposes them.
- Improve phone-photo inputs through deskewing, denoising, adaptive binarization, and conditional upscaling.
- Provide a self-contained, offline Tesseract OCR backend through one Python API.
- Support printed Malayalam (`mal`) Tesseract extraction.
- Surface low confidence and failures clearly for the review UI.

**Non-goals**

- Word/grapheme segmentation, dyslexia analysis, speech alignment, and text-to-speech are separate modules.
- The module does not claim to produce final or semantically corrected text.
- It does not perform perspective dewarping, handwriting recognition, document storage, or HTTP/FastAPI routing.

## Input contract

`process_image(image_path: str | Path, engine: str = "tesseract", **options)` accepts:

- Local `jpg`, `jpeg`, `png`, `bmp`, `tif`, `tiff`, or `webp` image files.
- A readable raster image no larger than 20 MiB by default (`max_file_size_bytes` is configurable).
- Engine option: `tesseract_cmd` (optional executable path).
- Tesseract language: Malayalam (`mal`).

Unsupported, unreadable, oversized, or image-less inputs raise a clear `OCRProcessingError`.

## Output contract

`OCRResult.to_dict()` returns JSON-serializable data:

```json
{
  "raw_text": "NFC-normalized draft text",
  "paragraphs": [{"text": "line one\nline two", "lines": [{"text": "line one", "confidence": 0.94, "bbox": [0, 0, 100, 20], "low_confidence": false}]}],
  "lines": [{"text": "line one", "confidence": 0.94, "bbox": [0, 0, 100, 20], "low_confidence": false}],
  "engine_used": "tesseract",
  "warnings": ["Low confidence on line 3"],
  "is_draft": true
}
```

Bounding boxes are `[x1, y1, x2, y2]` in preprocessed-image pixels. Confidence may be `null` where a backend does not provide it. Paragraphs are groups of spatially adjacent lines, or blank-line groups when only text structure is available.

## Engine decision

The sole backend is **Tesseract** (`pytesseract` with Malayalam `mal` traineddata). It is free, offline, and does not require credentials, billing, or network access. The module bundles the Malayalam model.

Tesseract's Python integration and cost profile suit this project, but Malayalam conjunct, chillu/virama, font, rotation, and noisy-photo accuracy must be validated against Akshara's textbook corpus. PaddleOCR was considered but is not selected because its Malayalam model quality/version support needs a corpus benchmark.

## Preprocessing requirements

- The system must validate and decode the image before OCR.
- It must estimate and correct modest text rotation before OCR.
- It must denoise without excessive smoothing of dense Malayalam glyph edges.
- It must use adaptive, rather than global, thresholding to compensate for uneven lighting.
- It must upscale low-resolution images before OCR while avoiding unnecessary enlargement of already legible inputs.
- The original source is retained only by the caller; this module processes in memory and writes no image files by default.

## Malayalam text requirements

- All returned text must be NFC-normalized, including vowel signs, chillu characters, and virama sequences.
- The module must not aggressively substitute ambiguous Malayalam conjuncts. An editable correction map may apply only on low-confidence content, and every such change must create a warning.
- Low-confidence lines must remain in the output and be marked for human review.

## Error and edge cases

| Condition | Expected behavior |
| --- | --- |
| Unsupported/unreadable/oversized image | Raise `OCRProcessingError` with a user-safe reason. |
| No text found | Raise `NoTextDetectedError`; do not return an empty success payload. |
| Missing Tesseract executable or language data | Raise `OCRConfigurationError` explaining installation/configuration. |
| Very low confidence | Return the draft result and prominent warnings; do not discard or silently alter it. |
| Engines without confidence | Return `null` confidence and a review warning, rather than inventing a score. |

## Acceptance criteria

- [ ] A straight, well-lit Malayalam sample produces non-empty structured text.
- [ ] A sample rotated within ±10° is deskewed before extraction and produces structured text.
- [ ] A low-light/noisy sample traverses denoise and adaptive-threshold preprocessing without failure.
- [ ] The result is JSON serializable and conforms to the output contract, not a flat string.
- [ ] An unreadable image and a no-text image fail with clear typed errors and no crash.
- [ ] Low-confidence lines remain visible and are flagged rather than silently dropped or aggressively corrected.
- [ ] Tesseract implements the module's OCR engine interface.

The automated test suite verifies module behavior and preprocessing with synthetic rotated/noisy images. Final OCR-accuracy checks above require real Malayalam textbook photographs and configured OCR credentials/language data; those remain validation work, not claims of achieved accuracy.

## Open questions and risks

- Tesseract Malayalam accuracy on Akshara's target conjunct-heavy textbook fonts is unverified until corpus testing.
- Perspective distortion/page curvature may require a future dewarping stage if deskew alone is insufficient.
- The editable Malayalam confusion map should be curated from reviewed production errors, not guessed in advance.
