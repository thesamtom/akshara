"""Conservative preprocessing for photographed Malayalam print."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .errors import OCRProcessingError

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PreprocessConfig:
    max_file_size_bytes: int = 20 * 1024 * 1024
    min_dimension: int = 1400
    adaptive_block_size: int = 31
    adaptive_c: int = 11


def load_image(image_path: str | Path, config: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    path = Path(image_path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise OCRProcessingError(f"Unsupported image format: {path.suffix or 'no extension'}")
    if not path.is_file():
        raise OCRProcessingError(f"Image file does not exist: {path}")
    if path.stat().st_size > config.max_file_size_bytes:
        raise OCRProcessingError(f"Image exceeds {config.max_file_size_bytes // (1024 * 1024)} MiB limit")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise OCRProcessingError("Image is unreadable or contains no raster data")
    return image


def resize_if_needed(image: np.ndarray, min_dimension: int) -> np.ndarray:
    height, width = image.shape[:2]
    shortest = min(height, width)
    if shortest >= min_dimension:
        return image
    scale = min_dimension / shortest
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_CUBIC)


def estimate_skew_angle(gray: np.ndarray) -> float:
    # Text strokes become foreground after inversion; min-area rectangle estimates baseline angle.
    foreground = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    points = np.column_stack(np.where(foreground > 0))
    if len(points) < 100:
        return 0.0
    angle = cv2.minAreaRect(points.astype(np.float32))[2]
    return -(90 + angle) if angle < -45 else -angle


def deskew(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = estimate_skew_angle(gray)
    if abs(angle) < 0.15 or abs(angle) > 15:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess_image(image: np.ndarray, config: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    """Return a deskewed, denoised, adaptively binarized BGR image."""
    enlarged = resize_if_needed(image, config.min_dimension)
    straightened = deskew(enlarged)
    gray = cv2.cvtColor(straightened, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
    block = max(3, config.adaptive_block_size | 1)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, config.adaptive_c
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
