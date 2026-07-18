class OCRProcessingError(RuntimeError):
    """An image or OCR provider could not be processed safely."""


class OCRConfigurationError(OCRProcessingError):
    """An OCR engine is installed or configured incorrectly."""


class NoTextDetectedError(OCRProcessingError):
    """OCR completed but found no usable text."""
