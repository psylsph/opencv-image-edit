"""Application exception hierarchy."""
from __future__ import annotations


class AppError(Exception):
    """Base for all application errors."""

    status_code: int = 500

    def __init__(self, message: str = "") -> None:
        super().__init__(message)
        self.message = message or self.__class__.__name__


class DecodeError(AppError):
    """Failed to decode image bytes."""

    status_code = 400


class EncodeError(AppError):
    """Failed to encode image to bytes."""

    status_code = 500


class ProcessingError(AppError):
    """Image processing pipeline failed."""

    status_code = 500


class ValidationError(AppError):
    """Request validation failed."""

    status_code = 422


class ModelNotFoundError(AppError):
    """Required model file is missing."""

    status_code = 503
