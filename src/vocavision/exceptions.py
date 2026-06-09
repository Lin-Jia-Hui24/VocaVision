"""Custom exceptions for the VocaVision pipeline."""


class VocaVisionError(Exception):
    """Base exception for all project-specific errors."""


class ConfigurationError(VocaVisionError):
    """Raised when runtime configuration is missing or invalid."""


class ExternalServiceError(VocaVisionError):
    """Raised when a third-party API returns an invalid response."""


class JsonResponseError(VocaVisionError):
    """Raised when structured model output cannot be parsed."""


class CommandExecutionError(VocaVisionError):
    """Raised when FFmpeg or FFprobe execution fails."""
