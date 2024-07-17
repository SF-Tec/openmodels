"""
Custom exceptions for the OpenModels library.

This module defines custom exception classes used throughout the OpenModels
library to provide more specific error handling and improve debuggability.
"""


class OpenModelsError(Exception):
    """Base exception class for all OpenModels library errors."""


class SerializationError(OpenModelsError):
    """Exception raised when there's an error during model serialization."""


class DeserializationError(OpenModelsError):
    """Exception raised when there's an error during model deserialization."""


class UnsupportedFormatError(OpenModelsError):
    """Exception raised when an unsupported serialization format is requested."""


class UnsupportedEstimatorError(OpenModelsError):
    """Exception raised when an unsupported estimator is encountered."""
