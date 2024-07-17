"""
Format registry module for the OpenModels library.

This module provides a registry for format converters, allowing dynamic
registration and retrieval of converters for different serialization formats.
"""

from typing import Dict, Type
from .protocols import FormatConverter
from .exceptions import UnsupportedFormatError


class FormatRegistry:
    """
    A registry for format converters.

    This class manages the registration and retrieval of format converters,
    allowing the SerializationManager to support multiple serialization formats.

    Attributes
    ----------
    _converters : Dict[str, Type[FormatConverter]]
        A dictionary mapping format names to their respective converter classes.
    """

    _converters: Dict[str, Type[FormatConverter]] = {}

    @classmethod
    def register(cls, format_name: str, converter: Type[FormatConverter]) -> None:
        """
        Register a new format converter.

        Parameters
        ----------
        format_name : str
            The name of the format (e.g., "json", "pickle").
        converter : Type[FormatConverter]
            The converter class for the format.

        Examples
        --------
        >>> FormatRegistry.register("json", JSONConverter)
        """
        cls._converters[format_name] = converter

    @classmethod
    def get_converter(cls, format_name: str) -> Type[FormatConverter]:
        """
        Retrieve a format converter by name.

        Parameters
        ----------
        format_name : str
            The name of the format to retrieve.

        Returns
        -------
        Type[FormatConverter]
            The converter class for the specified format.

        Raises
        ------
        UnsupportedFormatError
            If the specified format is not supported.

        Examples
        --------
        >>> json_converter = FormatRegistry.get_converter("json")
        >>> serialized_data = json_converter.serialize_to_format(data_dict)
        """
        if format_name not in cls._converters:
            raise UnsupportedFormatError(f"Unsupported format: {format_name}")
        return cls._converters[format_name]
