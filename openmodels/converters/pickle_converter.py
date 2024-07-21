"""
Pickle converter for the OpenModels library.

This module provides a converter for serializing to and from pickle format.
"""

import pickle
from typing import Any, Dict

from openmodels.protocols import FormatConverter


class PickleConverter(FormatConverter):
    """
    Converter for pickle format.

    This class provides static methods to convert between dictionary
    representations and pickle byte strings.
    """

    @staticmethod
    def serialize_to_format(data: Dict[str, Any]) -> bytes:
        """
        Convert a dictionary to a pickle byte string.

        Parameters
        ----------
        data : Dict[str, Any]
            The dictionary to convert.

        Returns
        -------
        bytes
            The pickle byte string representation of the data.
        """
        return pickle.dumps(data)

    @staticmethod
    def deserialize_from_format(formatted_data: bytes) -> Dict[str, Any]:
        """
        Convert a pickle byte string to a dictionary.

        Parameters
        ----------
        formatted_data : bytes
            The pickle byte string to convert.

        Returns
        -------
        Dict[str, Any]
            The dictionary representation of the pickle data.

        Raises
        ------
        pickle.UnpicklingError
            If the input bytes cannot be unpickled.
        """
        return pickle.loads(formatted_data)
