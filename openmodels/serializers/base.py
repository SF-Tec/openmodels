# base.py
import numpy as np
from scipy.sparse import csr_matrix
from scipy.interpolate import interp1d
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from typing import Any


class SerializerMixin:
    """
    Base mixin providing recursive serialization and, native python objects 
    serialization a dispatch mechanism. Other mixins only need to implement
    `_get_handlers()` and serialization helpers.
    """

    def convert_to_serializable(self, value):
        """Recursively convert values into JSON-serializable types."""
        # First check custom handlers
        for typ, handler in self._get_handlers():
            if isinstance(value, typ):
                return handler(value)

        # Recursive case: dict, list, tuple
        if isinstance(value, dict):
            return {str(k): self.convert_to_serializable(v) for k, v in value.items()}

        if isinstance(value, (list, tuple)):
            return [self.convert_to_serializable(v) for v in value]

        return value
    
    def _serialize_slice(self, value: slice):
        return {"start": value.start, "stop": value.stop, "step": value.step}

    def _serialize_type(self, value: type):
        return {"type_name": value.__name__}

    def _get_handlers(self):
        """Each mixin extends this list."""
        return [
            (slice, self._serialize_slice),
            (type, self._serialize_type),
        ]


class NumpySerializerMixin(SerializerMixin):
    def _get_dtype(self, value: Any) -> str:
        """
        Get the dtype of a numpy array, otherwise return empty string.
        """
        if isinstance(value, np.ndarray):
            return str(value.dtype)  # Get the actual numpy dtype
        elif isinstance(value, (list, tuple)) and value:
            # If it's a list/tuple that will become an ndarray, check its elements
            first_elem = value[0]
            if isinstance(first_elem, (int, np.integer)):
                return "int32"  # Use int32 for integer lists
            elif isinstance(first_elem, (float, np.floating)):
                return "float64"  # Use float64 for float lists
        return ""
      
    def _serialize_ndarray(self, value: np.ndarray):
        return self.convert_to_serializable(value.tolist())

    def _serialize_generic(self, value: np.generic):
        return value.item()

    def _get_handlers(self):
        return super()._get_handlers() + [
            (np.ndarray, self._serialize_ndarray),
            (np.generic, self._serialize_generic),
            (np.dtype, str),
            (type(np.dtype("float64")), str),
            (
                np.random.RandomState,
                lambda v: [self.convert_to_serializable(x) for x in v.get_state()],
            ),
        ]


class ScipySerializerMixin(SerializerMixin):
    def _serialize_csr_matrix(self, value: csr_matrix):
        csr_value = csr_matrix(value)
        return {
            "data": self.convert_to_serializable(csr_value.data),
            "indptr": self.convert_to_serializable(csr_value.indptr.astype(np.int32)),
            "indices": self.convert_to_serializable(csr_value.indices.astype(np.int32)),
            "shape": self.convert_to_serializable(csr_value.shape),
        }

    def _serialize_interp1d(self, value: interp1d):
        fill_value = getattr(value, "fill_value", np.nan)
        if isinstance(fill_value, np.ndarray):
            fill_value = self.convert_to_serializable(fill_value)
        return {
            "x": self.convert_to_serializable(value.x),
            "y": self.convert_to_serializable(value.y),
            "kind": getattr(value, "_kind", "linear"),
            "fill_value": fill_value,
            "bounds_error": getattr(value, "bounds_error", None),
            "assume_sorted": getattr(value, "assume_sorted", False),
            "axis": getattr(value, "axis", -1),
            "copy": getattr(value, "copy", True),
        }

    def _serialize_scipy_dist(self, value: rv_continuous_frozen):
        return {"dist_name": value.dist.name, "args": value.args, "kwargs": value.kwds}

    def _get_handlers(self):
        return super()._get_handlers() + [
            (csr_matrix, self._serialize_csr_matrix),
            (interp1d, self._serialize_interp1d),
            (rv_continuous_frozen, self._serialize_scipy_dist),
        ]
