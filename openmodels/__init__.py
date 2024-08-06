"""
OpenModels: A flexible library for serializing and deserializing machine learning models.

This package provides tools for converting machine learning models to various
formats and back. It supports different model types and serialization formats
through a plugin-based system.

Main components:
- SerializationManager: Coordinates the serialization and deserialization process.
- SklearnSerializer: Handles serialization for scikit-learn models.
- JSONConverter, PickleConverter: Convert between dict and specific formats.
- FormatRegistry: Manages available format converters.

Example usage:
    from openmodels import SerializationManager, SklearnSerializer
    from sklearn.linear_model import LogisticRegression

    # Create and train a scikit-learn model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Create a SerializationManager
    manager = SerializationManager(SklearnSerializer())

    # Serialize the model to JSON
    serialized_model = manager.serialize(model, format="json")

    # Deserialize the model from JSON
    deserialized_model = manager.deserialize(serialized_model, format="json")

    # Use the deserialized model
    predictions = deserialized_model.predict(X_test)

For more advanced usage and custom serializers or converters, refer to the documentation.
"""

from .core import SerializationManager
from .serializers import SklearnSerializer
from .converters import JSONConverter, PickleConverter
from .format_registry import FormatRegistry

# Register the JSONConverter
FormatRegistry.register("json", JSONConverter)
FormatRegistry.register("pickle", PickleConverter)

__all__ = [
    "FormatRegistry",
    "SerializationManager",
    "SklearnSerializer",
]
