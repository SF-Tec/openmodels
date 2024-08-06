# OpenModels

[![PyPI version](https://badge.fury.io/py/openmodels.svg)](https://badge.fury.io/py/openmodels)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/openmodels.svg)](https://pypi.org/project/openmodels/)

OpenModels is a flexible and extensible library for serializing and deserializing machine learning models. It's designed to support any serialization format through a plugin-based architecture, providing a safe and transparent solution for exporting and sharing predictive models.

## Key Features

- **Format Agnostic**: Supports any serialization format through a plugin-based system.
- **Extensible**: Easily add support for new model types and serialization formats.
- **Safe**: Provides alternatives to potentially unsafe serialization methods like Pickle.
- **Transparent**: Supports human-readable formats for easy inspection of serialized models.

## Installation

```bash
pip install openmodels
```

## Quick Start

```python
from openmodels import SerializationManager, SklearnSerializer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Create and train a scikit-learn model
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0)
model.fit(X, y)

# Create a SerializationManager
manager = SerializationManager(SklearnSerializer())

# Serialize the model (default format is JSON)
serialized_model = manager.serialize(model)

# Deserialize the model
deserialized_model = manager.deserialize(serialized_model)

# Use the deserialized model
predictions = deserialized_model.predict(X[:5])
print(predictions)
```

## Extensibility

OpenModels is designed to be easily extended with new serialization formats and model types.

### Adding a New Format

To add a new serialization format, create a class that implements the `FormatConverter` protocol and register it with the `FormatRegistry`:

```python
from openmodels.protocols import FormatConverter
from openmodels.format_registry import FormatRegistry
from typing import Dict, Any

class YAMLConverter(FormatConverter):
    @staticmethod
    def serialize_to_format(data: Dict[str, Any]) -> str:
        import yaml
        return yaml.dump(data)

    @staticmethod
    def deserialize_from_format(formatted_data: str) -> Dict[str, Any]:
        import yaml
        return yaml.safe_load(formatted_data)

FormatRegistry.register("yaml", YAMLConverter)
```

### Adding a New Model Serializer

To add support for a new type of model, create a class that implements the `ModelSerializer` protocol:

```python
from openmodels.protocols import ModelSerializer
from typing import Any, Dict

class TensorFlowSerializer(ModelSerializer):
    def serialize(self, model: Any) -> Dict[str, Any]:
        # Implementation for serializing TensorFlow models
        ...

    def deserialize(self, data: Dict[str, Any]) -> Any:
        # Implementation for deserializing TensorFlow models
        ...
```

## Supported Models

OpenModels currently supports a wide range of scikit-learn models, including:

- Classification: LogisticRegression, RandomForestClassifier, SVC, etc.
- Regression: LinearRegression, RandomForestRegressor, SVR, etc.
- Clustering: KMeans
- Dimensionality Reduction: PCA

For a full list of supported models, please refer to the `SUPPORTED_ESTIMATORS` dictionary in `serializers/sklearn_serializer.py`.

## Contributing

We welcome contributions to OpenModels! Whether you want to add support for new models, implement new serialization formats, or improve the existing codebase, your help is appreciated.

Please refer to our [Contributing Guidelines](CONTRIBUTING.md) for more information on how to get started.

## Running Tests

To run the tests:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/openmodels.git
   cd openmodels
   ```

2. Install the package and its dependencies:

   ```bash
   pip install -e .
   ```

3. Run the tests:
   ```bash
   pytest
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Changelog

For a detailed changelog, please see the [CHANGELOG.md](CHANGELOG.md) file.

## Support

If you encounter any issues or have questions, please [file an issue](https://github.com/SF-Tec/openmodels) on our GitHub repository.

We're always looking to improve OpenModels. If you have any suggestions or feature requests, please let us know!
