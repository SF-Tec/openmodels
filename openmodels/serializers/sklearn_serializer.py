"""
Scikit-learn model serializer for the OpenModels library.

This module provides a serializer for scikit-learn models, allowing them to be
converted to and from dictionary representations.
"""

from typing import Any, Dict, List, Type
import numpy as np
from scipy.sparse import _csr, csr_matrix  # type: ignore

import sklearn
from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

# from sklearn.ensemble import (
#    RandomForestRegressor,
#    RandomForestClassifier,
#    GradientBoostingClassifier,
#    GradientBoostingRegressor,
# )

from sklearn.svm import SVR, SVC
from sklearn.linear_model import (
    LogisticRegression,
    Lasso,
    Ridge,
    LinearRegression,
    # Perceptron,
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.dummy import DummyClassifier

# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import NotFittedError

from openmodels.exceptions import UnsupportedEstimatorError, SerializationError
from openmodels.protocols import ModelSerializer

# Dictionary of supported estimators
SUPPORTED_ESTIMATORS: Dict[str, Type[sklearn.base.BaseEstimator]] = {
    "BernoulliNB": BernoulliNB,
    "ComplementNB": ComplementNB,
    "GaussianNB": GaussianNB,
    # "DecisionTreeClassifier": DecisionTreeClassifier, # tree_ instance
    # "DecisionTreeRegressor": DecisionTreeRegressor, # tree_ instance
    "DummyClassifier": DummyClassifier,
    # "GradientBoostingClassifier": GradientBoostingClassifier,
    # contains stimators_ attribute with DecisionTreeRegressor
    # "GradientBoostingRegressor": GradientBoostingRegressor,
    # contains stimators_ attribute with DecisionTreeRegressor
    "Lasso": Lasso,
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
    "LinearRegression": LinearRegression,
    "LogisticRegression": LogisticRegression,
    "KMeans": KMeans,
    # "MLPClassifier": MLPClassifier,  # needs _label_binarizer attribute with LabelBinarizer type
    "MLPRegressor": MLPRegressor,
    "MultinomialNB": MultinomialNB,
    "PCA": PCA,
    # "Perceptron": Perceptron, # contains loss_function_ attribut with Hinge type
    "PLSRegression": PLSRegression,
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis,
    # "RandomForestClassifier": RandomForestClassifier,
    # contains stimators_ attribute with DecisionTreeRegressor
    # "RandomForestRegressor": RandomForestRegressor,
    # contains stimators_ attribute with DecisionTreeRegressor
    "Ridge": Ridge,
    "SVC": SVC,
    "SVR": SVR,
}

# Dictionary of attribute exceptions
ATTRIBUTE_EXCEPTIONS: Dict[str, list] = {
    "BernoulliNB": [],
    "ComplementNB": [],
    # "DecisionTreeClassifier": [], # not suppoted
    # "DecisionTreeRegressor": [], # not suppoted
    "DummyClassifier": ["_strategy"],
    "GaussianNB": [],
    # "GradientBoostingClassifier": [], # not supported
    "GradientBoostingRegressor": [],
    "Lasso": [],
    "LinearDiscriminantAnalysis": [],
    "LinearRegression": [],
    "LogisticRegression": [],
    "KMeans": ["_n_threads"],
    # "MLPClassifier": ["_label_binarizer"],  # not supported
    "MLPRegressor": [],  # not supported
    "MultinomialNB": [],
    "PCA": [],
    # "Perceptron": [], # not supported
    "PLSRegression": ["_x_mean", "_predict_1d"],
    "QuadraticDiscriminantAnalysis": [],
    # "RandomForestClassifier": [], # not supported
    # "RandomForestRegressor": [], # not supported
    "Ridge": [],
    "SVC": [
        "_sparse",
        "_n_support",
        "_dual_coef_",
        "_intercept_",
        "_probA",
        "_probB",
        "_gamma",
    ],
    "SVR": [
        "_sparse",
        "_n_support",
        "_dual_coef_",
        "_intercept_",
        "_probA",
        "_probB",
        "_gamma",
    ],
}

# List of supported types for serialization
SUPPORTED_TYPES: List[Type] = [
    bool,
    str,
    int,
    float,
    list,
    tuple,
    np.float64,
    np.ndarray,
]


class SklearnSerializer(ModelSerializer):
    """
    Serializer for scikit-learn estimators.

    This class provides methods to convert scikit-learn estimators to and from
    dictionary representations, which can then be used with various format converters.

    The serializer supports a wide range of scikit-learn estimators and handles
    the conversion of numpy arrays and other non-JSON-serializable types.

    Attributes
    ----------
    SUPPORTED_ESTIMATORS : Dict[str, Type[BaseEstimator]]
        A dictionary of supported scikit-learn estimator classes.
    SUPPORTED_TYPES : List[Type]
        A list of supported types for serialization.
    """

    @staticmethod
    def _convert_to_serializable_types(value: Any) -> Any:
        """
        Convert a value to a serializable type.

        Parameters
        ----------
        value : Any
            The value to convert.

        Returns
        -------
        Any
            The serializable representation of the value.
        """
        if isinstance(value, (np.ndarray, list)):
            return SklearnSerializer._array_to_list(value)
        if isinstance(value, _csr.csr_matrix):
            # Convert indices and indptr to int32 explicitly
            csr_value = csr_matrix(value)
            serialized_sparse_matrix = {
                "data": SklearnSerializer._array_to_list(csr_value.data),
                "indptr": SklearnSerializer._array_to_list(
                    csr_value.indptr.astype(np.int32)
                ),
                "indices": SklearnSerializer._array_to_list(
                    csr_value.indices.astype(np.int32)
                ),
                "shape": SklearnSerializer._array_to_list(csr_value.shape),
            }
            return serialized_sparse_matrix

        return value

    @staticmethod
    def _convert_to_sklearn_types(value: Any, attr_type: str = "none") -> Any:
        """
        Convert a JSON-deserialized value to its scikit-learn type.

        Parameters
        ----------
        value : Any
            The JSON-deserialized value.
        attr_type : str
            The target type to convert to.

        Returns
        -------
        Any
            The scikit-learn type of the value.
        """
        # Base case: if attr_type is not a list, convert value based on attr_type
        if isinstance(attr_type, str):
            if attr_type == "csr_matrix":
                # Ensure all sparse matrix components are of correct dtype
                return csr_matrix(
                    (
                        np.array(value["data"], dtype=np.float64),
                        np.array(value["indices"], dtype=np.int32),
                        np.array(value["indptr"], dtype=np.int32),
                    ),
                    shape=tuple(value["shape"]),
                )
            elif attr_type == "ndarray":
                return np.array(value)
            elif attr_type == "int":
                return int(value)
            elif attr_type == "float":
                return float(value)
            elif attr_type == "str":
                return str(value)
            # Add other types as needed
            return value  # Return as-is if no specific conversion is needed
        # Recursive case: if attr_type is a list, process each element in value
        elif isinstance(attr_type, list) and isinstance(value, list):
            return [
                SklearnSerializer._convert_to_sklearn_types(v, t)
                for v, t in zip(value, attr_type)
            ]

        return value

    @staticmethod
    def _array_to_list(array: Any) -> Any:
        """
        Recursively convert numpy arrays to nested lists.

        Parameters
        ----------
        array : array-like
            The array or nested structure to convert.

        Returns
        -------
        list or Any
            The input converted to a nested list structure, or the original value if not an array.
        """
        if isinstance(array, np.ndarray):
            return SklearnSerializer._array_to_list(array.tolist())
        elif isinstance(array, list):
            return [SklearnSerializer._array_to_list(item) for item in array]
        elif isinstance(array, tuple):
            return tuple(SklearnSerializer._array_to_list(item) for item in array)
        else:
            return array

    @staticmethod
    def get_nested_types(item: Any) -> Any:
        """
        Recursively determine the type of elements within nested lists.

        Parameters
        ----------
        item : Any
            The item to inspect for nested types.

        Returns
        -------
        Any
            A nested list representing the types of elements in the input item.

        Examples
        ---------

        [1, [1, 2, [1, 2, 3]], 2] -> ['int',['int','int','ndarray'],'int']

        """
        if isinstance(item, list) and item:  # If it's a list and not empty
            return [SklearnSerializer.get_nested_types(subitem) for subitem in item]
        else:
            # Return the type name if it's not a list or it's an empty list
            return type(item).__name__

    def serialize(self, model: BaseEstimator) -> Dict[str, Any]:
        """
        Serialize a scikit-learn estimator to a dictionary.

        This method extracts relevant attributes from the model, converts them to
        JSON-serializable types, and returns a dictionary representation of the model.

        Parameters
        ----------
        model : BaseEstimator
            The scikit-learn estimator to serialize.

        Returns
        -------
        Dict[str, Any]
            A dictionary representation of the model.

        Raises
        ------
        SerializationError
            If the model has not been fitted or if there's an error during serialization.

        Examples
        --------
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=100, n_features=20, n_classes=2)
        >>> model = LogisticRegression().fit(X, y)
        >>> serializer = SklearnSerializer()
        >>> serialized_dict = serializer.serialize(model)
        """
        try:
            check_is_fitted(model)
        except NotFittedError as e:
            raise SerializationError("Cannot serialize an unfitted model") from e

        # Get all attributes that are not private, not properties, and not callable
        # Attributes that have been estimated from the data must always have a name ending with
        # trailing underscore,
        # for example the coefficients of some regression estimator would be stored in a
        # coef_ attribute after fit has been called.
        # https://scikit-learn.org/stable/glossary.html#term-attributes
        # https://scikit-learn.org/stable/developers/develop.html#estimated-attributes
        # NOTE: This is not always true for all estimators, but it is a good starting point.
        filtered_attribute_keys = [
            key
            for key in dir(model)
            if not key.startswith("__")  # not private
            and key.endswith("_")
            and not key.endswith("__")
            and not isinstance(getattr(type(model), key, None), property)
            and not callable(getattr(model, key))
        ]

        # There are some attributes that are removed in the previous filter according to the
        # sklearn documentation.
        # However, they are still needed in the serialized model so we add them to the list.
        filtered_attribute_keys = (
            filtered_attribute_keys + ATTRIBUTE_EXCEPTIONS[model.__class__.__name__]
        )

        attribute_values = [getattr(model, key) for key in filtered_attribute_keys]

        # Generate attribute types with nested structure.
        # These types are used to convert the serialized attributes back to their original types.
        attribute_types = [
            SklearnSerializer.get_nested_types(value) for value in attribute_values
        ]

        serializable_attribute_values = [
            self._convert_to_serializable_types(value) for value in attribute_values
        ]

        # We losely follow the ONNX standard for the serialized model.
        # https://github.com/onnx/onnx/blob/main/docs/IR.md
        return {
            "attributes": dict(
                zip(filtered_attribute_keys, serializable_attribute_values)
            ),
            "attribute_types": dict(zip(filtered_attribute_keys, attribute_types)),
            "estimator_class": model.__class__.__name__,
            "params": model.get_params(),
            "producer_name": model.__module__.split(".")[0],
            "producer_version": model.__getstate__()["_sklearn_version"],
            "model_version": model.__getstate__()["_sklearn_version"],
            "domain": "sklearn",
        }

    def deserialize(self, data: Dict[str, Any]) -> BaseEstimator:
        """
        Deserialize a dictionary representation back into a scikit-learn estimator.

        This method reconstructs a scikit-learn estimator from its dictionary
        representation, converting attributes back to their original types.

        Parameters
        ----------
        data : Dict[str, Any]
            The dictionary representation of the model.

        Returns
        -------
        BaseEstimator
            The deserialized scikit-learn estimator.

        Raises
        ------
        UnsupportedEstimatorError
            If the estimator class is not supported.

        Examples
        --------
        >>> serializer = SklearnSerializer()
        >>> deserialized_model = serializer.deserialize(serialized_dict)
        >>> predictions = deserialized_model.predict(X_test)
        """
        estimator_class = data["estimator_class"]
        if estimator_class not in SUPPORTED_ESTIMATORS:
            raise UnsupportedEstimatorError(
                f"Unsupported estimator class: {estimator_class}"
            )

        model = SUPPORTED_ESTIMATORS[estimator_class](**data["params"])

        for attribute, value in data["attributes"].items():
            # Retrieve the attribute type from data["attribute_types"]
            attr_type = data["attribute_types"].get(attribute)
            # Pass both value and attr_type to _convert_to_sklearn_types
            setattr(model, attribute, self._convert_to_sklearn_types(value, attr_type))

        return model
