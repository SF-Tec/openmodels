"""
Scikit-learn model serializer for the OpenModels library.

This module provides a serializer for scikit-learn models, allowing them to be
converted to and from dictionary representations.
"""

from typing import Any, Dict, List, Type
import numpy as np

import sklearn
from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.svm import SVR, SVC
from sklearn.linear_model import (
    LogisticRegression,
    Lasso,
    Ridge,
    LinearRegression,
    Perceptron,
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.exceptions import NotFittedError

from openmodels.exceptions import UnsupportedEstimatorError, SerializationError

# Dictionary of supported estimators
SUPPORTED_ESTIMATORS: Dict[str, Type[sklearn.base.BaseEstimator]] = {
    "BernoulliNB": BernoulliNB,
    "ComplementNB": ComplementNB,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "DummyClassifier": DummyClassifier,
    "GaussianNB": GaussianNB,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "Lasso": Lasso,
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
    "LinearRegression": LinearRegression,
    "LogisticRegression": LogisticRegression,
    "KMeans": KMeans,
    "MLPClassifier": MLPClassifier,
    "MLPRegressor": MLPRegressor,
    "MultinomialNB": MultinomialNB,
    "PCA": PCA,
    "Perceptron": Perceptron,
    "PLSRegression": PLSRegression,
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis,
    "RandomForestClassifier": RandomForestClassifier,
    "RandomForestRegressor": RandomForestRegressor,
    "Ridge": Ridge,
    "SVC": SVC,
    "SVR": SVR,
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


class SklearnSerializer:
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
        return value

    @staticmethod
    def _convert_to_sklearn_types(value: Any) -> Any:
        """
        Convert a JSON-deserialized value to its scikit-learn type.

        Parameters
        ----------
        value : Any
            The JSON-deserialized value.

        Returns
        -------
        Any
            The scikit-learn type of the value.
        """
        if isinstance(value, list):
            return np.array(value)
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

        filtered_attribute_keys = [
            key
            for key in dir(model)
            if not callable(getattr(model, key))
            and not key.endswith("__")
            and type(getattr(model, key)) in SUPPORTED_TYPES
            and (
                not isinstance(getattr(type(model), key, None), property)
                or getattr(type(model), key).fset is not None
            )
        ]

        attribute_values = [getattr(model, key) for key in filtered_attribute_keys]
        attribute_types = [type(value) for value in attribute_values]
        serializable_attribute_values = [
            self._convert_to_serializable_types(value) for value in attribute_values
        ]

        return {
            "attributes": dict(
                zip(filtered_attribute_keys, serializable_attribute_values)
            ),
            "attribute_types": [str(attr_type) for attr_type in attribute_types],
            "estimator_class": model.__class__.__name__,
            "params": model.get_params(),
            "producer_name": "sklearn",
            "producer_version": sklearn.__version__,
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
            setattr(model, attribute, self._convert_to_sklearn_types(value))

        return model
