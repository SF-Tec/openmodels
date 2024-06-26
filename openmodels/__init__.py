import json
from typing import Any, Dict, List, Type
import numpy as np

import sklearn
from sklearn.base import check_is_fitted
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


def array_to_list(array):
    if isinstance(array, np.ndarray):
        return array_to_list(array.tolist())
    elif isinstance(array, list):
        return [array_to_list(item) for item in array]
    elif isinstance(array, tuple):
        return tuple(array_to_list(item) for item in array)
    else:
        return array


def _convert_to_json_types(value: Any) -> Any:
    """
    Convert a value to a JSON-serializable type.

    This process may entail removing elements that are not JSON-serializable,
    changing the type of certain elements to make them JSON-serializable,
    or performing other transformations as necessary.

    Parameters
    ----------
    value : Any
        The value to convert.

    Returns
    -------
    Any
        The JSON-serializable representation of the value.

    Examples
    --------
    >>> convert_to_json_serializable(np.array([1, 2, 3]))
    [1, 2, 3]
    >>> convert_to_json_serializable({'a': 1, 'b': np.array([1, 2, 3])})
    {'a': 1, 'b': [1, 2, 3]}
    """
    if isinstance(value, np.ndarray) or isinstance(value, list):
        return array_to_list(value)
    return value


def _convert_to_sklearn_types(value: Any) -> Any:
    """
    Convert a JSON-deserialized value to its sklearn type.

    This process may entail adding elements that were removed during the
    serialization process, changing the type of certain elements to their
    sklearn type, or performing other transformations as necessary.

    Parameters
    ----------
    value : Any
        The JSON-deserialized value.

    Returns
    -------
    Any
        The original type of the value.

    Examples
    --------
    >>> convert_from_json_serializable([1, 2, 3])
    array([1, 2, 3])
    >>> convert_from_json_serializable({'a': 1, 'b': [1, 2, 3]})
    {'a': 1, 'b': array([1, 2, 3])}
    """
    if isinstance(value, list):
        return np.array(value)
    return value


def model_to_json_dict(model: sklearn.base.BaseEstimator) -> Dict[str, Any]:
    """
    Serialize a scikit-learn model to a JSON-serializable dictionary.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        The scikit-learn model to serialize.

    Returns
    -------
    Dict[str, Any]
        The JSON-serializable dictionary representation of the model.

    Raises
    ------
    NotFittedError
        If the model has not been fitted yet.
    """
    check_is_fitted(model)

    # IMPORTANT: We need to filter only the attributes that are needed to rebuild the
    # model (estimated and fitted attributes)
    # https://scikit-learn.org/stable/developers/develop.html#developer-api-for-set-output
    # https://scikit-learn.org/stable/developers/develop.html#estimated-attributes

    # 1. Filter out methods and internal
    filtered_attribute_keys = [
        key
        for key in dir(model)
        if not callable(getattr(model, key)) and not key.endswith("__")
    ]

    # 2. Filter the attributes that our library currently allows to serialize (some
    # may need a type conversion)
    filtered_attribute_keys = [
        key
        for key in filtered_attribute_keys
        if type(getattr(model, key)) in SUPPORTED_TYPES
    ]

    # 3. Filter the attributes that allow setting a value
    filtered_attribute_keys = [
        key
        for key in filtered_attribute_keys
        if not isinstance(getattr(type(model), key, None), property)
        or getattr(type(model), key).fset is not None
    ]

    # 4. Convert non-serializable types to serializable ones
    attribute_values = [getattr(model, key) for key in filtered_attribute_keys]
    attribute_types = [type(value) for value in attribute_values]
    serializable_attribute_values = [
        _convert_to_json_types(value) for value in attribute_values
    ]

    serialized_model = {
        "attributes": dict(zip(filtered_attribute_keys, serializable_attribute_values)),
        "attribute_types": [str(attr_type) for attr_type in attribute_types],
        "estimator_class": model.__class__.__name__,
        "params": model.get_params(),
        "producer_name": "sklearn",
        "producer_version": sklearn.__version__,
    }

    return serialized_model


def model_from_json_dict(model_dict: Dict[str, Any]) -> sklearn.base.BaseEstimator:
    """
    Deserialize a scikit-learn model from a JSON-serializable dictionary.

    Parameters
    ----------
    model_dict : Dict[str, Any]
        The JSON-serializable dictionary representation of the model.

    Returns
    -------
    sklearn.base.BaseEstimator
        The deserialized scikit-learn model.
    """
    deserialized_model = SUPPORTED_ESTIMATORS[model_dict["estimator_class"]](
        **model_dict["params"]
    )

    for attribute, value in model_dict["attributes"].items():
        setattr(deserialized_model, attribute, _convert_to_sklearn_types(value))

    return deserialized_model


def model_to_json_file(model: sklearn.base.BaseEstimator, file_path: str) -> None:
    """
    Serialize a scikit-learn model to a JSON file.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        The scikit-learn model to serialize.
    file_path : str
        The path to the JSON file to write the model to.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(model_to_json_dict(model), file)


def model_from_json_file(file_path: str) -> sklearn.base.BaseEstimator:
    """
    Deserialize a scikit-learn model from a JSON file.

    Parameters
    ----------
    file_path : str
        The path to the JSON file to read the model from.

    Returns
    -------
    sklearn.base.BaseEstimator
        The deserialized scikit-learn model.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        model_dict = json.load(file)
    return model_from_json_dict(model_dict)
