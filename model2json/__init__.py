import json
from typing import Any, Dict, List, Type
import numpy as np

import sklearn
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, LinearRegression, Perceptron
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Dictionary of supported estimators
SUPPORTED_ESTIMATORS: Dict[str, Type[sklearn.base.BaseEstimator]] = {
    'BernoulliNB': BernoulliNB,
    'ComplementNB': ComplementNB,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'DecisionTreeRegressor': DecisionTreeRegressor,
    'DummyClassifier': DummyClassifier,
    'GaussianNB': GaussianNB,
    'Lasso': Lasso,
    'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis,
    'LinearRegression': LinearRegression,
    'LogisticRegression': LogisticRegression,
    'MLPClassifier': MLPClassifier,
    'MLPRegressor': MLPRegressor,
    'MultinomialNB': MultinomialNB,
    'PCA': PCA,
    'Perceptron': Perceptron,
    'PLSRegression': PLSRegression,
    'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis,
    'RandomForestClassifier': RandomForestClassifier,
    'RandomForestRegressor': RandomForestRegressor,
    'Ridge': Ridge,
    'SVC': SVC,
    'SVR': SVR,
}

# List of supported types for serialization
SUPPORTED_TYPES: List[Type] = [bool, str, int, float, list, tuple, np.float64, np.ndarray]

def untype(value: Any) -> Any:
    """Convert a value to a JSON-serializable type."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value

def retype(value: Any) -> Any:
    """Convert a JSON-deserialized value to its original type."""
    if isinstance(value, list):
        return np.array(value)
    return value

def serialize_model(model: sklearn.base.BaseEstimator) -> Dict[str, Any]:
    """Serialize a scikit-learn model to a JSON-serializable dictionary."""
    attribute_keys = [key for key in dir(model)
                      if not callable(getattr(model, key))
                      and not key.endswith('__')]
    filtered_attribute_keys = [key for key in attribute_keys
                               if type(getattr(model, key)) in SUPPORTED_TYPES]
    filtered_attribute_keys = [key for key in filtered_attribute_keys
                                if not isinstance(getattr(type(model), key, None), property)
                                or getattr(type(model), key).fset is not None]
    attribute_values = map(lambda filtered_attribute_key: getattr(model, filtered_attribute_key), filtered_attribute_keys)
    attribute_types = map(lambda attribute_value: type(attribute_value), attribute_values)
    attribute_untyped_values = map(untype, attribute_values)
    serialized_model = {
        'attributes': dict(zip(filtered_attribute_keys, list(attribute_untyped_values))),
        'attribute_types': list(attribute_types),
        'estimator_class': model.__class__.__name__,
        'params': model.get_params(),
        'producer_name': 'sklearn',
        'producer_version': sklearn.__version__
    }
    return serialized_model

def deserialize_model(model_dict: Dict[str, Any]) -> sklearn.base.BaseEstimator:
    """Deserialize a scikit-learn model from a JSON-serializable dictionary."""
    deserialized_model = SUPPORTED_ESTIMATORS[model_dict['estimator_class']](**model_dict['params'])
    for attribute, value in model_dict['attributes'].items():
        setattr(deserialized_model, attribute, retype(value))
    return deserialized_model

def to_dict(model: sklearn.base.BaseEstimator) -> Dict[str, Any]:
    """Serialize a scikit-learn model to a JSON-serializable dictionary."""
    return serialize_model(model)

def from_dict(model_dict: Dict[str, Any]) -> sklearn.base.BaseEstimator:
    """Deserialize a scikit-learn model from a JSON-serializable dictionary."""
    return deserialize_model(model_dict)

def to_json(model: sklearn.base.BaseEstimator, model_name: str) -> None:
    """Serialize a scikit-learn model to a JSON file."""
    with open(model_name, 'w') as model_json:
        json.dump(serialize_model(model), model_json)

def from_json(model_name: str) -> sklearn.base.BaseEstimator:
    """Deserialize a scikit-learn model from a JSON file."""
    with open(model_name, 'r') as model_json:
        model_dict = json.load(model_json)
        return deserialize_model(model_dict)
