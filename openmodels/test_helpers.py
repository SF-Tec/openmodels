"""
This file contains functions for testing the serialization and deserialization of scikit-learn models
using the `openmodels` library.
"""

import os
from typing import Optional, Union, Protocol, runtime_checkable, TypeVar, cast
import numpy as np
from numpy import testing
from sklearn.base import BaseEstimator

from openmodels import SerializationManager, SklearnSerializer


@runtime_checkable
class PredictorModel(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


@runtime_checkable
class TransformerModel(Protocol):
    def transform(self, X: np.ndarray) -> np.ndarray:
        ...


@runtime_checkable
class FittableModel(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "FittableModel":
        ...


ModelType = Union[PredictorModel, TransformerModel, FittableModel]
T = TypeVar("T", bound=Union[BaseEstimator, ModelType])


def fit_model(model: T, x: np.ndarray, y: np.ndarray, abs: bool = False) -> T:
    """
    Fits a model to the provided data.

    Parameters
    ----------
    model : T
        The scikit-learn model to fit.
    x : np.ndarray
        The training input samples.
    y : np.ndarray
        The target values (class labels in classification, real numbers in regression).
    abs : bool, default=False
        Whether to take the absolute value of the input data before fitting the model.

    Returns
    -------
    T
        The fitted scikit-learn model.
    """
    if not isinstance(model, FittableModel):
        raise TypeError("Model must have a 'fit' method")

    if abs:
        model.fit(np.absolute(x), y)
    else:
        model.fit(x, y)
    return model


def test_predictions(
    model1: PredictorModel, model2: PredictorModel, x: np.ndarray
) -> None:
    """
    Compares the predictions of two models on the given data.

    Parameters
    ----------
    model1 : PredictorModel
        The first scikit-learn model with a predict method.
    model2 : PredictorModel
        The second scikit-learn model with a predict method.
    x : np.ndarray
        The input samples.

    Raises
    ------
    AssertionError
        If the predictions of the two models are not equal.
    """
    expected_predictions = model1.predict(x)
    actual_predictions = model2.predict(x)
    testing.assert_array_equal(expected_predictions, actual_predictions)


def test_transformed_data(
    model1: TransformerModel, model2: TransformerModel, x: np.ndarray
) -> None:
    """
    Compares the transformed data of two models on the given data.

    Parameters
    ----------
    model1 : TransformerModel
        The first scikit-learn model with a transform method.
    model2 : TransformerModel
        The second scikit-learn model with a transform method.
    x : np.ndarray
        The input samples.

    Raises
    ------
    AssertionError
        If the transformed data of the two models are not almost equal.
    """
    expected_transformed_data = model1.transform(x)
    actual_transformed_data = model2.transform(x)
    testing.assert_array_almost_equal(
        expected_transformed_data, actual_transformed_data
    )


def run_test_model(
    model: Union[BaseEstimator, ModelType],
    x: np.ndarray,
    y: np.ndarray,
    x_sparse: Optional[np.ndarray],
    y_sparse: Optional[np.ndarray],
    model_name: str,
    abs: bool = False,
) -> None:
    """
    Tests the serialization and deserialization of a scikit-learn model.

    Parameters
    ----------
    model : Union[BaseEstimator, ModelType]
        The scikit-learn model to test.
    x : np.ndarray
        The training input samples.
    y : np.ndarray
        The target values (class labels in classification, real numbers in regression).
    x_sparse : np.ndarray or None
        The sparse training input samples.
    y_sparse : np.ndarray or None
        The sparse target values.
    model_name : str
        The name of the file to save the serialized model to.
    abs : bool, default=False
        Whether to take the absolute value of the input data before fitting the model.
    """
    # Fit and test the model
    fitted_model = fit_model(model, x, y, abs)
    if x_sparse is not None and y_sparse is not None:
        fit_model(model, x_sparse, y_sparse, abs)

    # Create a SerializationManager instance
    manager = SerializationManager(SklearnSerializer())

    # Serialize and deserialize the model
    serialized_model = manager.serialize(fitted_model, format_name="json")
    deserialized_model = manager.deserialize(serialized_model, format_name="json")

    # Test the deserialized model
    if isinstance(model, PredictorModel):
        test_predictions(
            cast(PredictorModel, fitted_model),
            cast(PredictorModel, deserialized_model),
            x,
        )
    elif isinstance(model, TransformerModel):
        test_transformed_data(
            cast(TransformerModel, fitted_model),
            cast(TransformerModel, deserialized_model),
            x,
        )

    # Serialize and deserialize the model to/from a file
    model_file_path = f"./test/temp/{model_name}.json"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)

    with open(model_file_path, "w", encoding="utf-8") as f:
        f.write(serialized_model)

    with open(model_file_path, "r", encoding="utf-8") as f:
        serialized_model_from_file = f.read()

    deserialized_model_from_file = manager.deserialize(
        serialized_model_from_file, format_name="json"
    )

    # Test the deserialized model from file
    if isinstance(model, PredictorModel):
        test_predictions(
            cast(PredictorModel, fitted_model),
            cast(PredictorModel, deserialized_model_from_file),
            x,
        )
    elif isinstance(model, TransformerModel):
        test_transformed_data(
            cast(TransformerModel, fitted_model),
            cast(TransformerModel, deserialized_model_from_file),
            x,
        )

    # Clean up the temporary file
    os.remove(model_file_path)


def create_test_data(
    n_samples: int = 100, n_features: int = 5, random_state: int = 42
) -> tuple:
    """
    Creates test data for model fitting and testing.

    Parameters
    ----------
    n_samples : int, optional
        The number of samples to generate (default is 100).
    n_features : int, optional
        The number of features to generate (default is 5).
    random_state : int, optional
        The random state for reproducibility (default is 42).

    Returns
    -------
    tuple
        A tuple containing (X, y) where X is the feature matrix and y is the target vector.
    """
    np.random.seed(random_state)
    x = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, size=n_samples)
    return x, y
