import numpy as np
from numpy import testing
import model2json


def fit_model(model, x, y, abs=False):
    """Fits a model to the provided data.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        The scikit-learn model to fit.
    x : array-like of shape (n_samples, n_features)
        The training input samples.
    y : array-like of shape (n_samples,)
        The target values (class labels in classification, real numbers in regression).
    abs : bool, default=False
        Whether to take the absolute value of the input data before fitting the model.

    Returns
    -------
    sklearn.base.BaseEstimator
        The fitted scikit-learn model.
    """
    if abs:
        model.fit(np.absolute(x), y)
    else:
        model.fit(x, y)
    return model


def test_predictions(model1, model2, x):
    """Compares the predictions of two models on the given data.

    Parameters
    ----------
    model1 : sklearn.base.BaseEstimator
        The first scikit-learn model.
    model2 : sklearn.base.BaseEstimator
        The second scikit-learn model.
    x : array-like of shape (n_samples, n_features)
        The input samples.

    Raises
    ------
    AssertionError
        If the predictions of the two models are not equal.
    """
    expected_predictions = model1.predict(x)
    actual_predictions = model2.predict(x)
    testing.assert_array_equal(expected_predictions, actual_predictions)


def test_transformed_data(model1, model2, x):
    """Compares the transformed data of two models on the given data.

    Parameters
    ----------
    model1 : sklearn.base.BaseEstimator
        The first scikit-learn model.
    model2 : sklearn.base.BaseEstimator
        The second scikit-learn model.
    x : array-like of shape (n_samples, n_features)
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


def test_model(model, x, y, x_sparse, y_sparse, model_name, abs=False):
    """Tests the serialization and deserialization of a scikit-learn model.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        The scikit-learn model to test.
    x : array-like of shape (n_samples, n_features)
        The training input samples.
    y : array-like of shape (n_samples,)
        The target values (class labels in classification, real numbers in regression).
    x_sparse : array-like of shape (n_samples, n_features), or None
        The sparse training input samples.
    y_sparse : array-like of shape (n_samples,), or None
        The sparse target values.
    model_name : str
        The name of the file to save the serialized model to.
    abs : bool, default=False
        Whether to take the absolute value of the input data before fitting the model.
    """
    # Fit and test the model
    fit_model(model, x, y, abs)
    if x_sparse is not None and y_sparse is not None:
        fit_model(model, x_sparse, y_sparse, abs)

    # Serialize and deserialize the model
    serialized_model = model2json.to_dict(model)
    deserialized_model = model2json.from_dict(serialized_model)

    # Serialize and deserialize the model to JSON
    model2json.to_json(model, model_name)
    deserialized_model_json = model2json.from_json(model_name)

    # Check predictions or transformed data
    if hasattr(model, "predict"):
        test_predictions(model, deserialized_model, x)
        test_predictions(model, deserialized_model_json, x)
    else:
        test_transformed_data(model, deserialized_model, x)
        test_transformed_data(model, deserialized_model_json, x)
