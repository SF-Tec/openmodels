import pytest
from sklearn import cross_decomposition
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_regression
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from numpy import testing
import random
import model2json


@pytest.fixture(scope="module")
def data():
    X, y = make_regression(
        n_samples=50, n_features=3, n_informative=3, random_state=0, shuffle=False
    )

    feature_hasher = FeatureHasher(n_features=3)
    features = []
    for i in range(0, 100):
        features.append(
            {
                "a": random.randint(0, 2),
                "b": random.randint(3, 5),
                "c": random.randint(6, 8),
            }
        )
    y_sparse = [random.random() for i in range(0, 100)]
    X_sparse = feature_hasher.transform(features)

    return X, y, X_sparse, y_sparse


def check_model(model, X, y):
    model.fit(X, y)
    serialized_model = model2json.to_dict(model)
    deserialized_model = model2json.from_dict(serialized_model)
    expected_predictions = model.predict(X)
    actual_predictions = deserialized_model.predict(X)
    testing.assert_array_equal(expected_predictions, actual_predictions)


def test_linear_regression(data):
    X, y, X_sparse, y_sparse = data
    check_model(LinearRegression(), X, y)
    check_model(LinearRegression(), X_sparse, y_sparse)


def test_lasso_regression(data):
    X, y, X_sparse, y_sparse = data
    check_model(Lasso(alpha=0.1), X, y)
    check_model(Lasso(alpha=0.1), X_sparse, y_sparse)


def test_ridge_regression(data):
    X, y, X_sparse, y_sparse = data
    check_model(Ridge(alpha=0.5), X, y)
    check_model(Ridge(alpha=0.5), X_sparse, y_sparse)


def test_svr(data):
    X, y, X_sparse, y_sparse = data
    check_model(SVR(gamma="scale", C=1.0, epsilon=0.2), X, y)
    check_model(SVR(gamma="scale", C=1.0, epsilon=0.2), X_sparse, y_sparse)


def test_decision_tree_regression(data):
    X, y, X_sparse, y_sparse = data
    check_model(DecisionTreeRegressor(), X, y)
    check_model(DecisionTreeRegressor(), X_sparse, y_sparse)


def test_gradient_boosting_regression(data):
    X, y, X_sparse, y_sparse = data
    check_model(GradientBoostingRegressor(), X, y)
    check_model(GradientBoostingRegressor(), X_sparse, y_sparse)


def test_random_forest_regression(data):
    X, y, X_sparse, y_sparse = data
    check_model(
        RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100), X, y
    )
    check_model(
        RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100),
        X_sparse,
        y_sparse,
    )


def test_mlp_regression(data):
    X, y, X_sparse, y_sparse = data
    check_model(MLPRegressor(), X, y)
    check_model(MLPRegressor(), X_sparse, y_sparse)


def test_pls_regression(data):
    X, y, _, _ = data
    check_model(PLSRegression(n_components=2), X, y)
