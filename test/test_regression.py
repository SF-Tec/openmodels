import random
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_regression
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import ARDRegression,LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from openmodels.test_helpers import run_test_model, ensure_correct_sparse_format


@pytest.fixture(scope="module")
def data():
    x, y = make_regression(  # type: ignore
        n_samples=50,
        n_features=3,
        n_informative=3,
        random_state=0,
        shuffle=False,
    )

    feature_hasher = FeatureHasher(n_features=3)
    features = []
    for _ in range(0, 100):
        features.append(
            {
                "a": random.randint(0, 2),
                "b": random.randint(3, 5),
                "c": random.randint(6, 8),
            }
        )
    y_sparse = [random.random() for i in range(0, 100)]
    x_sparse = feature_hasher.transform(iter(features))

    return x, y, x_sparse, y_sparse


# Test each model
def test_ard_regression(data):
    x, y, x_sparse, y_sparse = data
    # Convert sparse matrix to dense for ARDRegression
    x_sparse_dense = x_sparse.toarray()
    run_test_model(
        ARDRegression(),
        x,
        y,
        x_sparse_dense,
        y_sparse,
        "ard-regression.json"
    )

def test_linear_regression(data):
    x, y, x_sparse, y_sparse = data
    run_test_model(
        LinearRegression(), x, y, x_sparse, y_sparse, "linear-regression.json"
    )


def test_lasso_regression(data):
    x, y, x_sparse, y_sparse = data
    run_test_model(Lasso(alpha=0.1), x, y, x_sparse, y_sparse, "lasso-regression.json")


def test_ridge_regression(data):
    x, y, x_sparse, y_sparse = data
    run_test_model(Ridge(alpha=0.5), x, y, x_sparse, y_sparse, "ridge-regression.json")


def test_svr(data):
    x, y, x_sparse, y_sparse = data
    # Ensure sparse data is properly formatted before testing
    x_sparse = ensure_correct_sparse_format(x_sparse)

    run_test_model(
        SVR(gamma="scale", C=1.0, epsilon=0.2), x, y, x_sparse, y_sparse, "svr.json"
    )


@pytest.mark.skip(reason="Feature not ready")
def test_decision_tree_regression(data):
    x, y, x_sparse, y_sparse = data
    run_test_model(
        DecisionTreeRegressor(),
        x,
        y,
        x_sparse,
        y_sparse,
        "decision-tree-regression.json",
    )


@pytest.mark.skip(reason="Feature not ready")
def test_gradient_boosting_regression(data):
    x, y, x_sparse, y_sparse = data
    run_test_model(
        GradientBoostingRegressor(),
        x,
        y,
        x_sparse,
        y_sparse,
        "gradient-boosting-regression.json",
    )


@pytest.mark.skip(reason="Feature not ready")
def test_random_forest_regression(data):
    x, y, x_sparse, y_sparse = data
    run_test_model(
        RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100),
        x,
        y,
        x_sparse,
        y_sparse,
        "random-forest-regression.json",
    )


def test_mlp_regression(data):
    x, y, x_sparse, y_sparse = data
    run_test_model(MLPRegressor(), x, y, x_sparse, y_sparse, "mlp-regression.json")


def test_pls_regression(data):
    x, y, _, _ = data
    run_test_model(
        PLSRegression(n_components=2), x, y, None, None, "pls-regression.json"  # type: ignore
    )
