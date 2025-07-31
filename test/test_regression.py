import pytest
import random
import numpy as np
from sklearn.utils.discovery import all_estimators
from sklearn.datasets import make_regression
from sklearn.feature_extraction import FeatureHasher
from openmodels.test_helpers import run_test_model
from openmodels.serializers.sklearn_serializer import NOT_SUPPORTED_ESTIMATORS

# Get all regressor estimators, filtering out not supported regressors
REGRESSORS = [cls for name, cls in all_estimators(type_filter="regressor")
    if name not in NOT_SUPPORTED_ESTIMATORS]

@pytest.fixture(scope="module")
def data():
    x, y = make_regression(  # type: ignore
        n_samples=50,
        n_features=3,
        n_informative=3,
        random_state=42,
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


@pytest.mark.parametrize("Regressor", REGRESSORS)
def test_regressor(Regressor, data):
    x, y, x_sparse, y_sparse  = data

    args = {}
    # Handle specific regressors that require special treatment
    if Regressor.__name__ in ["CCA", "PLSCanonical"]:
        args["n_components"] = 1
    elif Regressor.__name__ in ["GammaRegressor", "PoissonRegressor"]:
        y = np.abs(y) + 1e-3
    elif Regressor.__name__ == "IsotonicRegression":
        x, y = make_regression(n_samples=10, n_features=1, random_state=41)
        x_sparse = None
        y_sparse = None
    elif Regressor.__name__ == "PLSRegression":
        x = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
        y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]] 
        x_sparse = None
        y_sparse = None
    elif Regressor.__name__ in ["MultiTaskElasticNet", "MultiTaskElasticNetCV", "MultiTaskLasso", "MultiTaskLassoCV"]:
        x, y = make_regression(n_samples=50, n_features=3, n_targets=2, random_state=42)
        x_sparse = None
        y_sparse = None
    
    regressor = Regressor(**args)

    try:
        # Try with sparse input
        run_test_model(regressor, x, y, x_sparse, y_sparse, f"{Regressor.__name__.lower()}.json")
    except TypeError as e:
        if "Sparse data was passed" in str(e):
            # Retry with dense input
            run_test_model(regressor, x, y, x_sparse.toarray(), y_sparse, f"{Regressor.__name__.lower()}.json")
        else:
            raise