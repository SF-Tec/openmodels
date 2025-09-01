import pytest
import random
import numpy as np
from sklearn.utils.discovery import all_estimators
from sklearn.datasets import make_classification
from sklearn.feature_extraction import FeatureHasher
from openmodels.test_helpers import run_test_model
from openmodels.serializers.sklearn_serializer import NOT_SUPPORTED_ESTIMATORS
from test.test_regression import REGRESSORS
from test.test_classification import CLASSIFIERS

# Get all transformer estimators, filtering out not supported ones and those that are also regressors/classifiers
TRANSFORMERS = [
    cls for name, cls in all_estimators(type_filter="transformer")
    if name not in NOT_SUPPORTED_ESTIMATORS
    and name not in [reg.__name__ for reg in REGRESSORS]
    and name not in [clf.__name__ for clf in CLASSIFIERS]
]

@pytest.fixture(scope="module")
def data():
    # Generate dense data
    x, y = make_classification(
        n_samples=50,
        n_features=5,
        n_classes=3,
        n_informative=3,
        n_redundant=0,
        random_state=0,
        shuffle=False,
    )
    # Generate sparse data
    feature_hasher = FeatureHasher(n_features=5)
    features = [
        {
            "a": random.randint(0, 2),
            "b": random.randint(3, 5),
            "c": random.randint(6, 8),
        }
        for _ in range(0, 100)
    ]
    x_sparse = feature_hasher.transform(iter(features))
    return x, y, x_sparse

@pytest.mark.parametrize("Transformer", TRANSFORMERS)
def test_transformer(Transformer, data):
    x, y, x_sparse = data

    args = {}
    if Transformer.__name__ in ["AdditiveChi2Sampler", "MiniBatchNMF", "LatentDirichletAllocation", "NMF"]:
        x = np.abs(x)
    
    if Transformer.__name__ not in ["CCA", "GenericUnivariateSelect", "PLSCanonical", "PLSRegression", "PLSSVD", "SelectFdr", "SelectFpr", "SelectFwe", "SelectKBest", "SelectPercentile"]:
        y = None
    if Transformer.__name__ in ["CCA", "PLSCanonical"]:
        args["n_components"] = 1
    if Transformer.__name__ in ["FeatureHasher"]:
        y = None
        x = None
        x_sparse = None
    if Transformer.__name__ in ["DictVectorizer"]:
        y = None
        x = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
        x_sparse = None
    if Transformer.__name__ in ["GaussianRandomProjection"]:  
        rng = np.random.RandomState(42)
        x = rng.rand(25, 3000)
        args["random_state"] = rng
    if Transformer.__name__ in ["KernelCenterer"]:
        from sklearn.metrics.pairwise import pairwise_kernels
        X = [[ 1., -2.,  2.], [ -2.,  1.,  3.], [ 4.,  1., -2.]]
        x = pairwise_kernels(X, metric="linear")
    if Transformer.__name__ in ["PLSSVD"]:
        args["n_components"] = 1


    transformer = Transformer(**args)

    run_test_model(transformer, x, y, x_sparse, None, f"{Transformer.__name__.lower()}.json")
