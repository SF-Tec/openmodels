import random
import pytest
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.feature_extraction import FeatureHasher
from sklearn import svm, discriminant_analysis
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from model2json.test_helpers import test_model

# Define constants
N_SAMPLES = 50
N_FEATURES = 3
N_CLASSES = 3
N_INFORMATIVE = 3
N_REDUNDANT = 0
RANDOM_STATE = 0
N_ESTIMATORS = 10
MAx_DEPTH = 5
LEARNING_RATE = 1.0
SOLVER = "lbfgs"
ALPHA = 1e-5
HIDDEN_LAYER_SIZES = (5, 2)
N_COMPONENTS = 2


@pytest.fixture(scope="module")
def data():
    # Generate test data
    x, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        n_informative=N_INFORMATIVE,
        n_redundant=N_REDUNDANT,
        random_state=RANDOM_STATE,
        shuffle=False,
    )

    feature_hasher = FeatureHasher(n_features=N_FEATURES)
    features = []
    for _ in range(0, 100):
        features.append(
            {
                "a": random.randint(0, 2),
                "b": random.randint(3, 5),
                "c": random.randint(6, 8),
            }
        )
    y_sparse = [random.randint(0, 2) for i in range(0, 100)]
    x_sparse = feature_hasher.transform(features)

    return x, y, x_sparse, y_sparse


# Test each model
def test_bernoulli_nb(data):
    x, y, x_sparse, y_sparse = data
    test_model(BernoulliNB(), x, y, x_sparse, y_sparse, "bernoulli-nb.json")


def test_gaussian_nb(data):
    x, y, _, _ = data
    test_model(GaussianNB(), x, y, None, None, "gaussian-nb.json")


def test_multinomial_nb(data):
    x, y, x_sparse, y_sparse = data
    test_model(
        MultinomialNB(), x, y, x_sparse, y_sparse, "multinomial-nb.json", abs=True
    )


def test_complement_nb(data):
    x, y, _, _ = data
    test_model(ComplementNB(), x, y, None, None, "complement-nb.json", abs=True)


def test_logistic_regression(data):
    x, y, x_sparse, y_sparse = data
    test_model(LogisticRegression(), x, y, x_sparse, y_sparse, "lr.json")


def test_lda(data):
    x, y, _, _ = data
    test_model(
        discriminant_analysis.LinearDiscriminantAnalysis(), x, y, None, None, "lda.json"
    )


def test_qda(data):
    x, y, _, _ = data
    test_model(
        discriminant_analysis.QuadraticDiscriminantAnalysis(),
        x,
        y,
        None,
        None,
        "qda.json",
    )


def test_svm(data):
    x, y, x_sparse, y_sparse = data
    test_model(
        svm.SVC(gamma=0.001, C=100.0, kernel="linear"),
        x,
        y,
        x_sparse,
        y_sparse,
        "svm.json",
    )


def test_decision_tree(data):
    x, y, x_sparse, y_sparse = data
    test_model(DecisionTreeClassifier(), x, y, x_sparse, y_sparse, "dt.json")


def test_gradient_boosting(data):
    x, y, x_sparse, y_sparse = data
    test_model(
        GradientBoostingClassifier(
            n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE
        ),
        x,
        y,
        x_sparse,
        y_sparse,
        "gb.json",
    )


def test_random_forest(data):
    x, y, x_sparse, y_sparse = data
    test_model(
        RandomForestClassifier(
            n_estimators=N_ESTIMATORS, max_depth=MAx_DEPTH, random_state=RANDOM_STATE
        ),
        x,
        y,
        x_sparse,
        y_sparse,
        "rf.json",
    )


def test_perceptron(data):
    x, y, x_sparse, y_sparse = data
    test_model(Perceptron(), x, y, x_sparse, y_sparse, "perceptron.json")


def test_mlp(data):
    x, y, x_sparse, y_sparse = data
    test_model(
        MLPClassifier(
            solver=SOLVER,
            alpha=ALPHA,
            hidden_layer_sizes=HIDDEN_LAYER_SIZES,
            random_state=RANDOM_STATE,
        ),
        x,
        y,
        x_sparse,
        y_sparse,
        "mlp.json",
    )


def test_pca(data):
    x, _, x_sparse, _ = data
    test_model(PCA(n_components=N_COMPONENTS), x, x, x_sparse, x_sparse, "pca.json")
