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
import random
import numpy as np
from numpy import testing
import model2json

@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=50, n_features=3, n_classes=3, n_informative=3, n_redundant=0, random_state=0, shuffle=False)

    feature_hasher = FeatureHasher(n_features=3)
    features = []
    for i in range(0, 100):
        features.append({'a': random.randint(0, 2), 'b': random.randint(3, 5), 'c': random.randint(6, 8)})
    y_sparse = [random.randint(0, 2) for i in range(0, 100)]
    X_sparse = feature_hasher.transform(features)

    return X, y, X_sparse, y_sparse

def check_model(model, X, y, abs=False):
    if abs:
        model.fit(np.absolute(X), y)
    else:
        model.fit(X, y)

    serialized_model = model2json.to_dict(model)
    deserialized_model = model2json.from_dict(serialized_model)

    if hasattr(model, 'predict'):
        expected_predictions = model.predict(X)
        actual_predictions = deserialized_model.predict(X)
        testing.assert_array_equal(expected_predictions, actual_predictions)
    else:
        expected_transformed_data = model.transform(X)
        actual_transformed_data = deserialized_model.transform(X)
        testing.assert_array_almost_equal(expected_transformed_data, actual_transformed_data)

def check_sparse_model(model, X_sparse, y_sparse, abs=False):
    if abs:
        model.fit(np.absolute(X_sparse), y_sparse)
    else:
        model.fit(X_sparse, y_sparse)

    serialized_model = model2json.to_dict(model)
    deserialized_model = model2json.from_dict(serialized_model)

    if hasattr(model, 'predict'):
        expected_predictions = model.predict(X_sparse)
        actual_predictions = deserialized_model.predict(X_sparse)
        testing.assert_array_equal(expected_predictions, actual_predictions)
    else:
        expected_transformed_data = model.transform(X_sparse)
        actual_transformed_data = deserialized_model.transform(X_sparse)
        testing.assert_array_almost_equal(expected_transformed_data, actual_transformed_data)

def check_model_json(model, model_name, X, y, abs=False):
    if abs:
        model.fit(np.absolute(X), y)
    else:
        model.fit(X, y)

    model2json.to_json(model, model_name)
    deserialized_model = model2json.from_json(model_name)

    if hasattr(model, 'predict'):
        expected_predictions = model.predict(X)
        actual_predictions = deserialized_model.predict(X)
        testing.assert_array_equal(expected_predictions, actual_predictions)
    else:
        expected_transformed_data = model.transform(X)
        actual_transformed_data = deserialized_model.transform(X)
        testing.assert_array_almost_equal(expected_transformed_data, actual_transformed_data)

def check_sparse_model_json(model, model_name, X_sparse, y_sparse, abs=False):
    if abs:
        model.fit(np.absolute(X_sparse), y_sparse)
    else:
        model.fit(X_sparse, y_sparse)

    model2json.to_json(model, model_name)
    deserialized_model = model2json.from_json(model_name)

    if hasattr(model, 'predict'):
        expected_predictions = model.predict(X_sparse)
        actual_predictions = deserialized_model.predict(X_sparse)
        testing.assert_array_equal(expected_predictions, actual_predictions)
    else:
        expected_transformed_data = model.transform(X_sparse)
        actual_transformed_data = deserialized_model.transform(X_sparse)
        testing.assert_array_almost_equal(expected_transformed_data, actual_transformed_data)

def test_bernoulli_nb(data):
    X, y, X_sparse, y_sparse = data
    check_model(BernoulliNB(), X, y)
    check_sparse_model(BernoulliNB(), X_sparse, y_sparse)
    model_name = 'bernoulli-nb.json'
    check_model_json(BernoulliNB(), model_name, X, y)
    check_sparse_model_json(BernoulliNB(), model_name, X_sparse, y_sparse)

def test_gaussian_nb(data):
    X, y, _, _ = data
    check_model(GaussianNB(), X, y)
    model_name = 'gaussian-nb.json'
    check_model_json(GaussianNB(), model_name, X, y)

def test_multinomial_nb(data):
    X, y, X_sparse, y_sparse = data
    check_model(MultinomialNB(), X, y, abs=True)
    check_sparse_model(MultinomialNB(), X_sparse, y_sparse, abs=True)
    model_name = 'multinomial-nb.json'
    check_model_json(MultinomialNB(), model_name, X, y, abs=True)
    check_sparse_model_json(MultinomialNB(), model_name, X_sparse, y_sparse, abs=True)

def test_complement_nb(data):
    X, y, _, _ = data
    check_model(ComplementNB(), X, y, abs=True)
    model_name = 'complement-nb.json'
    check_model_json(ComplementNB(), model_name, X, y, abs=True)

def test_logistic_regression(data):
    X, y, X_sparse, y_sparse = data
    check_model(LogisticRegression(), X, y)
    check_sparse_model(LogisticRegression(), X_sparse, y_sparse)
    model_name = 'lr.json'
    check_model_json(LogisticRegression(), model_name, X, y)
    check_sparse_model_json(LogisticRegression(), model_name, X_sparse, y_sparse)

def test_lda(data):
    X, y, _, _ = data
    check_model(discriminant_analysis.LinearDiscriminantAnalysis(), X, y)
    model_name = 'lda.json'
    check_model_json(discriminant_analysis.LinearDiscriminantAnalysis(), model_name, X, y)

def test_qda(data):
    X, y, _, _ = data
    check_model(discriminant_analysis.QuadraticDiscriminantAnalysis(), X, y)
    model_name = 'qda.json'
    check_model_json(discriminant_analysis.QuadraticDiscriminantAnalysis(), model_name, X, y)

def test_svm(data):
    X, y, X_sparse, y_sparse = data
    check_model(svm.SVC(gamma=0.001, C=100., kernel='linear'), X, y)
    check_sparse_model(svm.SVC(gamma=0.001, C=100., kernel='linear'), X_sparse, y_sparse)
    model_name = 'svm.json'
    check_model_json(svm.SVC(), model_name, X, y)
    check_sparse_model_json(svm.SVC(), model_name, X_sparse, y_sparse)

def test_decision_tree(data):
    X, y, X_sparse, y_sparse = data
    check_model(DecisionTreeClassifier(), X, y)
    check_sparse_model(DecisionTreeClassifier(), X_sparse, y_sparse)
    model_name = 'dt.json'
    check_model_json(DecisionTreeClassifier(), model_name, X, y)
    check_sparse_model_json(DecisionTreeClassifier(), model_name, X_sparse, y_sparse)

def test_gradient_boosting(data):
    X, y, X_sparse, y_sparse = data
    check_model(GradientBoostingClassifier(n_estimators=25, learning_rate=1.0), X, y)
    check_sparse_model(GradientBoostingClassifier(n_estimators=25, learning_rate=1.0), X_sparse, y_sparse)
    model_name = 'gb.json'
    check_model_json(GradientBoostingClassifier(), model_name, X, y)
    check_sparse_model_json(GradientBoostingClassifier(), model_name, X_sparse, y_sparse)

def test_random_forest(data):
    X, y, X_sparse, y_sparse = data
    check_model(RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0), X, y)
    check_sparse_model(RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0), X_sparse, y_sparse)
    model_name = 'rf.json'
    check_model_json(RandomForestClassifier(), model_name, X, y)
    check_sparse_model_json(RandomForestClassifier(), model_name, X_sparse, y_sparse)

def test_perceptron(data):
    X, y, X_sparse, y_sparse = data
    check_model(Perceptron(), X, y)
    check_sparse_model(Perceptron(), X_sparse, y_sparse)
    model_name = 'perceptron.json'
    check_model_json(Perceptron(), model_name, X, y)
    check_sparse_model_json(Perceptron(), model_name, X_sparse, y_sparse)
def test_mlp(data):
    X, y, X_sparse, y_sparse = data
    check_model(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), X, y)
    check_sparse_model(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), X_sparse, y_sparse)
    model_name = 'mlp.json'
    check_model_json(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), model_name, X, y)
    check_sparse_model_json(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), model_name, X_sparse, y_sparse)

def test_pca(data):
    X, _, X_sparse, _ = data
    check_model(PCA(n_components=2), X, X)
    check_sparse_model(PCA(n_components=2), X_sparse, X_sparse)
    model_name = 'pca.json'
    check_model_json(PCA(n_components=2), model_name, X, X)
    check_sparse_model_json(PCA(n_components=2), model_name, X_sparse, X_sparse)
