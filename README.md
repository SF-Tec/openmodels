# model2json

Welcome to model2json, a package that allows exporting scikit-learn model files to JSON for sharing or deploying predictive models with a peace of mind.

# Why model2json?

Other methods for exporting scikit-learn models require Pickle or Joblib (based on Pickle). Serializing model files with Pickle provides a simple attack vector for malicious users-- they give an attacker the ability to execute arbitrary code wherever the file is deserialized. For an example see: https://www.smartfile.com/blog/python-pickle-security-problems-and-solutions/.

model2json is a safe and transparent solution for exporting scikit-learn model files.

### Safe

Export model files to 100% JSON which cannot execute code on deserialization.

### Transparent

Model files are serialized in JSON (i.e., not binary), so you have the ability to see exactly what's inside.

# Getting Started

model2json makes exporting model files to JSON simple.

## Install

```
pip install model2json
```

## Example Usage

```python
import model2json
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0).fit(X, y)

model2json.to_json(model, file_name)
deserialized_model = model2json.from_json(file_name)

deserialized_model.predict(X)
```

# Features

The list of supported models is rapidly growing. If you have a request for a model or feature, please reach out to info@sftec.es.

model2json requires scikit-learn >= 1.5.0

## Currently Supported scikit-learn Models

- Classification

- `sklearn.naive_bayes.BernoulliNB`
- `sklearn.naive_bayes.ComplementNB`
- `sklearn.naive_bayes.GaussianNB`
- `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` (LDA)
- `sklearn.linear_model.LogisticRegression`
- `sklearn.naive_bayes.MultinomialNB`
- `sklearn.decomposition.PCA`
- `sklearn.linear_model.Perceptron`

- Regression
- `sklearn.linear_model.Lasso`
- `sklearn.linear_model.LinearRegression`
- `sklearn.cross_decomposition.PLSRegression`
- `sklearn.ensemble.RandomForestRegressor`
- `sklearn.linear_model.Ridge`
