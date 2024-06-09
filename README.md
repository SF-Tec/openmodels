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

model2json.model_to_json_file(model, file_path)
deserialized_model = model2json.model_from_json_file(file_path)

deserialized_model.predict(X)
```

# Features

The list of supported models is rapidly growing. If you have a request for a model or feature, please reach out to info@sftec.es.

model2json requires scikit-learn >= 1.5.0

## Currently Supported scikit-learn Models

Classification

- `sklearn.naive_bayes.BernoulliNB`
- `sklearn.naive_bayes.ComplementNB`
- `sklearn.naive_bayes.GaussianNB`
- `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` (LDA)
- `sklearn.linear_model.LogisticRegression`
- `sklearn.naive_bayes.MultinomialNB`
- `sklearn.decomposition.PCA`
- `sklearn.linear_model.Perceptron`

Regression

- `sklearn.linear_model.Lasso`
- `sklearn.linear_model.LinearRegression`
- `sklearn.cross_decomposition.PLSRegression`
- `sklearn.linear_model.Ridge`

## Contributing

We welcome contributions to model2json from anyone interested in improving the package. Whether you have ideas for new features, bug reports, or just want to help improve the code, we appreciate your contributions! You are also welcome to see the [Project Board]() to see what we are currently working on.

To contribute to model2json, please follow the [contributing guidelines](CONTRIBUTING.md).

## License

This package is distributed under the MIT license. See the [LICENSE](LICENSE) file for more information.
