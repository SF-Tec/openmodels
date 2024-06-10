# OpenModels

Welcome to OpenModels, a package that allows exporting scikit-learn model files to JSON for sharing or deploying predictive models with a peace of mind.

# Why OpenModels?

Other methods for exporting scikit-learn models require Pickle or Joblib (based on Pickle). Serializing model files with Pickle provides a simple attack vector for malicious users-- they give an attacker the ability to execute arbitrary code wherever the file is deserialized. For an example see: https://www.smartfile.com/blog/python-pickle-security-problems-and-solutions/.

OpenModels is a safe and transparent solution for exporting scikit-learn model files.

### Safe

Export model files to 100% JSON which cannot execute code on deserialization.

### Transparent

Model files are serialized in JSON (i.e., not binary), so you have the ability to see exactly what's inside.

# Getting Started

OpenModels makes exporting model files to JSON simple.

## Install

```
pip install OpenModels
```

## Example Usage

```python
import OpenModels
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0).fit(X, y)

OpenModels.model_to_json_file(model, file_path)
deserialized_model = OpenModels.model_from_json_file(file_path)

deserialized_model.predict(X)
```

# Features

The list of supported models is rapidly growing. If you have a request for a model or feature, please reach out to info@sftec.es.

OpenModels requires scikit-learn >= 1.5.0

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

We welcome contributions to OpenModels from anyone interested in improving the package. Whether you have ideas for new features, bug reports, or just want to help improve the code, we appreciate your contributions! You are also welcome to see the [Project Board]() to see what we are currently working on.

To contribute to OpenModels, please follow the [contributing guidelines](CONTRIBUTING.md).

## Running the Tests

This project uses `poetry` for dependency management and packaging. To run the tests, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
   ```

2. **Create a virtual environment and install dependencies:**

   ```bash
    poetry install
   ```

3. **Run the tests:**

   ```bash
    poetry run pytest
   ```

## License

This package is distributed under the MIT license. See the [LICENSE](LICENSE) file for more information.
