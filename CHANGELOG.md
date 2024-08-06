# Changelog

All notable changes to the OpenModels project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha.1] - 2024-08-06

### Added

- Initial release of OpenModels library
- Core functionality for serializing and deserializing machine learning models
- Support for scikit-learn models:
  - Classification: LogisticRegression, RandomForestClassifier, SVC, BernoulliNB, GaussianNB, MultinomialNB, ComplementNB, Perceptron
  - Regression: LinearRegression, Lasso, Ridge, RandomForestRegressor, SVR
  - Clustering: KMeans
  - Dimensionality Reduction: PCA
  - Other: PLSRegression
- JSON serialization format
- Pickle serialization format
- Extensible architecture for adding new model types and serialization formats
- Basic test suite for supported models
- Documentation including README, LICENSE, and CONTRIBUTING guidelines

### Changed

- N/A (First release)

### Deprecated

- N/A (First release)

### Removed

- N/A (First release)

### Fixed

- N/A (First release)

### Security

- Implemented safe alternatives to pickle serialization

## [Unreleased]

### Planned

- Support for TensorFlow models
- YAML serialization format
- Enhanced documentation with more examples and use cases
- Improved test coverage
- Support for more scikit-learn models including ensemble methods and neural networks
