"""
Scikit-learn model serializer for the OpenModels library.

This module provides a serializer for scikit-learn models, allowing them to be
converted to and from dictionary representations.
"""

from typing import Any, Dict, List, Tuple, Type, Optional
import numpy as np
from scipy.sparse import _csr, csr_matrix  # type: ignore

from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.utils.discovery import all_estimators

from openmodels.exceptions import UnsupportedEstimatorError, SerializationError
from openmodels.protocols import ModelSerializer

ALL_ESTIMATORS = {
    name: cls for name, cls in all_estimators() if issubclass(cls, BaseEstimator)
}

NOT_SUPPORTED_ESTIMATORS: list[str] = [
    # Regressors:
    "AdaBoostRegressor",  # Object of type DecisionTreeRegressor is not JSON serializable
    "BaggingRegressor",  # Object of type DecisionTreeRegressor is not JSON serializable
    "DecisionTreeRegressor",  # Object of type Tree is not JSON serializable
    "ExtraTreeRegressor",  # Object of type Tree is not JSON serializable
    "ExtraTreesRegressor",  # Object of type ExtraTreeRegressor is not JSON serializable
    "GammaRegressor",  # Object of type HalfGammaLoss is not JSON serializable
    "GaussianProcessRegressor",  # Object of type Product is not JSON serializable
    "GradientBoostingRegressor",  # Object of type DecisionTreeRegressor is not JSON serializable
    "HistGradientBoostingRegressor",  # Object of type HalfSquaredError is not JSON serializable
    "IsotonicRegression",  # Object of type interp1d is not JSON serializable
    "MultiOutputRegressor",  # MultiOutputRegressor.__init__() missing 1 required positional argument: 'estimator'
    "PoissonRegressor",  # Object of type HalfPoissonLoss is not JSON serializable
    "RandomForestRegressor",  # Object of type DecisionTreeRegressor is not JSON serializable
    "RANSACRegressor",  # Object of type LinearRegression is not JSON serializable
    "RegressorChain",  # _BaseChain.__init__() missing 1 required positional argument: 'base_estimator'
    "StackingRegressor",  # StackingRegressor.__init__() missing 1 required positional argument: 'estimators'
    "TransformedTargetRegressor",  # Object of type LinearRegression is not JSON serializable
    "DecisionTreeClassifier",  # Object of type _Tree is not JSON serializable
    "TweedieRegressor",  # Object of type HalfTweedieLossIdentity is not JSON serializable
    "VotingRegressor",  # VotingRegressor.__init__() missing 1 required positional argument: 'estimators'
    # Classifiers:
    "AdaBoostClassifier",  # Object of type DecisionTreeClassifier is not JSON serializable
    "BaggingClassifier",  # Object of type DecisionTreeClassifier is not JSON serializable
    "CalibratedClassifierCV",  # Object of type _CalibratedClassifier is not JSON serializable
    "ClassifierChain",  # ClassifierChain.__init__() missing 1 required positional argument: 'base_estimator'
    "DecisionTreeClassifier",  # Object of type _Tree is not JSON serializable
    "ExtraTreeClassifier",  # Object of type _Tree is not JSON serializable
    "ExtraTreesClassifier",  # Object of type ExtraTreeClassifier is not JSON serializable
    "FixedThresholdClassifier",  # FixedThresholdClassifier.__init__() missing 1 required positional argument: 'estimator'
    "GaussianProcessClassifier",  # Object of type OneVsRestClassifier is not JSON serializable
    "GradientBoostingClassifier",  # Object of type DecisionTreeRegressor is not JSON serializable
    "HistGradientBoostingClassifier",  # Object of type TreePredictor is not JSON serializable
    "KNeighborsClassifier",  # Object of type KDTree is not JSON serializable
    "MLPClassifier",  # Object of type LabelBinarizer is not JSON serializable
    "MultiOutputClassifier",  # MultiOutputClassifier.__init__() missing 1 required positional argument: 'estimator'
    "OneVsOneClassifier",  # OneVsOneClassifier.__init__() missing 1 required positional argument: 'estimator'
    "OneVsRestClassifier",  # OneVsOneClassifier.__init__() missing 1 required positional argument: 'estimator'
    "OutputCodeClassifier",  # OneVsOneClassifier.__init__() missing 1 required positional argument: 'estimator'
    "PassiveAggressiveClassifier",  # Object of type Hinge is not JSON serializable
    "Perceptron",  # Object of type Hinge is not JSON serializable
    "RadiusNeighborsClassifier",  # Object of type KDTree is not JSON serializable
    "RandomForestClassifier",  # Object of type DecisionTreeClassifier is not JSON serializable
    "RidgeClassifier",  # Object of type LabelBinarizer is not JSON serializable
    "RidgeClassifierCV",  # Object of type LabelBinarizer is not JSON serializable
    "SGDClassifier",  # Object of type Hinge is not JSON serializable
    "SelfTrainingClassifier",  # ValueError: You must pass an estimator to SelfTrainingClassifier. Use `estimator`.
    "StackingClassifier",  # StackingClassifier.__init__() missing 1 required positional argument: 'estimators'
    "TunedThresholdClassifierCV",  # TunedThresholdClassifierCV.__init__() missing 1 required positional argument: 'estimator'
    "VotingClassifier",  # VotingClassifier.__init__() missing 1 required positional argument: 'estimators'
    # Clusters:
    "Birch",  # Object of type _CFNode is not JSON serializable
    "BisectingKMeans",  # Object of type _BisectingTree is not JSON serializable
    "FeatureAgglomeration",  # Object of type _ArrayFunctionDispatcher is not JSON serializable
    "HDBSCAN",  # data type "[('left_node', '<i8'), ('right_node', '<i8')...]" not understood
    # Transformers:
    "ColumnTransformer",  # ColumnTransformer.__init__() missing 1 required positional argument: 'transformers'
    "DictVectorizer",  # Object of type type is not JSON serializable (in params)
    "FeatureHasher",  # openmodels.exceptions.SerializationError: Cannot serialize an unfitted model
    "FeatureUnion",  # FeatureUnion.__init__() missing 1 required positional argument: 'transformer_list'
    "GaussianRandomProjection",  # Object of type RandomState is not JSON serializable
    "GenericUnivariateSelect",  # Object of type function is not JSON serializable
    "HashingVectorizer",  # openmodels.exceptions.SerializationError: Cannot serialize an unfitted model
    "Isomap",  # Object of type KernelPCA is not JSON serializable
    "KBinsDiscretizer",  # Object of type OneHotEncoder is not JSON serializable
    "KNeighborsTransformer",  # Object of type KDTree is not JSON serializable
    "KernelPCA",  # Object of type KernelCenterer is not JSON serializable
    "LatentDirichletAllocation",  # Object of type RandomState is not JSON serializable
    "LinearDiscriminantAnalysis",  # This LinearDiscriminantAnalysis estimator requires y to be passed, but the target y is None
    "LocallyLinearEmbedding",  # Object of type NearestNeighbors is not JSON serializable"
    "LabelBinarizer",  # LabelBinarizer.fit() takes 2 positional arguments but 3 were given
    "LabelEncoder",  # LabelEncoder.fit() takes 2 positional arguments but 3 were given
    "MultiLabelBinarizer",  # MultiLabelBinarizer.fit() takes 2 positional arguments but 3 were given
    "NeighborhoodComponentsAnalysis",  # This NeighborhoodComponentsAnalysis estimator requires y to be passed, but the target y is None.
    "OneHotEncoder",  # Object of type type is not JSON serializable
    "OrdinalEncoder",  # Object of type type is not JSON serializable
    "PatchExtractor",  # ValueError: not enough values to unpack (expected 3, got 2)
    "PowerTransformer",  # Object of type StandardScaler is not JSON serializable
    "RFE",  # RFE.__init__() missing 1 required positional argument: 'estimator'
    "RFECV",  # RFECV.__init__() missing 1 required positional argument: 'estimator'
    "RadiusNeighborsTransformer",  # Object of type KDTree is not JSON serializable
    "RandomTreesEmbedding",  # Object of type ExtraTreeRegressor is not JSON serializable
    "SelectFdr",  # Object of type function is not JSON serializable
    "SelectFpr",  # Object of type function is not JSON serializable
    "SelectFromModel",  # SelectFromModel.__init__() missing 1 required positional argument: 'estimator'
    "SelectFwe",  # Object of type function is not JSON serializable
    "SelectKBest",  # Object of type function is not JSON serializable
    "SelectPercentile",  # Object of type function is not JSON serializable
    "SequentialFeatureSelector",  # SequentialFeatureSelector.__init__() missing 1 required positional argument: 'estimator'
    "SimpleImputer",  # Object of type Float64DType is not JSON serializable
    "SkewedChi2Sampler",  # ValueError: X may not contain entries smaller than -skewedness.
    "SparseCoder",  # SparseCoder.__init__() missing 1 required positional argument: 'dictionary'
    "SparseRandomProjection",  # ValueError: lead to a target dimension of 3353 which is larger than the original space with n_features=5
    "SplineTransformer",  # Object of type BSpline is not JSON serializable
    "TargetEncoder",  # ValueError: Expected array-like (array or non-string sequence), got None
    "TfidfTransformer",  # ValueError
    # Others:
    "BayesianGaussianMixture",  # Object of type ndarray is not JSON serializable
    "CountVectorizer",  # AttributeError: 'numpy.ndarray' object has no attribute 'lower'
    "FrozenEstimator",  # FrozenEstimator.__init__() missing 1 required positional argument: 'estimator'
    "GridSearchCV",  # GridSearchCV.__init__() missing 2 required positional arguments: 'estimator' and 'param_grid'
    "IsolationForest",  # Object of type ExtraTreeRegressor is not JSON serializable
    "KernelDensity",  # Object of type KDTree is not JSON serializable
    "LocalOutlierFactor",  # AttributeError: This 'LocalOutlierFactor' has no attribute 'predict'
    "Pipeline",  # Pipeline.__init__() missing 1 required positional argument: 'steps'
    "RandomizedSearchCV",  # RandomizedSearchCV.__init__() missing 2 required positional arguments: 'estimator' and 'param_distributions'
    "SGDOneClassSVM",  # Object of type Hinge is not JSON serializable
    "TfidfVectorizer",  # AttributeError: 'numpy.ndarray' object has no attribute 'lower'
]

# Dictionary of attribute exceptions
ATTRIBUTE_EXCEPTIONS: Dict[str, List] = {
    # Regressors:
    "PLSRegression": ["_x_mean", "_predict_1d"],
    "SVR": [
        "_sparse",
        "_n_support",
        "_dual_coef_",
        "_intercept_",
        "_probA",
        "_probB",
        "_gamma",
    ],
    "KNeighborsRegressor": ["_fit_method", "_fit_X", "_y"],
    "NuSVR": ["_sparse", "_gamma", "_n_support", "_probA", "_probB"],
    "TweedieRegressor": ["_base_loss"],
    "GaussianProcessRegressor": ["kernel_"],
    "HistGradientBoostingRegressor": ["_loss"],
    "RadiusNeighborsRegressor": ["_fit_method", "_fit_X", "_y"],
    "CCA": ["_x_mean", "_predict_1d"],
    "GammaRegressor": ["_base_loss"],
    "PoissonRegressor": ["_base_loss"],
    "PLSCanonical": ["_x_mean", "_predict_1d"],
    "IsotonicRegression": ["f_"],
    # Clusters:
    "BisectingKMeans": ["_bisecting_tree"],
    "KMeans": ["_n_threads"],
    "MiniBatchKMeans": ["_n_threads"],
    # Classifiers:
    "DummyClassifier": ["_strategy"],
    "HistGradientBoostingClassifier": [
        "_preprocessor",
        "_baseline_prediction",
        "_predictors",
    ],
    "MLPClassifier": ["_label_binarizer"],
    "NuSVC": ["_sparse", "_n_support", "_probA", "_probB", "_gamma"],
    "KNeighborsClassifier": ["_fit_method", "_fit_X", "_y", "_tree"],
    "RadiusNeighborsClassifier": ["_fit_method", "_fit_X", "_y", "_tree"],
    "RidgeClassifier": ["_label_binarizer"],
    "RidgeClassifierCV": ["_label_binarizer"],
    "SVC": [
        "_sparse",
        "_n_support",
        "_dual_coef_",
        "_intercept_",
        "_probA",
        "_probB",
        "_gamma",
    ],
    # Transformers:
    "KBinsDiscretizer": ["_encoder"],
    "KernelPCA": ["_centerer"],
    "KNNImputer": ["_mask_fit_X", "_valid_mask"],
    "KNeighborsTransformer": ["_fit_method", "_tree"],
    "PowerTransformer": ["_scaler"],
    "RadiusNeighborsTransformer": ["_fit_method", "_tree"],
    "SimpleImputer": ["_fit_dtype"],
    "MiniBatchNMF": ["_n_components", "_transform_max_iter", "_beta_loss", "_gamma"],
    "MissingIndicator": ["_n_features", "_precomputed"],
    "PolynomialFeatures": ["_max_degree", "_n_out_full", "_min_degree"],
    "PLSSVD": ["_x_mean", "_x_std"],
    # Others:
    "OneClassSVM": ["_sparse", "_n_support", "_probA", "_probB", "_gamma"],
}


class SklearnSerializer(ModelSerializer):
    """
    Serializer for scikit-learn estimators.

    This class provides methods to convert scikit-learn estimators to and from
    dictionary representations, which can then be used with various format converters.

    The serializer supports a wide range of scikit-learn estimators and handles
    the conversion of numpy arrays and other non-JSON-serializable types.

    Attributes
    ----------
    SUPPORTED_ESTIMATORS : Dict[str, Type[BaseEstimator]]
        A dictionary of supported scikit-learn estimator classes.
    SUPPORTED_TYPES : List[Type]
        A list of supported types for serialization.
    """

    @staticmethod
    def all_estimators(
        type_filter: Optional[str] = None,
    ) -> List[Tuple[str, Type[BaseEstimator]]]:
        """
        Get all scikit-learn supported estimators.

        Returns
        -------
        Dict[str, BaseEstimator]
            A dictionary of all scikit-learn supported estimators.
        """

        return [
            (name, cls)
            for name, cls in all_estimators(type_filter=type_filter)
            if name not in NOT_SUPPORTED_ESTIMATORS
        ]

    @staticmethod
    def _convert_to_serializable_types(value: Any) -> Any:
        """
        Convert a value to a serializable type.

        Parameters
        ----------
        value : Any
            The value to convert.

        Returns
        -------
        Any
            The serializable representation of the value.
        """
        if isinstance(value, dict):
            # Scikit-learn estimators (e.g., LogisticRegressionCV) may use non-string types
            # (such as np.int64 or float) as dictionary keys for attributes like `coefs_paths_`.
            # However, JSON serialization requires all dictionary keys to be strings.
            # The following logic ensures all dictionary keys are converted to strings
            # to guarantee compatibility with JSON serialization and deserialization.
            # NOTE: This ensures that LogisticRegressionCV works, but to put back the original
            # types we need to convert the keys back to the original types (done on fit).
            return {
                str(k): SklearnSerializer._convert_to_serializable_types(v)
                for k, v in value.items()
            }
        if isinstance(value, (np.ndarray, List)):
            return SklearnSerializer._array_to_list(value)
        if isinstance(value, _csr.csr_matrix):
            # Convert indices and indptr to int32 explicitly
            csr_value = csr_matrix(value)
            serialized_sparse_matrix = {
                "data": SklearnSerializer._array_to_list(csr_value.data),
                "indptr": SklearnSerializer._array_to_list(
                    csr_value.indptr.astype(np.int32)
                ),
                "indices": SklearnSerializer._array_to_list(
                    csr_value.indices.astype(np.int32)
                ),
                "shape": SklearnSerializer._array_to_list(csr_value.shape),
            }
            return serialized_sparse_matrix
        if isinstance(value, (np.generic)):
            # Convert numpy scalar (e.g., np.int64, np.float64) to native Python type
            return value.item()

        return value

    @staticmethod
    def _convert_to_sklearn_types(
        value: Any, attr_type: Any = "none", attr_dtype: Optional[str] = None
    ) -> Any:
        """
        Convert a JSON-deserialized value to its scikit-learn type.

        Parameters
        ----------
        value : Any
            The JSON-deserialized value.
        attr_type : str
            The target type to convert to.

        Returns
        -------
        Any
            The scikit-learn type of the value.
        """
        # Base case: if attr_type is not a list, convert value based on attr_type
        if isinstance(attr_type, str):
            type_map = {
                "int": int,
                "int64": np.int64,
                "int32": np.int32,
                "float": float,
                "float64": np.float64,
                "str": str,
                "tuple": tuple,
            }
            if attr_type == "csr_matrix":
                # Ensure all sparse matrix components are of correct dtype
                return csr_matrix(
                    (
                        np.array(value["data"], dtype=attr_dtype or np.float64),
                        np.array(value["indices"], dtype=np.int32),
                        np.array(value["indptr"], dtype=np.int32),
                    ),
                    shape=tuple(value["shape"]),
                )
            elif attr_type == "ndarray":
                return np.array(value, dtype=attr_dtype or np.float64)
            elif attr_type in type_map:
                return type_map[attr_type](value)
            # Add other types as needed
            return value  # Return as-is if no specific conversion is needed
        # Recursive case: if attr_type is a list, process each element in value
        elif isinstance(attr_type, List) and isinstance(value, List):
            return [
                SklearnSerializer._convert_to_sklearn_types(v, t, attr_dtype)
                for v, t in zip(value, attr_type)
            ]

        return value

    @staticmethod
    def _array_to_list(array: Any) -> Any:
        """
        Recursively convert numpy arrays to nested lists.

        Parameters
        ----------
        array : array-like
            The array or nested structure to convert.

        Returns
        -------
        list or Any
            The input converted to a nested list structure, or the original value if not an array.
        """
        if isinstance(array, np.ndarray):
            return SklearnSerializer._array_to_list(array.tolist())
        elif isinstance(array, list):
            return [SklearnSerializer._array_to_list(item) for item in array]
        elif isinstance(array, tuple):
            return tuple(SklearnSerializer._array_to_list(item) for item in array)
        elif isinstance(array, np.generic):
            return array.item()
        else:
            return array

    @staticmethod
    def get_nested_types(item: Any) -> Any:
        """
        Recursively determine the type of elements within nested lists.

        Parameters
        ----------
        item : Any
            The item to inspect for nested types.

        Returns
        -------
        Any
            A nested list representing the types of elements in the input item.

        Examples
        ---------

        [1, [1, 2, [1, 2, 3]], 2] -> ['int',['int','int','ndarray'],'int']

        """
        if isinstance(item, List) and item:  # If it's a list and not empty
            return [SklearnSerializer.get_nested_types(subitem) for subitem in item]
        else:
            # Return the type name if it's not a list or it's an empty list
            return type(item).__name__

    @staticmethod
    def get_dtype(value: Any) -> str:
        """
        Get the dtype of a numpy array, otherwise return empty string.
        """
        if isinstance(value, np.ndarray):
            return str(value.dtype)  # Get the actual numpy dtype
        return ""

    def serialize(self, model: BaseEstimator) -> Dict[str, Any]:
        """
        Serialize a scikit-learn estimator to a dictionary.

        This method extracts relevant attributes from the model, converts them to
        JSON-serializable types, and returns a dictionary representation of the model.

        Parameters
        ----------
        model : BaseEstimator
            The scikit-learn estimator to serialize.

        Returns
        -------
        Dict[str, Any]
            A dictionary representation of the model.

        Raises
        ------
        SerializationError
            If the model has not been fitted or if there's an error during serialization.

        Examples
        --------
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=100, n_features=20, n_classes=2)
        >>> model = LogisticRegression().fit(X, y)
        >>> serializer = SklearnSerializer()
        >>> serialized_dict = serializer.serialize(model)
        """
        try:
            check_is_fitted(model)
        except NotFittedError as e:
            raise SerializationError("Cannot serialize an unfitted model") from e

        # Get all attributes that are not private, not properties, and not callable
        # Attributes that have been estimated from the data must always have a name ending with
        # trailing underscore,
        # for example the coefficients of some regression estimator would be stored in a
        # coef_ attribute after fit has been called.
        # https://scikit-learn.org/stable/glossary.html#term-attributes
        # https://scikit-learn.org/stable/developers/develop.html#estimated-attributes
        # NOTE: This is not always true for all estimators, but it is a good starting point.
        filtered_attribute_keys = [
            key
            for key in dir(model)
            if not key.startswith("__")  # not private
            and key.endswith("_")
            and not key.endswith("__")
            and not isinstance(getattr(type(model), key, None), property)
            and not callable(getattr(model, key))
        ]

        # There are some attributes that are removed in the previous filter according to the
        # sklearn documentation.
        # However, they are still needed in the serialized model so we add them to the list.
        filtered_attribute_keys = filtered_attribute_keys + ATTRIBUTE_EXCEPTIONS.get(
            model.__class__.__name__, []
        )

        attribute_values = [getattr(model, key) for key in filtered_attribute_keys]

        # Generate attribute types with nested structure.
        # These types are used to convert the serialized attributes back to their original types.
        attribute_types = [
            SklearnSerializer.get_nested_types(value) for value in attribute_values
        ]

        attribute_dtypes_map = {
            key: SklearnSerializer.get_dtype(value)
            for key, value in zip(filtered_attribute_keys, attribute_values)
            if isinstance(value, np.ndarray)  # Only include NumPy arrays
        }

        attribute_types_map = dict(zip(filtered_attribute_keys, attribute_types))

        serializable_attribute_values = [
            self._convert_to_serializable_types(value) for value in attribute_values
        ]

        # We losely follow the ONNX standard for the serialized model.
        # https://github.com/onnx/onnx/blob/main/docs/IR.md
        return {
            "attributes": dict(
                zip(filtered_attribute_keys, serializable_attribute_values)
            ),
            "attribute_types": attribute_types_map,
            "attribute_dtypes": attribute_dtypes_map,
            "estimator_class": model.__class__.__name__,
            "params": model.get_params(),
            "producer_name": model.__module__.split(".")[0],
            "producer_version": getattr(model, "_sklearn_version", None),
            "model_version": getattr(model, "_sklearn_version", None),
            "domain": "sklearn",
        }

    def deserialize(self, data: Dict[str, Any]) -> BaseEstimator:
        """
        Deserialize a dictionary representation back into a scikit-learn estimator.

        This method reconstructs a scikit-learn estimator from its dictionary
        representation, converting attributes back to their original types.

        Parameters
        ----------
        data : Dict[str, Any]
            The dictionary representation of the model.

        Returns
        -------
        BaseEstimator
            The deserialized scikit-learn estimator.

        Raises
        ------
        UnsupportedEstimatorError
            If the estimator class is not supported.

        Examples
        --------
        >>> serializer = SklearnSerializer()
        >>> deserialized_model = serializer.deserialize(serialized_dict)
        >>> predictions = deserialized_model.predict(X_test)
        """
        estimator_class = data["estimator_class"]
        if estimator_class in NOT_SUPPORTED_ESTIMATORS:
            raise UnsupportedEstimatorError(
                f"Unsupported estimator class: {estimator_class}"
            )

        model = ALL_ESTIMATORS[estimator_class](**data["params"])

        for attribute, value in data["attributes"].items():
            # Retrieve the attribute type from data["attribute_types"]
            attr_type = data["attribute_types"].get(attribute)
            # Get dtype if available
            attr_dtype = data.get("attribute_dtypes", {}).get(attribute)
            # Pass both value and attr_type to _convert_to_sklearn_types
            setattr(
                model,
                attribute,
                self._convert_to_sklearn_types(value, attr_type, attr_dtype),
            )

        return model
