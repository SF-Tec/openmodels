"""
Scikit-learn model serializer for the OpenModels library.

This module provides a serializer for scikit-learn models, allowing them to be
converted to and from dictionary representations.
"""

from typing import Any, Callable, Dict, List, Tuple, Type, Optional
import numpy as np
import inspect
from scipy.sparse import _csr, csr_matrix  # type: ignore
from scipy.stats._distn_infrastructure import rv_continuous_frozen  # type: ignore
import scipy.stats  # type: ignore

import sklearn
from sklearn._loss.loss import HalfGammaLoss
from sklearn.tree._tree import Tree
from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.utils.discovery import all_estimators

from openmodels.exceptions import UnsupportedEstimatorError, SerializationError
from openmodels.protocols import ModelSerializer
import warnings

ConverterFunc = Callable[[Any], Any]

ALL_ESTIMATORS = {
    name: cls for name, cls in all_estimators() if issubclass(cls, BaseEstimator)
}

TESTED_VERSIONS = ["1.6.1", "1.7.1"]

NOT_SUPPORTED_ESTIMATORS: list[str] = [
    # Regressors:
    # "GammaRegressor",  # Object of type HalfGammaLoss is not JSON serializable
    # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/_loss/loss.py
    "GaussianProcessRegressor",  # Object of type Product is not JSON serializable
    "GradientBoostingRegressor",  # Object of type RandomState is not JSON serializable
    "HistGradientBoostingRegressor",  # Object of type HalfSquaredError is not JSON serializable
    "IsotonicRegression",  # Object of type interp1d is not JSON serializable
    "PoissonRegressor",  # Object of type HalfPoissonLoss is not JSON serializable
    "TweedieRegressor",  # Object of type HalfTweedieLossIdentity is not JSON serializable
    # Classifiers:
    "CalibratedClassifierCV",  # Object of type _CalibratedClassifier is not JSON serializable
    "GaussianProcessClassifier",  # Object of type OneVsRestClassifier is not JSON serializable
    "GradientBoostingClassifier",  # Object of type RandomState is not JSON serializable
    "HistGradientBoostingClassifier",  # Object of type TreePredictor is not JSON serializable
    "KNeighborsClassifier",  # Object of type KDTree is not JSON serializable
    "MLPClassifier",  # Object of type LabelBinarizer is not JSON serializable
    "OneVsOneClassifier",  # AttributeError: 'dict' object has no attribute 'predict'
    "OneVsRestClassifier",  # openmodels.exceptions.UnsupportedEstimatorError: Unsupported estimator class: LabelBinarizer
    "OutputCodeClassifier",  # AttributeError: 'dict' object has no attribute 'predict_proba'
    "PassiveAggressiveClassifier",  # Object of type Hinge is not JSON serializable
    "Perceptron",  # Object of type Hinge is not JSON serializable
    "RadiusNeighborsClassifier",  # Object of type KDTree is not JSON serializable
    "RidgeClassifier",  # openmodels.exceptions.UnsupportedEstimatorError: Unsupported estimator class: LabelBinarizer
    "RidgeClassifierCV",  # openmodels.exceptions.UnsupportedEstimatorError: Unsupported estimator class: LabelBinarizer
    "SGDClassifier",  # Object of type Hinge is not JSON serializable
    "StackingClassifier",  # openmodels.exceptions.UnsupportedEstimatorError: Unsupported estimator class: LabelEncoder
    "TunedThresholdClassifierCV",  # TypeError: Object of type _CurveScorer is not JSON serializable
    "VotingClassifier",  # openmodels.exceptions.UnsupportedEstimatorError: Unsupported estimator class: LabelEncoder
    # Clusters:
    "Birch",  # Object of type _CFNode is not JSON serializable
    "BisectingKMeans",  # Object of type _BisectingTree is not JSON serializable
    "FeatureAgglomeration",  # Object of type _ArrayFunctionDispatcher is not JSON serializable
    "HDBSCAN",  # data type "[('left_node', '<i8'), ('right_node', '<i8')...]" not understood
    # Transformers:
    "ColumnTransformer",  # AttributeError: 'dict' object has no attribute 'transform'
    "DictVectorizer",  # ValueError:
    "FeatureHasher",  # openmodels.exceptions.SerializationError: Cannot serialize an unfitted model
    "FeatureUnion",  # AttributeError: 'dict' object has no attribute 'transform'
    "GaussianRandomProjection",  # Object of type RandomState is not JSON serializable
    "GenericUnivariateSelect",  # Object of type function is not JSON serializable
    "HashingVectorizer",  # openmodels.exceptions.SerializationError: Cannot serialize an unfitted model
    "Isomap",  # Object of type KDTree is not JSON serializable
    "KBinsDiscretizer",  # Object of type Float64DType is not JSON serializable
    "KNeighborsTransformer",  # Object of type KDTree is not JSON serializable
    "LatentDirichletAllocation",  # Object of type RandomState is not JSON serializable
    "LinearDiscriminantAnalysis",  # This LinearDiscriminantAnalysis estimator requires y to be passed, but the target y is None
    "LocallyLinearEmbedding",  # Object of type NearestNeighbors is not JSON serializable"
    "LabelBinarizer",  # LabelBinarizer.fit() takes 2 positional arguments but 3 were given
    "LabelEncoder",  # LabelEncoder.fit() takes 2 positional arguments but 3 were given
    "MultiLabelBinarizer",  # MultiLabelBinarizer.fit() takes 2 positional arguments but 3 were given
    "NeighborhoodComponentsAnalysis",  # This NeighborhoodComponentsAnalysis estimator requires y to be passed, but the target y is None.
    "OneHotEncoder",  # ValueError:
    "PatchExtractor",  # ValueError: not enough values to unpack (expected 3, got 2)
    "RadiusNeighborsTransformer",  # Object of type KDTree is not JSON serializable
    "RandomTreesEmbedding",  # openmodels.exceptions.UnsupportedEstimatorError: Unsupported estimator class: OneHotEncoder
    "SelectFdr",  # Object of type function is not JSON serializable
    "SelectFpr",  # Object of type function is not JSON serializable
    "SelectFwe",  # Object of type function is not JSON serializable
    "SelectKBest",  # Object of type function is not JSON serializable
    "SelectPercentile",  # Object of type function is not JSON serializable
    "SimpleImputer",  # Object of type Float64DType is not JSON serializable
    "SkewedChi2Sampler",  # ValueError: X may not contain entries smaller than -skewedness.
    "SparseRandomProjection",  # ValueError: lead to a target dimension of 3353 which is larger than the original space with n_features=5
    "SplineTransformer",  # Object of type BSpline is not JSON serializable
    "TargetEncoder",  # ValueError: Expected array-like (array or non-string sequence), got None
    "TfidfTransformer",  # ValueError
    # Others:
    "BayesianGaussianMixture",  # Object of type ndarray is not JSON serializable
    "CountVectorizer",  # AttributeError: 'numpy.ndarray' object has no attribute 'lower'
    "IsolationForest",  # TypeError: only integer scalar arrays can be converted to a scalar index
    "KernelDensity",  # Object of type KDTree is not JSON serializable
    "LocalOutlierFactor",  # AttributeError: This 'LocalOutlierFactor' has no attribute 'predict'
    "NearestNeighbors",  # Object of type KDTree is not JSON serializable
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
    "TransformedTargetRegressor": ["_training_dim"],
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
    "StackingClassifier": ["_label_encoder"],
    "SVC": [
        "_sparse",
        "_n_support",
        "_dual_coef_",
        "_intercept_",
        "_probA",
        "_probB",
        "_gamma",
    ],
    "TunedThresholdClassifierCV": ["_curve_scorer"],
    # Transformers:
    "ColumnTransformer": ["_columns", "_remainder"],
    "OneHotEncoder": [
        "_infrequent_enabled",
        "_drop_idx_after_grouping",
        "_n_features_outs",
    ],
    "OrdinalEncoder": ["_missing_indices", "_infrequent_enabled"],
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
    "IsolationForest": [
        "_max_features",
        "_max_samples",
        "_decision_path_lengths",
        "_average_path_length_per_tree",
    ],
    "OneClassSVM": ["_sparse", "_n_support", "_probA", "_probB", "_gamma"],
    "NearestNeighbors": ["_fit_method", "_tree"],
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

    def check_version(self, stored_version: Optional[str]) -> None:
        """
        Check compatibility between stored scikit-learn version and the current environment.

        Parameters
        ----------
        stored_version : str
            The scikit-learn version recorded during serialization.

        Notes
        -----
        - Issues a warning if the stored version does not match the current version.
        - Mentions the baseline supported version (1.7.1).
        - Does nothing if no version is stored (for backward compatibility).
        """
        if not stored_version:
            return  # No version info available

        current_version = sklearn.__version__
        if stored_version != current_version:
            warnings.warn(
                f"Version mismatch detected in sklearn deserialization:\n"
                f"- Model serialized with scikit-learn {stored_version}\n"
                f"- Current environment: scikit-learn {current_version}\n\n"
                f"OpenModels has been tested under {TESTED_VERSIONS}. ",
                UserWarning,
            )

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

    def _serialize_tree(self, tree: Tree) -> Dict[str, Any]:
        """
        Serializes a sklearn.tree._tree.Tree object to a dictionary.

        Parameters:
            tree (sklearn.tree._tree.Tree): The internal tree structure from a fitted tree-based model (e.g., model.tree_)

        Returns:
            dict: Serialized tree attributes
        """
        state = tree.__getstate__()

        return {
            "n_features": tree.n_features,
            "n_outputs": tree.n_outputs,
            "n_classes": tree.n_classes.tolist(),
            "state": {
                k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in state.items()
            },
            "nodes_dtype": [list(t) for t in state["nodes"].dtype.descr],  # for JSON
        }

    def _deserialize_tree(self, tree_data: Dict[str, Any]) -> Tree:
        """
        Deserializes a dictionary representation of a tree back to a sklearn.tree._tree.Tree object.

        """
        tree = Tree(
            tree_data["n_features"],
            np.array(tree_data["n_classes"], dtype=np.intp),
            tree_data["n_outputs"],
        )

        state = {}
        for key, value in tree_data["state"].items():
            if key == "nodes":
                # Restore dtype
                nodes_dtype_descr = [
                    tuple(field) for field in tree_data["nodes_dtype"] if field[0] != ""
                ]
                nodes_dtype = np.dtype(nodes_dtype_descr)
                if isinstance(value, list) and isinstance(value[0], list):
                    value = [tuple(row) for row in value]
                state["nodes"] = np.array(value, dtype=nodes_dtype)
            else:
                state[key] = np.array(value)

        tree.__setstate__(state)
        return tree

    def _serialize_type_object(self, value: Type) -> Dict[str, Any]:
        """
        Serializes a Python type object to a dictionary.

        """
        return {"__type__": True, "type_name": value.__name__}

    def _serialize_slice_object(self, value: slice) -> Dict[str, Any]:
        """
        Serializes a slice object to a dictionary.
        """
        return {
            "__slice__": True,
            "start": value.start,
            "stop": value.stop,
            "step": value.step,
        }

    def _serialize_csr_matrix(self, value: _csr.csr_matrix) -> Dict[str, Any]:
        """
        Serializes a sparse CSR matrix to a dictionary.
        """
        csr_value = csr_matrix(value)
        serialized_sparse_matrix = {
            "data": self._array_to_list(csr_value.data),
            "indptr": self._array_to_list(csr_value.indptr.astype(np.int32)),
            "indices": self._array_to_list(csr_value.indices.astype(np.int32)),
            "shape": self._array_to_list(csr_value.shape),
        }
        return serialized_sparse_matrix

    def _serialize_scipy_dist(self, value: rv_continuous_frozen) -> Dict[str, Any]:
        """
        Serializes a scipy.stats.rv_continuous_frozen object to a dictionary.
        """
        return {
            "__scipy_dist__": True,
            "dist_name": value.dist.name,
            "args": value.args,
            "kwargs": value.kwds,
        }

    def _convert_to_serializable_types(self, value: Any) -> Any:
        """
        Converts a value to a serializable type.
        """

        # Dispatch table
        handlers = [
            (HalfGammaLoss, lambda _: {"__half_gamma_loss__": True}),
            (rv_continuous_frozen, self._serialize_scipy_dist),
            (type, self._serialize_type_object),
            (slice, self._serialize_slice_object),
            (Tree, self._serialize_tree),
            (_csr.csr_matrix, self._serialize_csr_matrix),
            (np.generic, lambda v: v.item()),
        ]

        for typ, handler in handlers:
            if isinstance(value, typ):
                return handler(value)

        if isinstance(value, BaseEstimator):
            # If the value is a BaseEstimator, serialize it using SklearnSerializer
            # This allows for nested estimators to be serialized correctly
            # Check if this is the unfitted estimator template
            try:
                check_is_fitted(value)
                return self.serialize(value)
            except NotFittedError:
                return {
                    "__template__": value.__class__.__name__,
                    "params": self._convert_to_serializable_types(value.get_params()),
                }

        if isinstance(value, dict):
            # Scikit-learn estimators (e.g., LogisticRegressionCV) may use non-string types
            # (such as np.int64 or float) as dictionary keys for attributes like `coefs_paths_`.
            # However, JSON serialization requires all dictionary keys to be strings.
            # The following logic ensures all dictionary keys are converted to strings
            # to guarantee compatibility with JSON serialization and deserialization.
            # NOTE: This ensures that LogisticRegressionCV works, but to put back the original
            # types we need to convert the keys back to the original types (done on fit).
            return {
                str(k): self._convert_to_serializable_types(v) for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            # If the list contains tuples of (name, estimator, columns), serialize the estimator
            if value and all(
                isinstance(item, tuple)
                and len(item) == 3
                and isinstance(item[1], BaseEstimator)
                for item in value
            ):
                return [
                    (item[0], self._convert_to_serializable_types(item[1]), item[2])
                    for item in value
                ]
            # If the list contains tuples of (name, estimator), serialize each estimator
            if value and all(
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[1], BaseEstimator)
                for item in value
            ):
                return [
                    (item[0], self._convert_to_serializable_types(item[1]))
                    for item in value
                ]
            # If the list contains BaseEstimator objects, serialize each one
            if value and all(isinstance(item, BaseEstimator) for item in value):
                return [self._convert_to_serializable_types(item) for item in value]
            # Otherwise use the default array to list conversion
            return self._array_to_list(value)

        if isinstance(value, (np.ndarray)):
            # Special handling for arrays of estimators
            if value.dtype == np.dtype("O") and value.size > 0:
                first_elem = value.ravel()[0]
                if isinstance(first_elem, BaseEstimator):
                    # This is an array of estimators, serialize each one
                    return [
                        [self._convert_to_serializable_types(est) for est in row]
                        for row in value
                    ]
            # Regular array handling
            return self._array_to_list(value)

        return value

    def _convert_to_sklearn_types(
        self, value: Any, attr_type: Any = "none", attr_dtype: Optional[str] = None
    ) -> Any:
        """
        Convert a JSON-deserialized value to its scikit-learn type.

        """
        if isinstance(value, dict) and "__half_gamma_loss__" in value:
            return HalfGammaLoss()

        if isinstance(value, dict) and value.get("__scipy_dist__"):
            dist = getattr(scipy.stats, value["dist_name"])
            return dist(*value["args"], **value["kwargs"])

        if isinstance(value, dict) and "estimator_class" in value:
            # Recursively deserialize fitted estimators
            return self.deserialize(value)

        if isinstance(value, dict) and value.get("__type__"):
            # Deserialize Python type objects from their string name
            return getattr(
                __builtins__, value["type_name"], float
            )  # Default to float if not found

        if isinstance(value, dict) and value.get("__slice__"):
            # Deserialize slice objects
            return slice(value["start"], value["stop"], value["step"])

        if isinstance(value, dict) and "__template__" in value:
            estimator_class = ALL_ESTIMATORS[value["__template__"]]
            params = value.get("params", {})
            # Recursively convert all params
            for k in params:
                params[k] = self._convert_to_sklearn_types(params[k], None, None)
            return estimator_class(**params)

        # Recursive case: if attr_type is a list, process each element in value
        if isinstance(attr_type, list) and isinstance(value, list):
            return [
                self._convert_to_sklearn_types(v, t, attr_dtype)
                for v, t in zip(value, attr_type)
            ]

        if isinstance(attr_type, str):
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
            converters: Dict[str, ConverterFunc] = {
                "int": int,
                "int64": np.int64,
                "int32": np.int32,
                "float": float,
                "float64": np.float64,
                "str": str,
                "tuple": tuple,
                "ndarray": lambda x: np.array(
                    x, dtype=attr_dtype if attr_dtype else None
                ),
            }

            if attr_type in converters:
                return converters[attr_type](value)
            if attr_type in ALL_ESTIMATORS:
                # This is an estimator type
                if isinstance(value, dict):
                    if "__template__" in value:
                        estimator_class = ALL_ESTIMATORS[value["__template__"]]
                        return estimator_class(**value["params"])
                    return self.deserialize(value)

            return value  # Return as-is if no specific conversion is needed

        return value

    def _array_to_list(self, array: Any) -> Any:
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
            return self._array_to_list(array.tolist())
        elif isinstance(array, list):
            return [self._array_to_list(item) for item in array]
        elif isinstance(array, tuple):
            return tuple(self._array_to_list(item) for item in array)
        elif isinstance(array, np.generic):
            return array.item()
        else:
            return array

    def get_nested_types(self, item: Any) -> Any:
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
            return [self.get_nested_types(subitem) for subitem in item]
        elif isinstance(item, BaseEstimator):
            # For estimators, return their class name instead of just 'BaseEstimator'
            return item.__class__.__name__
        else:
            # Return the type name if it's not a list or it's an empty list
            return type(item).__name__

    def get_dtype(self, value: Any) -> str:
        """
        Get the dtype of a numpy array, otherwise return empty string.
        """
        if isinstance(value, np.ndarray):
            return str(value.dtype)  # Get the actual numpy dtype
        elif isinstance(value, (list, tuple)) and value:
            # If it's a list/tuple that will become an ndarray, check its elements
            first_elem = value[0]
            if isinstance(first_elem, (int, np.integer)):
                return "int32"  # Use int32 for integer lists
            elif isinstance(first_elem, (float, np.floating)):
                return "float64"  # Use float64 for float lists
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
        attribute_types = [self.get_nested_types(value) for value in attribute_values]

        attribute_dtypes_map = {
            key: self.get_dtype(value)
            for key, value in zip(filtered_attribute_keys, attribute_values)
            if isinstance(value, np.ndarray)  # Only include NumPy arrays
        }

        attribute_types_map = dict(zip(filtered_attribute_keys, attribute_types))

        serializable_attribute_values = [
            self._convert_to_serializable_types(value) for value in attribute_values
        ]

        # Serialize model parameters
        params = model.get_params()
        serializable_params = self._convert_to_serializable_types(params)
        param_types = {
            param_name: self.get_nested_types(param_value)
            for param_name, param_value in params.items()
        }
        param_dtypes = {
            param_name: self.get_dtype(param_value)
            for param_name, param_value in params.items()
            if isinstance(param_value, np.ndarray)
            or (isinstance(param_value, (list, tuple)) and param_value)
        }

        # We losely follow the ONNX standard for the serialized model.
        # https://github.com/onnx/onnx/blob/main/docs/IR.md
        result = {
            "attributes": dict(
                zip(filtered_attribute_keys, serializable_attribute_values)
            ),
            "attribute_types": attribute_types_map,
            "attribute_dtypes": attribute_dtypes_map,
            "estimator_class": model.__class__.__name__,
            "params": serializable_params,
            "param_types": param_types,
            "param_dtypes": param_dtypes,
            "producer_name": model.__module__.split(".")[0],
            "producer_version": getattr(model, "_sklearn_version", None),
            "model_version": getattr(model, "_sklearn_version", None),
            "domain": "sklearn",
        }
        print(result)
        return {
            "attributes": dict(
                zip(filtered_attribute_keys, serializable_attribute_values)
            ),
            "attribute_types": attribute_types_map,
            "attribute_dtypes": attribute_dtypes_map,
            "estimator_class": model.__class__.__name__,
            "params": serializable_params,
            "param_types": param_types,
            "param_dtypes": param_dtypes,
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
        # Version control check
        self.check_version(data.get("producer_version"))

        estimator_class = data["estimator_class"]
        if estimator_class in NOT_SUPPORTED_ESTIMATORS:
            raise UnsupportedEstimatorError(
                f"Unsupported estimator class: {estimator_class}"
            )

        # Reconstruct params with correct types/dtypes
        params = data.get("params", {})
        param_types = data.get("param_types", {})
        param_dtypes = data.get("param_dtypes", {})

        # Get valid constructor arguments for the estimator
        estimator_cls = ALL_ESTIMATORS[estimator_class]
        valid_args = list(inspect.signature(estimator_cls.__init__).parameters.keys())
        # Remove 'self' if present
        valid_args = [arg for arg in valid_args if arg != "self"]

        reconstructed_params = {}
        for param_name, param_value in params.items():
            # Only include params that are valid constructor arguments
            if param_name not in valid_args:
                continue
            param_type = param_types.get(param_name)
            param_dtype = param_dtypes.get(param_name)
            # Special handling for Pipeline steps
            if estimator_class == "Pipeline" and param_name == "steps":
                reconstructed_params[param_name] = [
                    (name, self._convert_to_sklearn_types(est, None, None))
                    for name, est in param_value
                ]
            else:
                reconstructed_params[param_name] = self._convert_to_sklearn_types(
                    param_value, param_type, param_dtype
                )
        model = estimator_cls(**reconstructed_params)

        for attribute, value in data["attributes"].items():
            attr_type = data["attribute_types"].get(attribute)
            attr_dtype = data.get("attribute_dtypes", {}).get(attribute)
            # Handle tree_ separately
            if attr_type == "Tree":
                model.tree_ = self._deserialize_tree(value)
                continue
            # Use _convert_to_sklearn_types for all attributes
            setattr(
                model,
                attribute,
                self._convert_to_sklearn_types(value, attr_type, attr_dtype),
            )

        return model
