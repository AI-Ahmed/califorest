"""
Calibrated Random Forest Classifier Module.

This module implements a calibrated random forest classifier that uses 
train-test split for probability calibration.
"""

from typing import Optional, Union, Literal

import numpy as np
from numpy.typing import NDArray
from loguru import logger

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


# Constants
DEFAULT_N_ESTIMATORS = 30
DEFAULT_MAX_DEPTH = 3
DEFAULT_MIN_SAMPLES_SPLIT = 2
DEFAULT_MIN_SAMPLES_LEAF = 1
DEFAULT_TEST_SIZE = 0.3
SUPPORTED_CALIBRATION_TYPES = {"logistic", "isotonic"}
SUPPORTED_CRITERIA = {"gini", "entropy"}


class CalibratedForest(ClassifierMixin, BaseEstimator):
    """A calibrated random forest classifier using train-test split for calibration.
    
    This classifier combines a RandomForestClassifier with probability calibration.
    Unlike CalibratedTree which uses out-of-bag predictions, this implementation
    uses a traditional train-test split approach for calibration.
    
    Parameters
    ----------
    n_estimators : int, default=30
        The number of trees in the random forest. Must be positive.
    max_depth : int, default=3
        The maximum depth of the trees. Must be positive.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    ctype : {"isotonic", "logistic"}, default="isotonic"
        The calibration method to use:
        - "isotonic": Isotonic regression calibration
        - "logistic": Logistic regression calibration
    test_size : float, default=0.3
        The proportion of the dataset to include in the test split for calibration.
        Must be between 0.0 and 1.0.
    verbose : bool, default=False
        Whether to enable verbose logging during training.
    random_state : int or None, default=None
        Random state seed for reproducibility.
    **forest_kwargs : dict
        Additional keyword arguments passed to RandomForestClassifier constructor.
        Common parameters include:
        - criterion : {"gini", "entropy"}, default="gini"
            The function to measure the quality of a split.
        - max_features : {"sqrt", "log2", None, int, float}, default="sqrt"
            The number of features to consider when looking for the best split.
        - bootstrap : bool, default=True
            Whether bootstrap samples are used when building trees.
        - class_weight : dict, list of dict or "balanced", default=None
            Weights associated with classes for handling imbalanced datasets.
        - And other RandomForestClassifier constructor parameters.
        
    Attributes
    ----------
    model_ : RandomForestClassifier
        The fitted random forest model.
    calibrator_ : Union[LogisticRegression, IsotonicRegression]
        The fitted calibration model.
    n_features_in_ : int
        Number of features seen during fit.
    is_fitted_ : bool
        Whether the model has been fitted.
        
    Notes
    -----
    This implementation uses a simpler train-test split approach compared to
    CalibratedTree's out-of-bag methodology. While this may use less data
    for training, it provides a cleaner separation between model fitting
    and calibration.
    
    Examples
    --------
    >>> from califorest import CalibratedForest
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=10, 
    ...                           n_classes=2, random_state=42)
    >>> # Basic usage
    >>> clf = CalibratedForest(n_estimators=50, random_state=42)
    >>> clf.fit(X, y)
    >>> probabilities = clf.predict_proba(X)
    >>> predictions = clf.predict(X)
    >>> 
    >>> # With additional forest parameters
    >>> clf_advanced = CalibratedForest(
    ...     n_estimators=50, 
    ...     criterion='entropy',
    ...     class_weight='balanced',
    ...     max_features='log2',
    ...     random_state=42
    ... )
    >>> clf_advanced.fit(X, y)
    
    References
    ----------
    .. [1] Platt, J. (1999). Probabilistic outputs for support vector machines 
           and comparisons to regularized likelihood methods.
    .. [2] Zadrozny, B. & Elkan, C. (2002). Transforming classifier scores 
           into accurate multiclass probability estimates.
    """
    
    def __init__(
        self,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_depth: int = DEFAULT_MAX_DEPTH,
        min_samples_split: int = DEFAULT_MIN_SAMPLES_SPLIT,
        min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
        ctype: Literal["isotonic", "logistic"] = "isotonic",
        test_size: float = DEFAULT_TEST_SIZE,
        verbose: bool = False,
        random_state: Optional[int] = None,
        **forest_kwargs
    ) -> None:
        """Initialize the CalibratedForest classifier."""
        # Validate parameters
        self._validate_init_parameters(
            n_estimators, max_depth, min_samples_split, min_samples_leaf,
            ctype, test_size
        )
        
        # Set parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ctype = ctype
        self.test_size = test_size
        self.verbose = verbose
        self.random_state = random_state
        self.forest_kwargs = forest_kwargs
        
        # Initialize attributes
        self.model_: Optional[RandomForestClassifier] = None
        self.calibrator_: Optional[Union[LogisticRegression, IsotonicRegression]] = None
        self.n_features_in_: Optional[int] = None
        self.is_fitted_: bool = False
    
    def _validate_init_parameters(
        self,
        n_estimators: int,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
        ctype: str,
        test_size: float
    ) -> None:
        """Validate initialization parameters.
        
        Parameters
        ----------
        n_estimators : int
            Number of estimators to validate.
        max_depth : int
            Maximum depth to validate.
        min_samples_split : int
            Minimum samples for split to validate.
        min_samples_leaf : int
            Minimum samples per leaf to validate.
        ctype : str
            Calibration type to validate.
        test_size : float
            Test size proportion to validate.
            
        Raises
        ------
        ValueError
            If any parameter is invalid.
        TypeError
            If any parameter has wrong type.
        """
        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError(f"n_estimators must be a positive integer, got {n_estimators}")
            
        if not isinstance(max_depth, int) or max_depth <= 0:
            raise ValueError(f"max_depth must be a positive integer, got {max_depth}")
            
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError(f"min_samples_split must be >= 2, got {min_samples_split}")
            
        if not isinstance(min_samples_leaf, int) or min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >= 1, got {min_samples_leaf}")
            
        if ctype not in SUPPORTED_CALIBRATION_TYPES:
            raise ValueError(f"ctype must be one of {SUPPORTED_CALIBRATION_TYPES}, got '{ctype}'")
            
        if not isinstance(test_size, (int, float)) or not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")
    

    
    def _setup_calibrator(self) -> Union[LogisticRegression, IsotonicRegression]:
        """Setup the calibrator based on the calibration type.
        
        Returns
        -------
        Union[LogisticRegression, IsotonicRegression]
            The configured calibrator.
        """
        if self.ctype == "logistic":
            return LogisticRegression(
                C=1e20,  # High C for minimal regularization
                solver="lbfgs",
                max_iter=1000,
                random_state=self.random_state
            )
        elif self.ctype == "isotonic":
            return IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                out_of_bounds="clip"
            )
        else:
            # This should never happen due to validation, but keeping for safety
            raise ValueError(f"Unsupported calibration type: {self.ctype}")
    
    def _create_model(self) -> RandomForestClassifier:
        """Create the random forest model.
        
        Returns
        -------
        RandomForestClassifier
            The configured random forest model.
        """
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            **self.forest_kwargs
        )
    
    def fit(
        self, 
        X: NDArray, 
        y: NDArray, 
        sample_weight: Optional[NDArray] = None
    ) -> 'CalibratedForest':
        """Fit the calibrated random forest.
        
        Parameters
        ----------
        X : NDArray of shape (n_samples, n_features)
            Training features.
        y : NDArray of shape (n_samples,)
            Training labels (binary: 0 or 1).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights for training.
            
        Returns
        -------
        CalibratedForest
            The fitted estimator.
            
        Raises
        ------
        ValueError
            If the input data is invalid or not binary classification.
        """
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False)
        
        # Check for binary classification
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError(
                f"CalibratedForest only supports binary classification. "
                f"Found {len(unique_labels)} unique labels: {unique_labels}"
            )
        
        # Ensure labels are 0 and 1
        if not np.array_equal(unique_labels, [0, 1]):
            raise ValueError(f"Labels must be 0 and 1, got {unique_labels}")
        
        # Store number of features
        self.n_features_in_ = X.shape[1]
        
        if self.verbose:
            logger.info(f"Starting training with {X.shape[0]} samples and {X.shape[1]} features")
            logger.info(f"Using {self.ctype} calibration with test_size={self.test_size}")
        
        # Validate sample_weight if provided
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(
                    f"sample_weight has {sample_weight.shape[0]} samples, "
                    f"but X has {X.shape[0]} samples"
                )
        
        # Create model and calibrator
        self.model_ = self._create_model()
        self.calibrator_ = self._setup_calibrator()
        
        # Split data for training and calibration
        if self.verbose:
            logger.info("Splitting data for training and calibration...")
        
        split_params = {'test_size': self.test_size, 'random_state': self.random_state, 'stratify': y}
        
        # Handle sample weights in split if provided
        if sample_weight is not None:
            X_train, X_cal, y_train, y_cal, sw_train, sw_cal = train_test_split(
                X, y, sample_weight, **split_params
            )
            train_fit_params = {'sample_weight': sw_train}
        else:
            X_train, X_cal, y_train, y_cal = train_test_split(X, y, **split_params)
            train_fit_params = {}
        
        if self.verbose:
            logger.info(f"Training forest on {X_train.shape[0]} samples...")
        
        # Train the random forest
        self.model_.fit(X_train, y_train, **train_fit_params)
        
        if self.verbose:
            logger.info(f"Calibrating on {X_cal.shape[0]} samples...")
        
        # Get predictions for calibration
        y_pred_proba = self.model_.predict_proba(X_cal)
        
        # Fit calibrator
        if self.ctype == "logistic":
            # Logistic regression expects 2D input
            self.calibrator_.fit(y_pred_proba[:, [1]], y_cal)
        elif self.ctype == "isotonic":
            # Isotonic regression expects 1D input
            self.calibrator_.fit(y_pred_proba[:, 1], y_cal)
        
        self.is_fitted_ = True
        
        if self.verbose:
            logger.info("Training completed successfully")
        
        return self
    
    def predict_proba(self, X: NDArray) -> NDArray:
        """Predict class probabilities.
        
        Parameters
        ----------
        X : NDArray of shape (n_samples, n_features)
            Input features.
            
        Returns
        -------
        NDArray of shape (n_samples, 2)
            Predicted class probabilities. First column is probability of class 0,
            second column is probability of class 1.
        """
        # Validate input and check if fitted
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, 'is_fitted_')
        
        # Check feature consistency
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but CalibratedForest was fitted with "
                f"{self.n_features_in_} features"
            )
        
        # Get forest predictions
        forest_probabilities = self.model_.predict_proba(X)
        
        # Apply calibration
        if self.ctype == "logistic":
            # Logistic regression expects 2D input and returns 2D output
            calibrated_probabilities = self.calibrator_.predict_proba(
                forest_probabilities[:, [1]]
            )
            return calibrated_probabilities
        elif self.ctype == "isotonic":
            # Isotonic regression expects 1D input and returns 1D output
            calibrated_prob_class1 = self.calibrator_.predict(forest_probabilities[:, 1])
            
            # Ensure probabilities are in valid range
            calibrated_prob_class1 = np.clip(calibrated_prob_class1, 0.0, 1.0)
            
            # Create probability matrix
            n_samples = X.shape[0]
            probability_matrix = np.zeros((n_samples, 2), dtype=np.float64)
            probability_matrix[:, 1] = calibrated_prob_class1
            probability_matrix[:, 0] = 1.0 - calibrated_prob_class1
            
            return probability_matrix
        else:
            raise ValueError(f"Unknown calibration type: {self.ctype}")
    
    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels.
        
        Parameters
        ----------
        X : NDArray of shape (n_samples, n_features)
            Input features.
            
        Returns
        -------
        NDArray of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1).astype(np.int64)
    
    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, return the parameters for this estimator and contained 
            subobjects that are estimators.
            
        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'ctype': self.ctype,
            'test_size': self.test_size,
            'verbose': self.verbose,
            'random_state': self.random_state,
            **self.forest_kwargs
        }
    
    def set_params(self, **params) -> 'CalibratedForest':
        """Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Parameter names and their new values.
            
        Returns
        -------
        CalibratedForest
            The estimator instance.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
        return self
    
    def __repr__(self) -> str:
        """Return string representation of the estimator."""
        params = self.get_params()
        param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
        return f"CalibratedForest({param_str})"


