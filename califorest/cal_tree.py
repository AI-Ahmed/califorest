"""
Calibrated Tree Ensemble Classifier Module.

This module implements a calibrated tree ensemble classifier that uses out-of-bag
predictions for probability calibration with Bayesian priors.
"""

import warnings
from typing import Optional, Union, Tuple, List, Literal

import numpy as np
from numpy.typing import NDArray
from loguru import logger

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


# Constants
DEFAULT_N_ESTIMATORS = 300
DEFAULT_MAX_DEPTH = 5
DEFAULT_MIN_SAMPLES_SPLIT = 2
DEFAULT_MIN_SAMPLES_LEAF = 1
DEFAULT_ALPHA0 = 100
DEFAULT_BETA0 = 25
SUPPORTED_CALIBRATION_TYPES = {"logistic", "isotonic"}
SUPPORTED_CRITERIA = {"gini", "entropy"}


class CalibratedTree(ClassifierMixin, BaseEstimator):
    """A calibrated tree ensemble classifier using out-of-bag predictions.
    
    This classifier combines multiple decision trees and uses out-of-bag (OOB)
    predictions for probability calibration. The calibration process employs
    Bayesian methods with Beta priors to improve probability estimates.
    
    Parameters
    ----------
    n_estimators : int, default=300
        The number of trees in the ensemble. Must be positive.
    criterion : {"gini", "entropy"}, default="gini"
        The criterion used for splitting nodes in decision trees.
    max_depth : int, default=5
        The maximum depth of the decision trees. Must be positive.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    ctype : {"isotonic", "logistic"}, default="isotonic"
        The calibration method to use:
        - "isotonic": Isotonic regression calibration
        - "logistic": Logistic regression calibration
    alpha0 : float, default=100
        Prior alpha parameter for Beta distribution in Bayesian calibration.
        Must be positive.
    beta0 : float, default=25
        Prior beta parameter for Beta distribution in Bayesian calibration.
        Must be positive.
    verbose : bool, default=False
        Whether to enable verbose logging during training.
    random_state : int or None, default=None
        Random state seed for reproducibility.
    **tree_kwargs : dict
        Additional keyword arguments passed to DecisionTreeClassifier constructor.
        Common parameters include:
        - class_weight : dict, list of dict or "balanced", default=None
            Weights associated with classes for handling imbalanced datasets.
        - ccp_alpha : non-negative float, default=0.0
            Complexity parameter for Minimal Cost-Complexity Pruning.
        - max_leaf_nodes : int, default=None
            Maximum number of leaf nodes for tree complexity control.
        - min_impurity_decrease : float, default=0.0
            Minimum impurity decrease required for node splitting.
        - And other DecisionTreeClassifier constructor parameters.
        
    Attributes
    ----------
    estimators_ : List[DecisionTreeClassifier]
        The fitted decision tree estimators.
    calibrator_ : Union[LogisticRegression, IsotonicRegression]
        The fitted calibration model.
    n_features_in_ : int
        Number of features seen during fit.
    classes_ : NDArray
        The classes labels (always [0, 1] for binary classification).
    n_classes_ : int
        The number of classes (always 2 for binary classification).
    n_outputs_ : int
        The number of outputs (always 1 for binary classification).
    feature_importances_ : NDArray
        The averaged feature importances across all estimators.
    is_fitted_ : bool
        Whether the model has been fitted.
        
    Notes
    -----
    The calibration process uses a Bayesian approach where the weights for
    calibration are computed as:
    
    .. math::
        w_i = \\frac{\\alpha_i}{\\beta_i}
        
    where :math:`\\alpha_i = \\alpha_0 + n_{oob,i}/2` and 
    :math:`\\beta_i = \\beta_0 + \\text{Var}(y_{oob,i}) \\cdot n_{oob,i}/2`.
    
    Examples
    --------
    >>> from califorest import CalibratedTree
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=10, 
    ...                           n_classes=2, random_state=42)
    >>> # Basic usage
    >>> clf = CalibratedTree(n_estimators=100, random_state=42)
    >>> clf.fit(X, y)
    >>> probabilities = clf.predict_proba(X)
    >>> predictions = clf.predict(X)
    >>> 
    >>> # With additional tree parameters
    >>> clf_advanced = CalibratedTree(
    ...     n_estimators=100, 
    ...     class_weight='balanced',
    ...     ccp_alpha=0.01,
    ...     random_state=42
    ... )
    >>> clf_advanced.fit(X, y)
    
    References
    ----------
    .. [1] Y. Park and J. C. Ho. 2020. **CaliForest: Calibrated Random Forest for Health Data**.
     *ACM Conference on Health, Inference, and Learning (2020)*
    """
    
    def __init__(
        self,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        criterion: str = "gini",
        max_depth: int = DEFAULT_MAX_DEPTH,
        min_samples_split: int = DEFAULT_MIN_SAMPLES_SPLIT,
        min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
        ctype: Literal["isotonic", "logistic"] = "isotonic",
        alpha0: float = DEFAULT_ALPHA0,
        beta0: float = DEFAULT_BETA0,
        verbose: bool = False,
        random_state: Optional[int] = None,
        **tree_kwargs
    ) -> None:
        """Initialize the CalibratedTree classifier."""
        # Validate parameters
        self._validate_init_parameters(
            n_estimators, criterion, max_depth, min_samples_split,
            min_samples_leaf, ctype, alpha0, beta0
        )
        
        # Set parameters
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ctype = ctype
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.verbose = verbose
        self.random_state = random_state
        self.tree_kwargs = tree_kwargs
        
        # Initialize attributes
        self.estimators_: List[DecisionTreeClassifier] = []
        self.calibrator_: Optional[Union[LogisticRegression, IsotonicRegression]] = None
        self.n_features_in_: Optional[int] = None
        self.is_fitted_: bool = False
        self.classes_: Optional[NDArray] = None
        self.n_classes_: Optional[int] = None
    
    def _validate_init_parameters(
        self,
        n_estimators: int,
        criterion: str,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
        ctype: str,
        alpha0: float,
        beta0: float
    ) -> None:
        """Validate initialization parameters.
        
        Parameters
        ----------
        n_estimators : int
            Number of estimators to validate.
        criterion : str
            Splitting criterion to validate.
        max_depth : int
            Maximum depth to validate.
        min_samples_split : int
            Minimum samples for split to validate.
        min_samples_leaf : int
            Minimum samples per leaf to validate.
        ctype : str
            Calibration type to validate.
        alpha0 : float
            Alpha prior to validate.
        beta0 : float
            Beta prior to validate.
            
        Raises
        ------
        ValueError
            If any parameter is invalid.
        TypeError
            If any parameter has wrong type.
        """
        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError(f"n_estimators must be a positive integer, got {n_estimators}")
            
        if criterion not in SUPPORTED_CRITERIA:
            raise ValueError(f"criterion must be one of {SUPPORTED_CRITERIA}, got '{criterion}'")
            
        if not isinstance(max_depth, int) or max_depth <= 0:
            raise ValueError(f"max_depth must be a positive integer, got {max_depth}")
            
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError(f"min_samples_split must be >= 2, got {min_samples_split}")
            
        if not isinstance(min_samples_leaf, int) or min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >= 1, got {min_samples_leaf}")
            
        if ctype not in SUPPORTED_CALIBRATION_TYPES:
            raise ValueError(f"ctype must be one of {SUPPORTED_CALIBRATION_TYPES}, got '{ctype}'")
            
        if not isinstance(alpha0, (int, float)) or alpha0 <= 0:
            raise ValueError(f"alpha0 must be positive, got {alpha0}")
            
        if not isinstance(beta0, (int, float)) or beta0 <= 0:
            raise ValueError(f"beta0 must be positive, got {beta0}")
    

    
    def _setup_calibrator(self) -> Union[LogisticRegression, IsotonicRegression]:
        """Setup the calibrator based on the calibration type.
        
        Returns
        -------
        Union[LogisticRegression, IsotonicRegression]
            The configured calibrator.
        """
        if self.ctype == "logistic":
            return LogisticRegression(
                penalty=None,
                solver="saga",
                max_iter=5000,
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
    
    def _create_estimators(self) -> List[DecisionTreeClassifier]:
        """Create the ensemble of decision tree estimators.
        
        Returns
        -------
        List[DecisionTreeClassifier]
            List of configured but unfitted decision trees.
        """
        return [
            DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features="sqrt",  # More efficient than "auto"
                random_state=self.random_state if self.random_state is None 
                            else self.random_state + i,
                **self.tree_kwargs
            )
            for i in range(self.n_estimators)
        ]
    
    def _generate_bootstrap_indices(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """Generate bootstrap indices and out-of-bag masks.
        
        Parameters
        ----------
        n_samples : int
            Number of samples in the dataset.
            
        Returns
        -------
        Tuple[NDArray[np.int_], NDArray[np.bool_]]
            Bootstrap indices and out-of-bag boolean masks.
        """
        # Set random state for reproducibility
        rng = np.random.default_rng(self.random_state)
        
        bootstrap_indices = np.zeros((n_samples, self.n_estimators), dtype=np.int_)
        oob_masks = np.full((n_samples, self.n_estimators), True, dtype=bool)
        
        for estimator_idx in range(self.n_estimators):
            # Generate bootstrap sample indices
            bootstrap_idx = rng.choice(n_samples, size=n_samples, replace=True)
            bootstrap_indices[:, estimator_idx] = bootstrap_idx
            
            # Mark in-bag samples as False in OOB mask
            oob_masks[bootstrap_idx, estimator_idx] = False
        
        return bootstrap_indices, oob_masks
    
    def _calculate_oob_predictions(
        self, 
        X: NDArray, 
        y: NDArray,
        fit_params: dict
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Calculate out-of-bag predictions and related statistics.
        
        Parameters
        ----------
        X : NDArray[np.floating]
            Training features.
        y : NDArray[np.integer]
            Training labels.
        fit_params : dict
            Additional parameters to pass to individual estimator fit methods.
            Includes sample_weight and other parameters from the main fit call.
            
        Returns
        -------
        Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.integer]]
            Mean OOB predictions, prediction variances, and OOB counts.
        """
        n_samples = X.shape[0]
        
        # Initialize arrays
        oob_predictions = np.full((n_samples, self.n_estimators), np.nan, dtype=np.float64)
        oob_counts = np.zeros(n_samples, dtype=np.int64)
        
        # Generate bootstrap indices and OOB masks
        bootstrap_indices, oob_masks = self._generate_bootstrap_indices(n_samples)
        
        if self.verbose:
            logger.info(f"Training {self.n_estimators} estimators...")
        
        # Train estimators and collect OOB predictions
        for estimator_idx, estimator in enumerate(self.estimators_):
            # Get bootstrap and OOB indices for this estimator
            bootstrap_idx = bootstrap_indices[:, estimator_idx]
            oob_mask = oob_masks[:, estimator_idx]
            
            # Prepare fit parameters for this bootstrap sample
            bootstrap_fit_params = {}
            for param_name, param_value in fit_params.items():
                if param_name == 'sample_weight' and param_value is not None:
                    # Bootstrap the sample weights
                    bootstrap_fit_params[param_name] = param_value[bootstrap_idx]
                elif param_name not in ['X', 'y']:  # Exclude X, y as they're handled separately
                    # Pass other parameters as-is
                    bootstrap_fit_params[param_name] = param_value
            
            # Train estimator on bootstrap sample with appropriate parameters
            estimator.fit(X[bootstrap_idx], y[bootstrap_idx], **bootstrap_fit_params)
            
            # Get OOB predictions
            if np.any(oob_mask):
                oob_indices = np.where(oob_mask)[0]
                oob_probas = estimator.predict_proba(X[oob_indices])
                
                # Store positive class probabilities
                oob_predictions[oob_indices, estimator_idx] = oob_probas[:, 1]
                oob_counts[oob_indices] += 1
            
            if self.verbose and (estimator_idx + 1) % 50 == 0:
                logger.info(f"Trained {estimator_idx + 1}/{self.n_estimators} estimators")
        
        # Filter samples with sufficient OOB predictions
        sufficient_oob_mask = oob_counts > 1
        n_sufficient = np.sum(sufficient_oob_mask)
        
        if n_sufficient == 0:
            raise ValueError("No samples have sufficient out-of-bag predictions for calibration")
        
        if n_sufficient < 0.1 * n_samples:
            warnings.warn(
                f"Only {n_sufficient}/{n_samples} samples have sufficient OOB predictions. "
                "Consider increasing n_estimators or reducing min_samples_leaf.",
                UserWarning
            )
        
        # Calculate statistics for samples with sufficient OOB predictions
        filtered_predictions = oob_predictions[sufficient_oob_mask]
        filtered_counts = oob_counts[sufficient_oob_mask]
        
        mean_predictions = np.nanmean(filtered_predictions, axis=1)
        prediction_variances = np.nanvar(filtered_predictions, axis=1)
        
        return mean_predictions, prediction_variances, filtered_counts, sufficient_oob_mask
    
    def _calculate_calibration_weights(
        self, 
        prediction_variances: NDArray, 
        oob_counts: NDArray
    ) -> NDArray:
        """Calculate Bayesian calibration weights.
        
        Parameters
        ----------
        prediction_variances : NDArray[np.floating]
            Variances of OOB predictions for each sample.
        oob_counts : NDArray[np.integer]
            Number of OOB predictions for each sample.
            
        Returns
        -------
        NDArray[np.floating]
            Calibration weights for each sample.
        """
        # Bayesian weight calculation with Beta priors
        alpha = self.alpha0 + oob_counts / 2.0
        beta = self.beta0 + prediction_variances * oob_counts / 2.0
        
        # Avoid division by zero
        beta = np.maximum(beta, 1e-10)
        weights = alpha / beta
        
        return weights
    
    def fit(
        self, 
        X: NDArray, 
        y: NDArray, 
        sample_weight: Optional[NDArray] = None,
        check_input: bool = True
    ) -> 'CalibratedTree':
        """Fit the calibrated tree ensemble.
        
        Parameters
        ----------
        X : NDArray of shape (n_samples, n_features)
            Training features.
        y : NDArray of shape (n_samples,)
            Training labels (binary: 0 or 1).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights for training. If provided, each sample will be 
            weighted accordingly during bootstrap sampling and tree training.
        check_input : bool, default=True
            Allow to bypass input checking for performance optimization.
            
        Returns
        -------
        CalibratedTree
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
                f"CalibratedTree only supports binary classification. "
                f"Found {len(unique_labels)} unique labels: {unique_labels}"
            )
        
        # Ensure labels are 0 and 1
        if not np.array_equal(unique_labels, [0, 1]):
            raise ValueError(f"Labels must be 0 and 1, got {unique_labels}")
        
        # Store number of features and classes
        self.n_features_in_ = X.shape[1]
        self.classes_ = unique_labels
        self.n_classes_ = len(unique_labels)
        
        if self.verbose:
            logger.info(f"Starting training with {X.shape[0]} samples and {X.shape[1]} features")
            logger.info(f"Using {self.ctype} calibration")
        
        # Setup calibrator
        self.calibrator_ = self._setup_calibrator()
        
        # Validate sample_weight if provided
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(
                    f"sample_weight has {sample_weight.shape[0]} samples, "
                    f"but X has {X.shape[0]} samples"
                )
        
        # Create fit parameters dictionary
        fit_params = {}
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
        if not check_input:
            fit_params['check_input'] = check_input
        
        # Create estimators
        self.estimators_ = self._create_estimators()
        
        # Calculate OOB predictions and statistics
        (mean_predictions, prediction_variances, 
         oob_counts, sufficient_oob_mask) = self._calculate_oob_predictions(X, y, fit_params)
        
        # Calculate calibration weights
        calibration_weights = self._calculate_calibration_weights(
            prediction_variances, oob_counts
        )
        
        # Get true labels for samples with sufficient OOB predictions
        true_labels = y[sufficient_oob_mask]
        
        # Fit calibrator
        if self.ctype == "logistic":
            self.calibrator_.fit(
                mean_predictions.reshape(-1, 1), 
                true_labels, 
                sample_weight=calibration_weights
            )
        elif self.ctype == "isotonic":
            self.calibrator_.fit(
                mean_predictions, 
                true_labels, 
                sample_weight=calibration_weights
            )
        
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
                f"X has {X.shape[1]} features, but CalibratedTree was fitted with "
                f"{self.n_features_in_} features"
            )
        
        n_samples = X.shape[0]
        
        # Get ensemble predictions
        ensemble_predictions = np.zeros(n_samples, dtype=np.float64)
        
        for estimator in self.estimators_:
            predictions = estimator.predict_proba(X)[:, 1]
            ensemble_predictions += predictions
        
        # Average ensemble predictions
        ensemble_predictions /= len(self.estimators_)
        
        # Apply calibration
        if self.ctype == "logistic":
            calibrated_probabilities = self.calibrator_.predict_proba(
                ensemble_predictions.reshape(-1, 1)
            )[:, 1]
        elif self.ctype == "isotonic":
            calibrated_probabilities = self.calibrator_.predict(ensemble_predictions)
        else:
            raise ValueError(f"Unknown calibration type: {self.ctype}")
        
        # Ensure probabilities are in valid range
        calibrated_probabilities = np.clip(calibrated_probabilities, 0.0, 1.0)
        
        # Create probability matrix
        probability_matrix = np.zeros((n_samples, 2), dtype=np.float64)
        probability_matrix[:, 1] = calibrated_probabilities
        probability_matrix[:, 0] = 1.0 - calibrated_probabilities
        
        return probability_matrix
    
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
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'ctype': self.ctype,
            'alpha0': self.alpha0,
            'beta0': self.beta0,
            'verbose': self.verbose,
            'random_state': self.random_state,
            **self.tree_kwargs
        }
    
    def set_params(self, **params) -> 'CalibratedTree':
        """Set the parameters of this estimator.
        
        This method handles both direct parameters of CalibratedTree and
        parameters that should be passed to the underlying DecisionTreeClassifier
        via tree_kwargs.
        
        Parameters
        ----------
        **params : dict
            Parameter names and their new values.
            
        Returns
        -------
        CalibratedTree
            The estimator instance.
            
        Raises
        ------
        ValueError
            If an invalid parameter is provided.
            
        Notes
        -----
        Parameters are categorized as follows:
        - Direct CalibratedTree parameters: n_estimators, criterion, max_depth,
          min_samples_split, min_samples_leaf, ctype, alpha0, beta0, verbose,
          random_state
        - DecisionTreeClassifier parameters: All other valid sklearn
          DecisionTreeClassifier parameters are stored in tree_kwargs
        """
        # Define the direct parameters of CalibratedTree
        direct_params = {
            'n_estimators', 'criterion', 'max_depth', 'min_samples_split',
            'min_samples_leaf', 'ctype', 'alpha0', 'beta0', 'verbose',
            'random_state', 'tree_kwargs'
        }
        
        # Get valid DecisionTreeClassifier parameters from a dummy instance
        # This ensures we only accept valid sklearn parameters
        from sklearn.tree import DecisionTreeClassifier
        dummy_tree = DecisionTreeClassifier()
        valid_tree_params = set(dummy_tree.get_params(deep=False).keys())
        
        for key, value in params.items():
            if key in direct_params:
                # Set direct CalibratedTree parameters
                setattr(self, key, value)
            elif key in valid_tree_params:
                # Set DecisionTreeClassifier parameters in tree_kwargs
                if not hasattr(self, 'tree_kwargs') or self.tree_kwargs is None:
                    self.tree_kwargs = {}
                self.tree_kwargs[key] = value
            else:
                # Invalid parameter
                raise ValueError(
                    f"Invalid parameter '{key}' for estimator CalibratedTree. "
                    f"Valid parameters are: {sorted(direct_params | valid_tree_params)}"
                )
        
        return self
    
    def __repr__(self) -> str:
        """Return string representation of the estimator."""
        params = self.get_params()
        param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
        return f"CalibratedTree({param_str})"
    
    @property
    def feature_importances_(self) -> NDArray:
        """Return the feature importances averaged over all estimators.
        
        Returns
        -------
        NDArray of shape (n_features,)
            The feature importances. Higher values indicate more important features.
            
        Raises
        ------
        AttributeError
            If the model has not been fitted yet.
        """
        check_is_fitted(self, 'is_fitted_')
        
        if not self.estimators_:
            raise AttributeError("Feature importances are not available before fitting")
        
        # Average feature importances across all estimators
        importances = np.zeros(self.n_features_in_, dtype=np.float64)
        
        for estimator in self.estimators_:
            importances += estimator.feature_importances_
        
        return importances / len(self.estimators_)
    
    @property 
    def n_outputs_(self) -> int:
        """Number of outputs when fit is called.
        
        Returns
        -------
        int
            Number of outputs (always 1 for binary classification).
        """
        return 1
    
    def decision_function(self, X: NDArray) -> NDArray:
        """Compute the decision function of X.
        
        Parameters
        ----------
        X : NDArray of shape (n_samples, n_features)
            Input features.
            
        Returns
        -------
        NDArray of shape (n_samples,)
            The decision function of the input samples. The order of the
            classes corresponds to that in the attribute classes_.
        """
        # Get probability of positive class and convert to decision scores
        probabilities = self.predict_proba(X)
        # Convert probabilities to decision scores (log-odds)
        positive_proba = probabilities[:, 1]
        # Avoid log(0) by clipping
        positive_proba = np.clip(positive_proba, 1e-15, 1 - 1e-15)
        return np.log(positive_proba / (1 - positive_proba))
    
    def apply(self, X: NDArray) -> NDArray:
        """Apply trees in the ensemble to X, return leaf indices.
        
        Parameters
        ----------
        X : NDArray of shape (n_samples, n_features)
            Input features.
            
        Returns
        -------
        NDArray of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in.
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=False)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but CalibratedTree was fitted with "
                f"{self.n_features_in_} features"
            )
        
        # Apply each estimator to get leaf indices
        leaf_indices = np.zeros((X.shape[0], len(self.estimators_)), dtype=np.int64)
        
        for idx, estimator in enumerate(self.estimators_):
            leaf_indices[:, idx] = estimator.apply(X)
        
        return leaf_indices
    
    def decision_path(self, X: NDArray) -> List:
        """Return the decision path in the ensemble.
        
        Parameters
        ----------
        X : NDArray of shape (n_samples, n_features)
            Input features.
            
        Returns
        -------
        List
            List of sparse matrices indicating the decision paths for each estimator.
            Each matrix has shape (n_samples, n_nodes) where entry (i, j) is 1 
            if sample i goes through node j.
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=False)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but CalibratedTree was fitted with "
                f"{self.n_features_in_} features"
            )
        
        # Get decision paths for each estimator
        decision_paths = []
        
        for idx, estimator in enumerate(self.estimators_):
            indicator = estimator.decision_path(X)
            decision_paths.append(indicator)
        
        return decision_paths
    