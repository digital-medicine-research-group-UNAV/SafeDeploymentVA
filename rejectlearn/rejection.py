from argparse import RawDescriptionHelpFormatter
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import check_cv
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from venn_abers import VennAbers
from numpy.typing import ArrayLike, NDArray
from typing import Any, Dict, Optional, Tuple, Union




class RejectionClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifiers with reject option.

    Parameters
    ----------

    estimator : Optional[BaseEstimator]
        Any scikit-learn compatible estimator.

    mechanism : Optional[str]
        Abstention mechanism to use for constructing the rejection classifier.
        Choose from:

        - 'naive': The rejection mechanism is based on the posterior scores of the classifier.
        - 'sigmoid: The rejection mechanism is based on the Platt-scaled posterior scores.
        - 'isotonic': The rejection mechanism is based on the isotonic regression of the posterior scores.
        - 'conformal': The rejection mechanism is based on the confidence scores of an inductive conformal classifier.
        - 'venn_abers': The rejection mechanism is based on the calibrated probability of a Venn-Abers predictor.

        By default, 'naive' is used.

    cv: Optional[Union[str, int]]
        Validation strategy for calibrating the underlying classifier when needed.
        Choose from:

        - 'prefit': The underlying classifier is assumed to be fitted.
        - int: Number of folds for cross-validation. It must be between 2 and 10.
        
        By default, 5-fold cross-validation is used.

    n_jobs: Optional[int]
        Number of jobs for parallel processing using joblib
        via the "locky" backend.
        At this moment, parallel processing is disabled.
        If ``-1`` all CPUs are used.
        If ``1`` is given, no parallel computing code is used at all,
        which is useful for debugging.
        For n_jobs below ``-1``, ``(n_cpus + 1 + n_jobs)`` are used.
        None is a marker for `unset` that will be interpreted as ``n_jobs=1``
        (sequential execution).

        By default ``None``.
    
    """

    valid_mechanisms = ['naive', 'sigmoid', 'isotonic', 'conformal', 'regva', 'avgva']
    valid_cv = ['prefit'] + list(range(2, 11))

    def __init__(
            self,
            estimator: Optional[BaseEstimator] = None,
            mechanism: Optional[str] = 'naive',
            cv: Optional[Union[str, int]] = 5,
            n_jobs: Optional[int] = 1,
        ) -> None:
            self.estimator = estimator
            self.mechanism = mechanism
            self.cv = cv
            self.n_jobs = n_jobs

    def _check_mechanism(self) -> None:
        if self.mechanism not in self.valid_mechanisms:
            raise ValueError(f"Invalid mechanism: {self.mechanism}. Choose from {self.valid_mechanisms}.")
    
    def _check_cv(self) -> None:
        if self.cv not in self.valid_cv:
            raise ValueError(f"Invalid cv: {self.cv}. Choose from {self.valid_cv}.")
        
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        mondrian: Optional[bool] = False,
        **fit_params,
    ) -> 'RejectionClassifier':
        """
        Fit the rejection classifier. It peforms a calibration step if needed.
        """

        if self.estimator is None:
            raise ValueError("An estimator must be set before fitting the classifier.")

        self._check_mechanism()

        # No calibration is required for the naive mechanism and cv is ignored.
        if self.mechanism == 'naive':
            self.estimator.fit(X, y, **fit_params)
        
        # For the 'sigmoid' and 'isotonic' mechanisms, the underlying classifier is calibrated using CalibratedClassifierCV.
        elif self.mechanism in ['sigmoid', 'isotonic']:
            self.estimator = CalibratedClassifierCV(self.estimator, method=self.mechanism, cv=self.cv)
            self.estimator.fit(X, y, **fit_params)

        elif self.mechanism in ["regva", "avgva"]:
            check_is_fitted(self.estimator)
            scores = self.estimator.predict_proba(X)
            self.ivap = VennAbers()
            self.ivap.fit(scores, y, precision=4)

    def predict(
        self,
        X: ArrayLike,
        calibrated: Optional[bool] = True,
    ) -> NDArray:
        """
        Compute point predictions for the complete set of samples using the underlying predictor.
        """
        if self.mechanism in ['naive', 'sigmoid', 'isotonic']:
            return self.estimator.predict(X)
        
        if self.mechanism in ["regva", "avgva"]:
            return np.round(self.predict_proba(X)[0][:, 1])

    def predict_proba(
        self,
        X: ArrayLike,
    ) -> NDArray:
        """
        Compute scores for the complete set of samples using the underlying predictor.
        """
        if self.mechanism not in ["regva", "avgva"]:
            return self.estimator.predict_proba(X)
        elif self.mechanism == "regva":
            scores = self.estimator.predict_proba(X)
            _, p0p1 = self.ivap.predict_proba(scores)
            low, high = p0p1[:, 0], p0p1[:, 1]
            tmp = high / (1 - low + high)
            va_proba = np.zeros(scores.shape)
            va_proba[:, 0] = 1 - tmp
            va_proba[:, 1] = tmp
            
            # If tmp < 0.5, inerchange low and high
            mask = tmp < 0.5
            temp = low[mask].copy()
            low[mask] = 1 - high[mask]
            high[mask] = 1 - temp
            
            return np.asarray(va_proba), low, high
        elif self.mechanism == "avgva":
            scores = self.estimator.predict_proba(X)
            _, p0p1 = self.ivap.predict_proba(scores)
            low, high = p0p1[:, 0], p0p1[:, 1]
            tmp = (low + high) / 2
            va_proba = np.zeros(scores.shape)
            va_proba[:, 0] = 1 - tmp
            va_proba[:, 1] = tmp

            # If tmp < 0.5, inerchange low and high
            mask = tmp < 0.5
            temp = low[mask].copy()
            low[mask] = 1 - high[mask]
            high[mask] = 1 - temp

            return np.asarray(va_proba), low, high

    def _get_probability_outputs(
        self,
        X: ArrayLike,
    ) -> Tuple[NDArray, Optional[NDArray], Optional[NDArray]]:
        if self.mechanism in ["regva", "avgva"]:
            y_probs, lower_bounds, upper_bounds = self.predict_proba(X)
            return np.asarray(y_probs), np.asarray(lower_bounds), np.asarray(upper_bounds)

        return np.asarray(self.predict_proba(X)), None, None

    def _get_accept_mask(
        self,
        max_probs: NDArray,
        rejection_rate: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[NDArray, float]:
        if rejection_rate is not None and threshold is not None:
            raise ValueError("Only one of rejection_rate or threshold can be provided.")

        if rejection_rate is None and threshold is None:
            raise ValueError("Either rejection_rate or threshold must be provided.")

        if threshold is None:
            if not 0 <= rejection_rate <= 1:
                raise ValueError("rejection_rate must be between 0 and 1.")
            threshold = float(np.quantile(max_probs, rejection_rate))
        else:
            threshold = float(threshold)

        accepted_mask = max_probs >= threshold
        return accepted_mask, threshold

    def selective_predict(
        self,
        X: ArrayLike,
        rejection_rate: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Perform selective classification and return the selection details.
        """
        y_probs, lower_bounds, upper_bounds = self._get_probability_outputs(X)
        max_probs = np.max(y_probs, axis=1)
        accepted_mask, threshold = self._get_accept_mask(
            max_probs,
            rejection_rate=rejection_rate,
            threshold=threshold,
        )

        classes = getattr(self.estimator, "classes_", np.arange(y_probs.shape[1]))
        y_preds = np.asarray(classes)[np.argmax(y_probs, axis=1)]

        result = {
            "predictions": y_preds,
            "probabilities": y_probs,
            "confidence": max_probs,
            "accepted_mask": accepted_mask,
            "threshold": threshold,
            "coverage": float(np.mean(accepted_mask)),
            "rejection_rate": float(1.0 - np.mean(accepted_mask)),
            "n_accepted": int(np.sum(accepted_mask)),
            "n_rejected": int(len(accepted_mask) - np.sum(accepted_mask)),
        }

        if lower_bounds is not None and upper_bounds is not None:
            result["lower_bounds"] = lower_bounds
            result["upper_bounds"] = upper_bounds

        return result

    def estimate_selective_performance(
        self,
        X: ArrayLike,
        rejection_rate: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Estimate selective performance from confidence scores.
        """
        selection = self.selective_predict(
            X,
            rejection_rate=rejection_rate,
            threshold=threshold,
        )

        accepted_mask = selection["accepted_mask"]
        if not np.any(accepted_mask):
            raise ValueError("No samples were accepted.")

        result = {
            "estimate": float(np.mean(selection["confidence"][accepted_mask])),
            "threshold": float(selection["threshold"]),
            "coverage": float(selection["coverage"]),
            "rejection_rate": float(selection["rejection_rate"]),
            "n_accepted": int(selection["n_accepted"]),
            "n_rejected": int(selection["n_rejected"]),
        }

        if "lower_bounds" in selection and "upper_bounds" in selection:
            result["lower_bound"] = float(np.mean(selection["lower_bounds"][accepted_mask]))
            result["upper_bound"] = float(np.mean(selection["upper_bounds"][accepted_mask]))

        return result

    def evaluate_selective_performance(
        self,
        X: ArrayLike,
        y: ArrayLike,
        rejection_rate: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Evaluate selective performance against true labels.
        """
        y = np.asarray(y)
        selection = self.selective_predict(
            X,
            rejection_rate=rejection_rate,
            threshold=threshold,
        )

        accepted_mask = selection["accepted_mask"]
        if not np.any(accepted_mask):
            raise ValueError("No samples were accepted.")

        estimate = float(np.mean(selection["confidence"][accepted_mask]))
        accuracy = float(
            accuracy_score(y[accepted_mask], selection["predictions"][accepted_mask])
        )

        result = {
            "estimate": estimate,
            "accuracy": accuracy,
            "error": float(np.abs(estimate - accuracy)),
            "threshold": float(selection["threshold"]),
            "coverage": float(selection["coverage"]),
            "rejection_rate": float(selection["rejection_rate"]),
            "n_accepted": int(selection["n_accepted"]),
            "n_rejected": int(selection["n_rejected"]),
        }

        if "lower_bounds" in selection and "upper_bounds" in selection:
            result["lower_bound"] = float(np.mean(selection["lower_bounds"][accepted_mask]))
            result["upper_bound"] = float(np.mean(selection["upper_bounds"][accepted_mask]))

        return result

