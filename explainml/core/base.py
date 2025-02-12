from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd

class BaseExplainer(ABC):
    """
    Abstract base class for all explainers.

    Provides a unified interface for model explainability methods, ensuring
    compatibility across different machine learning frameworks.
    """

    SUPPORTED_FRAMEWORKS = ["pytorch", "tensorflow", "sklearn"]

    def __init__(self, model: Any, framework: Optional[str] = None):
        """
        Initialize the explainer with a given model.

        Args:
            model (Any): The machine learning model to be explained.
            framework (Optional[str]): The ML framework ("pytorch", "tensorflow", "sklearn").
                                       If None, the framework is auto-detected.
        """
        self.model = model
        self.framework = framework or self._detect_framework()
        self._validate_model()

    def _detect_framework(self) -> str:
        """
        Detect the ML framework of the model.

        Returns:
            str: The detected framework ("pytorch", "tensorflow", "sklearn").

        Raises:
            ValueError: If the model framework is not supported.
        """
        try:
            import torch
            import tensorflow as tf
            from sklearn.base import BaseEstimator
        except ImportError as e:
            raise ImportError(f"Missing dependencies: {e}")

        if isinstance(self.model, torch.nn.Module):
            return "pytorch"
        elif isinstance(self.model, tf.keras.Model):
            return "tensorflow"
        elif isinstance(self.model, BaseEstimator):
            return "sklearn"
        else:
            raise ValueError(f"Unsupported model type: {type(self.model).__name__}")

    def _validate_model(self) -> None:
        """
        Validate model compatibility with the supported frameworks.

        Raises:
            ValueError: If the model framework is not supported.
            AttributeError: If the model lacks required prediction methods.
        """
        if self.framework not in self.SUPPORTED_FRAMEWORKS:
            raise ValueError(f"Framework '{self.framework}' is not supported.")

        # Ensure the model has a predict method for explainability
        if not hasattr(self.model, "predict"):
            raise AttributeError(f"The provided model does not have a 'predict' method.")

    @abstractmethod
    def explain(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate explanations for the given input data.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Input data to explain.

        Returns:
            Dict[str, Any]: Explanation results (e.g., feature importance, SHAP values).
        """
        pass

    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        """
        Fit the explainer to background data if required.

        This method can be overridden by subclasses if an explainer requires fitting.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Background data for explanation initialization.
        """
        pass
