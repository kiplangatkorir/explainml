import lime
import lime.lime_tabular
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from ..core.base import BaseExplainer

class LimeExplainer(BaseExplainer):
    """
    LIME-based model explainer.

    Uses LIME to generate feature importance explanations for models.
    Supports both classification and regression tasks.
    """

    def __init__(
        self, 
        model: Any, 
        framework: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize the LIME explainer.

        Args:
            model (Any): The machine learning model to explain.
            framework (Optional[str]): The ML framework ("pytorch", "tensorflow", "sklearn").
                                       If None, it is automatically detected.
            feature_names (Optional[List[str]]): List of feature names.
            class_names (Optional[List[str]]): List of class names (for classification).
        """
        super().__init__(model, framework)
        self.feature_names = feature_names
        self.class_names = class_names
        self.explainer = None
        self.is_fitted = False  

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        """
        Fit the LIME explainer using training data.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Training dataset.

        Raises:
            ValueError: If feature names do not match dataset dimensions.
        """
        if self.is_fitted:
            return  
        X_array = self._convert_to_numpy(X)

        if self.feature_names and len(self.feature_names) != X_array.shape[1]:
            raise ValueError("Number of feature names does not match dataset dimensions.")

        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_array,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode="classification" if self.class_names else "regression"
        )

        self.is_fitted = True

    def explain(
        self, 
        X: Union[np.ndarray, pd.DataFrame],
        num_features: int = 10
    ) -> Dict[str, Any]:
        """
        Generate LIME explanations for given input data.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Input dataset to explain.
            num_features (int): Number of top features to include in explanations.

        Returns:
            Dict[str, Any]: Dictionary containing LIME explanations and feature importance.

        Raises:
            ValueError: If `fit()` has not been called before explaining.
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")

        X_array = self._convert_to_numpy(X)

        explanations = []
        for i in range(len(X_array)):
            exp = self.explainer.explain_instance(
                X_array[i],
                self.model.predict_proba if self.class_names else self.model.predict,
                num_features=num_features
            )
            explanations.append(exp)

        return {
            "explanations": explanations,
            "feature_importance": self._aggregate_feature_importance(explanations)
        }

    def _aggregate_feature_importance(
        self, 
        explanations: List[Any]
    ) -> Dict[str, float]:
        """
        Aggregate feature importance across multiple explanations.

        Args:
            explanations (List[Any]): List of LIME explanation objects.

        Returns:
            Dict[str, float]: Dictionary of averaged feature importance scores.
        """
        importance_dict = {}

        for exp in explanations:
            for feature_index, importance in exp.local_exp[1]:  
                feature_name = self.feature_names[feature_index] if self.feature_names else f"Feature {feature_index}"
                if feature_name not in importance_dict:
                    importance_dict[feature_name] = []
                importance_dict[feature_name].append(abs(importance))

        return {feature: np.mean(values) for feature, values in importance_dict.items()}

    def _convert_to_numpy(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Convert input data to a NumPy array if it's a Pandas DataFrame.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Input dataset.

        Returns:
            np.ndarray: NumPy array representation of the input.
        """
        return X.to_numpy() if isinstance(X, pd.DataFrame) else X
