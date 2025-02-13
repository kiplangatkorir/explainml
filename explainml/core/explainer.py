from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import torch
import tensorflow as tf
from sklearn.base import BaseEstimator
import shap
import lime
import lime.lime_tabular
from captum.attr import IntegratedGradients
import warnings

class ModelExplainer:
    """
    Main explainer class that provides a unified interface for model interpretation.
    
    Attributes:
        model: The machine learning model to explain
        model_type: Type of the model ('classification' or 'regression')
        framework: ML framework used ('pytorch', 'tensorflow', 'sklearn')
        feature_names: Names of input features
        class_names: Names of output classes (for classification)
    """
    
    def __init__(
        self,
        model: Any,
        model_type: str = "classification",
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        background_data: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ):
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.class_names = class_names
        self.background_data = background_data
        self.framework = self._detect_framework()
        self._initialize_explainers()
        
    def _detect_framework(self) -> str:
        """Detect the ML framework of the model."""
        if isinstance(self.model, torch.nn.Module):
            return 'pytorch'
        elif isinstance(self.model, tf.keras.Model):
            return 'tensorflow'
        elif isinstance(self.model, BaseEstimator):
            return 'sklearn'
        else:
            raise ValueError(
                "Unsupported model type. Must be PyTorch, TensorFlow, or scikit-learn model."
            )

    def _initialize_explainers(self) -> None:
        """Initialize various explanation methods."""
        self.explainers = {}
        
        if self.background_data is not None:
            if self.framework == 'pytorch':
                self.explainers['shap'] = shap.DeepExplainer(
                    self.model, 
                    self.background_data
                )
            elif self.framework == 'tensorflow':
                self.explainers['shap'] = shap.DeepExplainer(
                    self.model, 
                    self.background_data
                )
            else:
                self.explainers['shap'] = shap.KernelExplainer(
                    self.model.predict_proba 
                    if self.model_type == 'classification' 
                    else self.model.predict,
                    self.background_data
                )
        
        # Initialize LIME
        self.explainers['lime'] = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.background_data if self.background_data is not None else np.zeros((1, 1)),
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=self.model_type
        )
        
        # Initialize Integrated Gradients for deep learning models
        if self.framework in ['pytorch', 'tensorflow']:
            self.explainers['integrated_gradients'] = IntegratedGradients(self.model)

    def explain_prediction(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        method: str = "shap",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate explanation for model predictions.
        
        Args:
            X: Input data to explain
            method: Explanation method ('shap', 'lime', 'integrated_gradients')
            **kwargs: Additional arguments for specific explanation methods
            
        Returns:
            Dictionary containing explanation results
        """
        if method not in self.explainers:
            raise ValueError(f"Unknown explanation method: {method}")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        explanation = self._generate_explanation(X, method, **kwargs)
        explanation['method'] = method
        explanation['feature_names'] = self.feature_names
        
        return explanation
    
    def _generate_explanation(
        self,
        X: np.ndarray,
        method: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate explanation using specified method."""
        if method == 'shap':
            return self._explain_shap(X, **kwargs)
        elif method == 'lime':
            return self._explain_lime(X, **kwargs)
        elif method == 'integrated_gradients':
            return self._explain_integrated_gradients(X, **kwargs)
            
    def _explain_shap(
        self,
        X: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate SHAP explanations."""
        shap_values = self.explainers['shap'].shap_values(X)
        
        if isinstance(shap_values, list):  # For classification
            shap_values = np.array(shap_values)
            
        return {
            'shap_values': shap_values,
            'base_value': self.explainers['shap'].expected_value,
            'feature_importance': np.abs(shap_values).mean(0)
        }
        
    def _explain_lime(
        self,
        X: np.ndarray,
        num_features: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate LIME explanations."""
        explanations = []
        feature_importance = {}
        
        for i in range(len(X)):
            exp = self.explainers['lime'].explain_instance(
                X[i],
                self.model.predict_proba if self.model_type == 'classification' else self.model.predict,
                num_features=num_features,
                **kwargs
            )
            explanations.append(exp)
            
            # Aggregate feature importance
            for feature, importance in exp.local_exp[1]:
                feature_name = self.feature_names[feature] if self.feature_names else str(feature)
                if feature_name not in feature_importance:
                    feature_importance[feature_name] = []
                feature_importance[feature_name].append(abs(importance))
        
        return {
            'explanations': explanations,
            'feature_importance': {
                feature: np.mean(values)
                for feature, values in feature_importance.items()
            }
        }
        
    def _explain_integrated_gradients(
        self,
        X: np.ndarray,
        n_steps: int = 50,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate Integrated Gradients explanations."""
        if self.framework not in ['pytorch', 'tensorflow']:
            raise ValueError("Integrated Gradients only supported for deep learning models")
            
        # Convert input to appropriate format
        if self.framework == 'pytorch':
            X = torch.tensor(X, dtype=torch.float32)
        else:  # tensorflow
            X = tf.convert_to_tensor(X, dtype=tf.float32)
            
        attributions = self.explainers['integrated_gradients'].attribute(
            X,
            n_steps=n_steps,
            **kwargs
        )
        
        if self.framework == 'pytorch':
            attributions = attributions.detach().numpy()
        else:
            attributions = attributions.numpy()
            
        return {
            'attributions': attributions,
            'feature_importance': np.abs(attributions).mean(0)
        }
        
    def feature_importance(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        method: str = "shap",
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculate global feature importance.
        
        Args:
            X: Input data
            method: Method to compute importance ('shap', 'lime', 'integrated_gradients')
            
        Returns:
            DataFrame with feature importance scores
        """
        explanation = self.explain_prediction(X, method, **kwargs)
        importance_values = explanation['feature_importance']
        
        feature_names = (
            self.feature_names 
            if self.feature_names is not None 
            else [f"Feature {i}" for i in range(len(importance_values))]
        )
        
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
        }).sort_values('Importance', ascending=False)
        
    def partial_dependence(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_idx: int,
        num_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate partial dependence for a specific feature.
        
        Args:
            X: Input data
            feature_idx: Index of the feature to analyze
            num_points: Number of points to evaluate
            
        Returns:
            Tuple of (feature_values, predictions)
        """
        feature_values = np.linspace(
            np.min(X[:, feature_idx]),
            np.max(X[:, feature_idx]),
            num_points
        )
        
        predictions = []
        for value in feature_values:
            X_modified = X.copy()
            X_modified[:, feature_idx] = value
            pred = self.model.predict(X_modified).mean()
            predictions.append(pred)
            
        return feature_values, np.array(predictions)
        
    def interaction_effects(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_indices: Optional[List[Tuple[int, int]]] = None
    ) -> pd.DataFrame:
        """
        Calculate interaction effects between features.
        
        Args:
            X: Input data
            feature_indices: List of feature index pairs to analyze
            
        Returns:
            DataFrame with interaction scores
        """
        if not isinstance(X, np.ndarray):
            X = X.values
            
        if feature_indices is None:
            n_features = X.shape[1]
            feature_indices = [
                (i, j)
                for i in range(n_features)
                for j in range(i + 1, n_features)
            ]
            
        interactions = []
        for i, j in feature_indices:
            score = self._calculate_interaction_score(X, i, j)
            interactions.append({
                'Feature 1': self.feature_names[i] if self.feature_names else f"Feature {i}",
                'Feature 2': self.feature_names[j] if self.feature_names else f"Feature {j}",
                'Interaction Score': score
            })
            
        return pd.DataFrame(interactions).sort_values('Interaction Score', ascending=False)
        
    def _calculate_interaction_score(
        self,
        X: np.ndarray,
        feature1_idx: int,
        feature2_idx: int
    ) -> float:
        """Calculate interaction score between two features."""
        # Simple H-statistic implementation
        shap_values = self.explain_prediction(X, method='shap')['shap_values']
        
        if isinstance(shap_values, list):  # For classification
            shap_values = np.array(shap_values).mean(0)  # Average over classes
            
        interaction = np.abs(
            shap_values[:, feature1_idx] * shap_values[:, feature2_idx]
        ).mean()
        
        return float(interaction)

    def save(self, path: str) -> None:
        """Save explainer to disk."""
        import joblib
        joblib.dump(self, path)
        
    @classmethod
    def load(cls, path: str) -> 'ModelExplainer':
        """Load explainer from disk."""
        import joblib
        return joblib.load(path)