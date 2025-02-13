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
import joblib
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import warnings

class ModelExplainer:
    def __init__(self, model: Any, model_type: str = "classification", feature_names: Optional[List[str]] = None,
                 class_names: Optional[List[str]] = None, background_data: Optional[Union[np.ndarray, pd.DataFrame]] = None):
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.class_names = class_names
        self.background_data = background_data
        self.framework = self._detect_framework()
        self._initialize_explainers()

    def _detect_framework(self) -> str:
        if isinstance(self.model, torch.nn.Module):
            return 'pytorch'
        elif isinstance(self.model, tf.keras.Model):
            return 'tensorflow'
        elif isinstance(self.model, BaseEstimator):
            return 'sklearn'
        else:
            raise ValueError("Unsupported model type. Must be PyTorch, TensorFlow, or scikit-learn model.")

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self.framework == 'pytorch':
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                return self.model(X_tensor).numpy()
        elif self.framework == 'tensorflow':
            return self.model.predict(X)
        elif self.framework == 'sklearn':
            return self.model.predict_proba(X) if self.model_type == 'classification' else self.model.predict(X)

    def _initialize_explainers(self) -> None:
        self.explainers = {}
        if self.background_data is not None:
            if self.framework in ['pytorch', 'tensorflow']:
                self.explainers['shap'] = shap.DeepExplainer(self.model, self.background_data)
            else:
                self.explainers['shap'] = shap.KernelExplainer(self._predict, self.background_data)

        self.explainers['lime'] = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.background_data if self.background_data is not None else np.zeros((1, 1)),
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=self.model_type
        )
        if self.framework in ['pytorch', 'tensorflow']:
            self.explainers['integrated_gradients'] = IntegratedGradients(self.model)

    def explain_prediction(self, X: Union[np.ndarray, pd.DataFrame], method: str = "shap", **kwargs) -> Dict[str, Any]:
        if method not in self.explainers:
            raise ValueError(f"Unknown explanation method: {method}")
        if isinstance(X, pd.DataFrame):
            X = X.values
        explanation = self._generate_explanation(X, method, **kwargs)
        explanation['method'] = method
        explanation['feature_names'] = self.feature_names
        return explanation

    def _generate_explanation(self, X: np.ndarray, method: str, **kwargs) -> Dict[str, Any]:
        if method == 'shap':
            return self._explain_shap(X, **kwargs)
        elif method == 'lime':
            return self._explain_lime(X, **kwargs)
        elif method == 'integrated_gradients':
            return self._explain_integrated_gradients(X, **kwargs)

    def _explain_shap(self, X: np.ndarray, **kwargs) -> Dict[str, Any]:
        shap_values = self.explainers['shap'].shap_values(X)
        base_value = self.explainers['shap'].expected_value
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        return {'shap_values': shap_values, 'base_value': base_value, 'feature_importance': np.abs(shap_values).mean(0)}

    def _explain_lime(self, X: np.ndarray, num_features: int = 10, **kwargs) -> Dict[str, Any]:
        explanations = []
        feature_importance = {}
        
        def explain_instance(i):
            exp = self.explainers['lime'].explain_instance(X[i], self._predict, num_features=num_features, **kwargs)
            return exp
        
        explanations = Parallel(n_jobs=-1)(delayed(explain_instance)(i) for i in range(len(X)))
        
        for exp in explanations:
            for feature, importance in exp.local_exp.items():
                feature_name = self.feature_names[feature] if self.feature_names else str(feature)
                if feature_name not in feature_importance:
                    feature_importance[feature_name] = []
                feature_importance[feature_name].append(abs(importance))
        
        return {'explanations': explanations, 'feature_importance': {k: np.mean(v) for k, v in feature_importance.items()}}

    def plot_shap(self, X: np.ndarray):
        explainer = self.explainers.get('shap')
        if explainer:
            shap_values = explainer.shap_values(X)
            shap.summary_plot(shap_values, features=X, feature_names=self.feature_names)
        else:
            raise ValueError("SHAP explainer not initialized.")

    def plot_lime(self, X: np.ndarray, instance_idx: int = 0):
        explainer = self.explainers.get('lime')
        if explainer:
            exp = explainer.explain_instance(X[instance_idx], self._predict, num_features=10)
            exp.show_in_notebook()
        else:
            raise ValueError("LIME explainer not initialized.")

    def save(self, path: str) -> None:
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str) -> 'ModelExplainer':
        return joblib.load(path)
