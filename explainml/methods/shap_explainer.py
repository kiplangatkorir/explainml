import shap
from typing import Any, Dict, Union, Optional
import numpy as np
import pandas as pd
from ..core.base import BaseExplainer

class ShapExplainer(BaseExplainer):
    """SHAP-based model explainer."""
    
    def __init__(
        self, 
        model: Any, 
        framework: Optional[str] = None,
        background_data: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ):
        super().__init__(model, framework)
        self.background_data = background_data
        self.explainer = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        """Initialize SHAP explainer with background data."""
        if self.framework == 'pytorch':
            self.explainer = shap.DeepExplainer(self.model, X)
        elif self.framework == 'tensorflow':
            self.explainer = shap.DeepExplainer(self.model, X)
        else:
            self.explainer = shap.KernelExplainer(self.model.predict_proba, X)
            
    def explain(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """Generate SHAP values and explanations."""
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
            
        shap_values = self.explainer.shap_values(X)
        
        return {
            'shap_values': shap_values,
            'base_value': self.explainer.expected_value,
            'feature_importance': np.abs(shap_values).mean(0)
        }
