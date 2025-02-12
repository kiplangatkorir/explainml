from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

class BaseExplainer(ABC):
    """Abstract base class for all explainers."""
    
    def __init__(self, model: Any, framework: Optional[str] = None):
        self.model = model
        self.framework = framework or self._detect_framework()
        self._validate_model()
        
    def _detect_framework(self) -> str:
        """Detect the ML framework of the model."""
        import torch
        import tensorflow as tf
        from sklearn.base import BaseEstimator
        
        if isinstance(self.model, torch.nn.Module):
            return 'pytorch'
        elif isinstance(self.model, tf.keras.Model):
            return 'tensorflow'
        elif isinstance(self.model, BaseEstimator):
            return 'sklearn'
        else:
            raise ValueError("Unsupported model type")
    
    def _validate_model(self) -> None:
        """Validate model compatibility."""
        supported_frameworks = ['pytorch', 'tensorflow', 'sklearn']
        if self.framework not in supported_frameworks:
            raise ValueError(f"Framework {self.framework} not supported")
    
    @abstractmethod
    def explain(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """Generate explanation for the input data."""
        pass
    
    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        """Fit the explainer to the background data if needed."""
        pass