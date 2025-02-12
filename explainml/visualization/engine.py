import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional
import numpy as np

class VisualizationEngine:
    """Engine for creating various explanation visualizations."""
    
    def __init__(self, style: str = 'default'):
        self.style = style
        self._set_style()
        
    def _set_style(self) -> None:
        """Set the visualization style."""
        if self.style == 'default':
            plt.style.use('seaborn')
        elif self.style == 'dark':
            plt.style.use('dark_background')
        
    def plot_feature_importance(
        self,
        importance_dict: Dict[str, float],
        title: str = 'Feature Importance',
        top_n: Optional[int] = None
    ) -> None:
        """Plot feature importance bars."""
        # Sort features by importance
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        if top_n:
            sorted_features = sorted_features[:top_n]
            
        features, importance = zip(*sorted_features)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(importance), y=list(features))
        plt.title(title)
        plt.xlabel('Importance')
        plt.tight_layout()
        
    def plot_shap_summary(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        max_display: int = 20
    ) -> None:
        """Plot SHAP summary plot."""
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        
    def create_explanation_dashboard(
        self,
        explanation: Dict[str, Any],
        output_path: str
    ) -> None:
        """Create an interactive HTML dashboard."""
        import dash
        from dash import dcc, html
        
        app = dash.Dash(__name__)
        app.layout = html.Div([
            html.H1('Model Explanation Dashboard'),
            dcc.Graph(
                id='feature-importance',
                figure={
                    'data': [{
                        'x': list(explanation['feature_importance'].keys()),
                        'y': list(explanation['feature_importance'].values()),
                        'type': 'bar'
                    }],
                    'layout': {'title': 'Feature Importance'}
                }
            )
        ])
        
        app.run_server(debug=False)