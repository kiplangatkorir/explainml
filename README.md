## This Repository has to be moved to a new library; CyberThreat-ML is a Python library for cybersecurity that uses machine learning to detect threats in computer networks. It uses explainable AI methods to  show you what's happening with easy-to-understand charts and explanations so you can see why something was flagged as dangerous. But for now the explainml library is archived till further notice, Stay tuned for the paper

# ExplainML

ExplainML is a comprehensive Python library for interpreting and explaining machine learning models. It provides a unified interface for various explanation methods, visualization tools, and robustness analysis.

## Features

- **Multiple Explanation Methods**
  - SHAP (SHapley Additive exPlanations)
  - LIME (Local Interpretable Model-agnostic Explanations)
  - Integrated Gradients
  - Counterfactual Explanations
  - Anchor Explanations

- **Framework Support**
  - PyTorch
  - TensorFlow
  - scikit-learn
  - XGBoost

- **Advanced Visualization**
  - Feature importance plots
  - SHAP summary plots
  - Interactive dashboards
  - Customizable styling options

- **Robustness Analysis**
  - Explanation stability metrics
  - Cross-method consistency checking
  - Confidence intervals
  - Sensitivity analysis

## Installation

```bash
# Basic installation
pip install explainml

# With all optional dependencies
pip install explainml[all]

# For development
pip install explainml[dev]
```

## Quick Start

```python
from explainml.methods import ShapExplainer
from explainml.visualization import VisualizationEngine

# Initialize explainer
explainer = ShapExplainer(model=your_model)

# Fit explainer with background data
explainer.fit(X_train)

# Generate explanations
explanation = explainer.explain(X_test)

# Visualize results
viz = VisualizationEngine()
viz.plot_feature_importance(explanation['feature_importance'])
```

## Examples

### Basic Model Explanation
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from explainml.methods import LimeExplainer

# Create and train a model
X = np.random.rand(100, 4)
y = (X[:, 0] + X[:, 1] > 1).astype(int)
model = RandomForestClassifier().fit(X, y)

# Initialize explainer
explainer = LimeExplainer(
    model=model,
    feature_names=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
)

# Generate explanation for a single instance
explanation = explainer.explain(X[0:1])
```

### Deep Learning Model Interpretation
```python
import torch
from explainml.methods import IntegratedGradientsExplainer

# Assuming you have a PyTorch model
explainer = IntegratedGradientsExplainer(
    model=your_pytorch_model,
    framework='pytorch'
)

# Generate and visualize explanations
explanation = explainer.explain(input_tensor)
viz = VisualizationEngine(style='dark')
viz.plot_attribution_map(explanation['attributions'])
```

## Advanced Usage

### Custom Visualization Dashboard
```python
from explainml.visualization import create_explanation_dashboard

# Generate comprehensive explanation
explanation = explainer.explain(X_test)

# Create interactive dashboard
create_explanation_dashboard(
    explanation,
    output_path='explanation_dashboard.html'
)
```

### Robustness Analysis
```python
from explainml.robustness import RobustnessAnalyzer

analyzer = RobustnessAnalyzer(explainer)

# Analyze explanation stability
stability_score = analyzer.explanation_stability(
    X_test,
    n_samples=100,
    perturbation_std=0.1
)
```

## Documentation

Comprehensive documentation is available at [https://explainml.readthedocs.io/](https://explainml.readthedocs.io/)

### Tutorials
- [Quick Start Guide](https://explainml.readthedocs.io/en/latest/tutorials/quickstart.html)
- [Advanced Usage](https://explainml.readthedocs.io/en/latest/tutorials/advanced.html)
- [Custom Explanations](https://explainml.readthedocs.io/en/latest/tutorials/custom.html)

### API Reference
- [Core API](https://explainml.readthedocs.io/en/latest/api/core.html)
- [Explanation Methods](https://explainml.readthedocs.io/en/latest/api/methods.html)
- [Visualization](https://explainml.readthedocs.io/en/latest/api/visualization.html)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/explainml.git
cd explainml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
flake8 explainml
mypy explainml
```

## Citation

If you use ExplainML in your research, please cite:

```bibtex
@software{explainml2024,
  title = {ExplainML: A Comprehensive Library for Machine Learning Interpretability},
  author = {Kiplangat Korir},
  year = {2024},
  url = {https://github.com/kiplangatkorir/explainml}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
