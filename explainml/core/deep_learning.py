from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import torch
import tensorflow as tf
from .base import BaseExplainer
from ..methods.integrated_gradients import IntegratedGradients

class DeepLearningExplainer(BaseExplainer):
    """
    Explainer for deep learning models (PyTorch and TensorFlow).
    Provides various explanation methods specifically designed for neural networks.
    """
    
    def __init__(
        self,
        model: Union[torch.nn.Module, tf.keras.Model],
        framework: Optional[str] = None,
        device: Optional[str] = None,
        layer_names: Optional[List[str]] = None
    ):
        """
        Initialize the deep learning explainer.

        Args:
            model: PyTorch or TensorFlow model
            framework: 'pytorch' or 'tensorflow' (auto-detected if None)
            device: Device to run computations on ('cpu' or 'cuda' for PyTorch)
            layer_names: Names of layers to analyze for attribution
        """
        super().__init__(model, framework)
        self.device = device or self._detect_device()
        self.layer_names = layer_names
        self.hooks = {}
        self.activations = {}
        self._setup_hooks()

    def _detect_device(self) -> str:
        """Detect available compute device."""
        if self.framework == 'pytorch':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return 'cpu'

    def _setup_hooks(self) -> None:
        """Set up hooks for accessing intermediate layer activations."""
        if self.framework == 'pytorch':
            self._setup_pytorch_hooks()
        elif self.framework == 'tensorflow':
            self._setup_tensorflow_hooks()

    def _setup_pytorch_hooks(self) -> None:
        """Set up PyTorch forward hooks."""
        def hook_fn(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook

        if self.layer_names:
            for name, module in self.model.named_modules():
                if name in self.layer_names:
                    self.hooks[name] = module.register_forward_hook(hook_fn(name))

    def _setup_tensorflow_hooks(self) -> None:
        """Set up TensorFlow layer tracking."""
        if self.layer_names:
            self.intermediate_models = {}
            for layer_name in self.layer_names:
                layer = self.model.get_layer(layer_name)
                self.intermediate_models[layer_name] = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=layer.output
                )

    def explain(
        self,
        X: Union[np.ndarray, torch.Tensor, tf.Tensor],
        method: str = "integrated_gradients",
        target_class: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate explanations for the input.

        Args:
            X: Input data
            method: Explanation method to use
            target_class: Target class for attribution (for classification)
            **kwargs: Additional method-specific parameters

        Returns:
            Dictionary containing explanations and metadata
        """
        X = self._prepare_input(X)
        
        methods = {
            "integrated_gradients": self._explain_integrated_gradients,
            "gradcam": self._explain_gradcam,
            "guided_backprop": self._explain_guided_backprop,
            "occlusion": self._explain_occlusion
        }
        
        if method not in methods:
            raise ValueError(f"Unknown explanation method: {method}")
            
        return methods[method](X, target_class, **kwargs)

    def _prepare_input(
        self,
        X: Union[np.ndarray, torch.Tensor, tf.Tensor]
    ) -> Union[torch.Tensor, tf.Tensor]:
        """Prepare input data for the appropriate framework."""
        if self.framework == 'pytorch':
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X)
            if not isinstance(X, torch.Tensor):
                raise ValueError("Input must be numpy array or PyTorch tensor")
            return X.to(self.device)
        else:  # TensorFlow
            if isinstance(X, np.ndarray):
                X = tf.convert_to_tensor(X)
            if not isinstance(X, tf.Tensor):
                raise ValueError("Input must be numpy array or TensorFlow tensor")
            return X

    def _explain_integrated_gradients(
        self,
        X: Union[torch.Tensor, tf.Tensor],
        target_class: Optional[int] = None,
        steps: int = 50,
        baseline: Optional[Union[torch.Tensor, tf.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Compute Integrated Gradients attribution.
        
        Args:
            X: Input data
            target_class: Target class for attribution
            steps: Number of steps for path integral
            baseline: Baseline input (zeros if None)
        """
        ig = IntegratedGradients(self.model, self.framework)
        attributions = ig.attribute(
            X,
            target_class=target_class,
            steps=steps,
            baseline=baseline
        )
        
        return {
            'method': 'integrated_gradients',
            'attributions': attributions,
            'baseline': baseline,
            'steps': steps
        }

    def _explain_gradcam(
        self,
        X: Union[torch.Tensor, tf.Tensor],
        target_class: Optional[int] = None,
        layer_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute Grad-CAM attribution.
        
        Args:
            X: Input data
            target_class: Target class for attribution
            layer_name: Name of layer to compute Grad-CAM for
        """
        if self.framework == 'pytorch':
            return self._gradcam_pytorch(X, target_class, layer_name)
        return self._gradcam_tensorflow(X, target_class, layer_name)

    def _explain_guided_backprop(
        self,
        X: Union[torch.Tensor, tf.Tensor],
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute Guided Backpropagation attribution.
        
        Args:
            X: Input data
            target_class: Target class for attribution
        """
        if self.framework == 'pytorch':
            return self._guided_backprop_pytorch(X, target_class)
        return self._guided_backprop_tensorflow(X, target_class)

    def _explain_occlusion(
        self,
        X: Union[torch.Tensor, tf.Tensor],
        target_class: Optional[int] = None,
        window_size: Tuple[int, ...] = (3, 3),
        stride: int = 1
    ) -> Dict[str, Any]:
        """
        Compute Occlusion attribution.
        
        Args:
            X: Input data
            target_class: Target class for attribution
            window_size: Size of occlusion window
            stride: Stride for sliding window
        """
        if self.framework == 'pytorch':
            return self._occlusion_pytorch(X, target_class, window_size, stride)
        return self._occlusion_tensorflow(X, target_class, window_size, stride)

    def analyze_layer(
        self,
        X: Union[torch.Tensor, tf.Tensor],
        layer_name: str,
        neuron_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze specific layer or neuron activation.
        
        Args:
            X: Input data
            layer_name: Name of layer to analyze
            neuron_index: Specific neuron to analyze (optional)
        """
        X = self._prepare_input(X)
        
        if self.framework == 'pytorch':
            with torch.no_grad():
                _ = self.model(X)
            activation = self.activations[layer_name]
        else:
            activation = self.intermediate_models[layer_name](X)

        analysis = {
            'layer_name': layer_name,
            'activation_mean': float(activation.mean()),
            'activation_std': float(activation.std()),
        }

        if neuron_index is not None:
            analysis['neuron_activation'] = activation[..., neuron_index]

        return analysis

    def get_layer_importance(
        self,
        X: Union[torch.Tensor, tf.Tensor],
        method: str = "activation"
    ) -> Dict[str, float]:
        """
        Compute importance scores for each layer.
        
        Args:
            X: Input data
            method: Method to compute importance ('activation' or 'gradient')
        """
        X = self._prepare_input(X)
        importance_scores = {}

        if method == "activation":
            if self.framework == 'pytorch':
                with torch.no_grad():
                    _ = self.model(X)
                for name, activation in self.activations.items():
                    importance_scores[name] = float(activation.abs().mean())
            else:
                for name, model in self.intermediate_models.items():
                    activation = model(X)
                    importance_scores[name] = float(tf.abs(activation).mean())

        elif method == "gradient":
            if self.framework == 'pytorch':
                for name in self.layer_names:
                    grad = self._compute_layer_gradient_pytorch(X, name)
                    importance_scores[name] = float(grad.abs().mean())
            else:
                for name in self.layer_names:
                    grad = self._compute_layer_gradient_tensorflow(X, name)
                    importance_scores[name] = float(tf.abs(grad).mean())

        return importance_scores

    def cleanup(self) -> None:
        """Clean up hooks and intermediate models."""
        if self.framework == 'pytorch':
            for hook in self.hooks.values():
                hook.remove()
        self.hooks.clear()
        self.activations.clear()
        if hasattr(self, 'intermediate_models'):
            self.intermediate_models.clear()

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()