# =============================================================================
# src/__init__.py
# =============================================================================
"""
iFood 2019 AlexNet Project

Source package containing:
- models: AlexNet variants
- data_loader: Dataset handling
- train: Training pipeline
- evaluate: Evaluation metrics
- analyze: Result analysis
"""

__version__ = "1.0.0"
__author__ = "deftorch"
__email__ = "qaidhaidaradila@gmail.com"


# =============================================================================
# src/models/__init__.py
# =============================================================================
"""
AlexNet Model Variants

Available models:
- alexnet_baseline: Original AlexNet
- alexnet_mod1: With Batch Normalization
- alexnet_mod2: With Dropout Tuning
- alexnet_combined: All modifications
"""

from .alexnet import (
    AlexNetBaseline,
    AlexNetMod1,
    AlexNetMod2,
    AlexNetCombined,
    get_model,
    count_parameters,
    model_summary
)

__all__ = [
    'AlexNetBaseline',
    'AlexNetMod1',
    'AlexNetMod2',
    'AlexNetCombined',
    'get_model',
    'count_parameters',
    'model_summary'
]
