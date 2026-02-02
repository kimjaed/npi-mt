# src/npi_mt/npi/__init__.py
"""
Neural residual refinement module for the Neuro-Physical Inverter (NPI).

This package provides:
- A ResNet-1D model for learning residual corrections to EnsCGP estimates
- Normalization utilities consistent with synthetic training
- High-level refinement routines for single models or ensembles
- Physics-based loss for fine-tuning on real MT data
"""

# ----------------------
# Models
# ----------------------
from .resnet1d import (
    ResNetResidual1D,
    ResBlock,
    freeze_bn_stats,
)

# ----------------------
# Normalization
# ----------------------
from .normalization import (
    NormalizationStats,
    load_normalization_stats,
)

# ----------------------
# Refinement API
# ----------------------
from .refinement import (
    RefinementOutput,
    refine_log_rho,
    refine_and_forward,
    RefinementLoopConfig,
    fine_tune_on_real_example,
)

# ----------------------
# Physics-constrained loss (optional)
# ----------------------
from .physics_loss import (
    PhysicsLoss,
)

__all__ = [
    # Models
    "ResNetResidual1D",
    "ResBlock",
    "freeze_bn_stats",

    # Normalization
    "NormalizationStats",
    "load_normalization_stats",

    # Refinement
    "RefinementOutput",
    "refine_log_rho",
    "refine_and_forward",
    "RefinementLoopConfig",
    "fine_tune_on_real_example",

    # Physics loss
    "PhysicsLoss",
]
