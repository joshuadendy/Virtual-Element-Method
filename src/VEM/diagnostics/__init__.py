"""Diagnostics and comparison utilities."""

from .compare_gradients import compare_gradient_projectors
from .compare_projectors import compare_projectors
from .eoc import estimate_eoc, mesh_size, plot_eoc_curves
from .error import error, projected_error

__all__ = [
    "compare_gradient_projectors",
    "compare_projectors",
    "estimate_eoc",
    "mesh_size",
    "plot_eoc_curves",
    "error",
    "projected_error",
]
