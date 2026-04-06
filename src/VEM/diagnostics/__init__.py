"""Diagnostics and comparison utilities."""
from .compare_projectors import compare_projectors
from .compare_gradients import compare_gradient_projectors
from .error import error, projected_error
from .eoc import mesh_size, estimate_eoc, plot_eoc_curves
