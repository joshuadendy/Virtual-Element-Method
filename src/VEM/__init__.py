from .assembly import (
    apply_dirichlet,
    assemble_l2_projection,
    assemble_poisson,
    solve_heat,
    unit_square_boundary_dofs,
)
from .diagnostics import (
    compare_gradient_projectors,
    compare_projectors,
    error,
    estimate_eoc,
    mesh_size,
    plot_eoc_curves,
    projected_error,
)
from .spaces import FEMSpace, VEMSpace

__all__ = [
    "FEMSpace",
    "VEMSpace",
    "apply_dirichlet",
    "assemble_l2_projection",
    "assemble_poisson",
    "solve_heat",
    "unit_square_boundary_dofs",
    "compare_gradient_projectors",
    "compare_projectors",
    "error",
    "estimate_eoc",
    "mesh_size",
    "plot_eoc_curves",
    "projected_error",
]
