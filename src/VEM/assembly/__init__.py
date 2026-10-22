"""Assembly routines."""

from .boundary import unit_square_boundary_dofs
from .heat import solve_heat
from .l2_projection import assemble_l2_projection
from .poisson import apply_dirichlet, assemble_poisson

__all__ = [
    "unit_square_boundary_dofs",
    "solve_heat",
    "assemble_l2_projection",
    "apply_dirichlet",
    "assemble_poisson",
]
