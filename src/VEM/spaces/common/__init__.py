"""Shared helper utilities for geometry, monomials, projectors, and mappings."""

from .triangle_geometry import (
    EDGES,
    REFERENCE_TRIANGLE_VERTICES,
    coerce_triangle_vertices,
    triangle_area,
    triangle_barycentre,
    triangle_diameter,
    bind_affine_triangle,
)
from .scaled_monomials import (
    scaled_monomials,
    scaled_monomial_gradients,
    total_degree_exponents,
)
from .cls_projector import solve_cls_kkt_all_rhs
from .vertex_scaling import build_vertex_effective_h
