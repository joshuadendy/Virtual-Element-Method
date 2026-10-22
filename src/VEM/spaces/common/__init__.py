"""Shared helper utilities for geometry, monomials, projectors, and mappings."""

from .cls_projector import solve_cls
from .scaled_monomials import (
    scaled_monomial_gradients,
    scaled_monomials,
    total_degree_exponents,
)
from .triangle_geometry import (
    EDGES,
    REFERENCE_TRIANGLE_VERTICES,
    bind_affine_triangle,
    coerce_triangle_vertices,
    triangle_area,
    triangle_barycentre,
    triangle_diameter,
)
from .vertex_scaling import build_vertex_effective_h

__all__ = [
    "solve_cls",
    "scaled_monomial_gradients",
    "scaled_monomials",
    "total_degree_exponents",
    "EDGES",
    "REFERENCE_TRIANGLE_VERTICES",
    "bind_affine_triangle",
    "coerce_triangle_vertices",
    "triangle_area",
    "triangle_barycentre",
    "triangle_diameter",
    "build_vertex_effective_h",
]
