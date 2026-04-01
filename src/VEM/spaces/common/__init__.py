"""Shared helper utilities for geometry, monomials, projectors, and mappings."""

from .triangle_geometry import (
    coerce_triangle_vertices,
    triangle_area,
    triangle_barycentre,
    triangle_diameter,
    bind_affine_triangle,
)
from .scaled_monomials import (
    P1_EXPONENTS,
    P3_EXPONENTS,
    monomials,
    monomial_gradients,
    scaled_coords,
    scaled_monomials,
    scaled_monomial_gradients,
    monomial_linear_transform_matrix,
    scaled_monomial_inverse_pullback_matrix,
)
from .cls_projector import solve_cls_kkt_all_rhs
from .hermite_mapping import (
    build_cubic_hermite_transform,
    build_k3_mapped_transform,
)
from .vertex_scaling import build_vertex_effective_h