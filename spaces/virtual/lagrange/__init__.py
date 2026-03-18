"""Lagrange-type virtual element spaces."""

from .lagrange_physical import LagrangePhysicalVEMSpace
from .lagrange_mapped import LagrangeMappedVEMSpace

__all__ = [
    "LagrangePhysicalVEMSpace",
    "LagrangeMappedVEMSpace",
]