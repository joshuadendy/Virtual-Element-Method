"""Hermite-type virtual element spaces."""

from .hermite_physical import HermitePhysicalVEMSpace
from .hermite_mapped import HermiteMappedVEMSpace

__all__ = [
    "HermitePhysicalVEMSpace",
    "HermiteMappedVEMSpace",
]