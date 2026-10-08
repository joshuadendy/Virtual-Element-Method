"""Fixed-order Lagrange-type virtual element spaces (Section 5.1)."""

from .mapped import MappedVEMSpace
from .physical import PhysicalVEMSpace


class LinearLagrangePhysicalVEMSpace(PhysicalVEMSpace):
    def __init__(self, view):
        super().__init__(view, order=1)


class LinearLagrangeMappedVEMSpace(MappedVEMSpace):
    def __init__(self, view):
        super().__init__(view, order=1)


class QuadraticLagrangePhysicalVEMSpace(PhysicalVEMSpace):
    def __init__(self, view):
        super().__init__(view, order=2)


class QuadraticLagrangeMappedVEMSpace(MappedVEMSpace):
    def __init__(self, view):
        super().__init__(view, order=2)
