"""Fixed-order Hermite-type virtual element spaces (Section 5.2)."""

from .mapped import MappedVEMSpace
from .physical import PhysicalVEMSpace


class CubicHermitePhysicalVEMSpace(PhysicalVEMSpace):
    def __init__(self, view):
        super().__init__(view, order=3, hermite=True)


class CubicHermiteMappedVEMSpace(MappedVEMSpace):
    def __init__(self, view):
        super().__init__(view, order=3, hermite=True)


class QuarticHermitePhysicalVEMSpace(PhysicalVEMSpace):
    def __init__(self, view):
        super().__init__(view, order=4, hermite=True)


class QuarticHermiteMappedVEMSpace(MappedVEMSpace):
    def __init__(self, view):
        super().__init__(view, order=4, hermite=True)
