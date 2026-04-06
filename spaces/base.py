"""Abstract interface for local finite/virtual element spaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


def coerce_triangle_vertices(element_or_vertices: Any) -> Any:
    """
    Accept either an element object or a raw vertex array and return vertices.

    This keeps the transition easy while some callers still pass vertices
    directly and others pass full element objects.
    """
    if hasattr(element_or_vertices, "geometry"):
        return element_or_vertices.geometry.corners
    return element_or_vertices


class SpaceBase(ABC):
    """
    Common interface for all local spaces used in the L2 projection code.
    """

    def __init__(self, view, mapper, order: int, name: str | None = None) -> None:
        self.view = view
        self.mapper = mapper
        self.order = order
        self.name = name or self.__class__.__name__

    @property
    @abstractmethod
    def local_dofs(self) -> int:
        """Number of local degrees of freedom on one element."""
        raise NotImplementedError

    @abstractmethod
    def bind(self, element_or_vertices) -> None:
        """
        Bind element-local state.

        Implementations may accept an element directly or coerce to vertices
        during the transition period.
        """
        raise NotImplementedError

    @abstractmethod
    def local_basis(self, xhat):
        """
        Evaluate the local basis (or projector-reconstructed basis) at xhat.
        """
        raise NotImplementedError

    @abstractmethod
    def interpolate(self, gf):
        """
        Interpolate / project a grid function into the global dof vector
        associated with this space.
        """
        raise NotImplementedError