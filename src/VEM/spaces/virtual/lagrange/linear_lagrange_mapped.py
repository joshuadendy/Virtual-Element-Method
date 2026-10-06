from .linear_lagrange_physical import LinearLagrangePhysicalVEMSpace


class LinearLagrangeMappedVEMSpace(LinearLagrangePhysicalVEMSpace):
    """
    k=1 scalar H1-conforming VEM on triangles.

    The projectors Pi_0 and Pi_1 are assembled once by a physical space bound to
    the reference triangle. Pi_0 is evaluated on the reference element and Pi_1 is
    mapped to the physical element with the covariant Piola factor J^{-T}.

    For this lowest-order triangular case Pi_1 coincides algebraically with the
    gradient of Pi_0, but we keep a separate projector so that it can be scaled,
    inspected, or replaced independently.
    """

    def __init__(self, view):
        self._ref = LinearLagrangePhysicalVEMSpace(view)
        super().__init__(view)

    def bind(self, element_or_vertices):
        self._bind_geometry(element_or_vertices)

    def evaluateLocal(self, x):
        return self._ref.evaluateLocal(x)

    def evaluateLocalGradient(self, x):
        return self._ref.evaluateLocalGradient(x).dot(self.Jinv.T)
