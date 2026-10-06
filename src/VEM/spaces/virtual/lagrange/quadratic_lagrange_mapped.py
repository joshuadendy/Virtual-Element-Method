from .quadratic_lagrange_physical import QuadraticLagrangePhysicalVEMSpace


class QuadraticLagrangeMappedVEMSpace(QuadraticLagrangePhysicalVEMSpace):
    """
    k=2 scalar H1-conforming mapped VEM on triangles.

    The projectors are assembled once by a physical space bound to the reference
    triangle. For the Lagrange family the dof map is trivial, so the scalar basis
    transform is the identity and Pi_1 is mapped with J^{-T}.
    """

    def __init__(self, view):
        self._ref = QuadraticLagrangePhysicalVEMSpace(view)
        super().__init__(view)

    def bind(self, element_or_vertices):
        self._bind_geometry(element_or_vertices)

    def evaluateLocal(self, x):
        return self._ref.evaluateLocal(x)

    def evaluateLocalGradient(self, x):
        return self._ref.evaluateLocalGradient(x).dot(self.Jinv.T)
