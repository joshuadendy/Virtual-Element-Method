import numpy

from .physical import EDGES, PhysicalVEMSpace


class MappedVEMSpace(PhysicalVEMSpace):
    """
    Reference-mapped counterpart of PhysicalVEMSpace (Section 4.2).

    Pi_0 and Pi_1 are assembled once on the reference triangle and evaluated on
    each element as the surrogates (4.73) and (4.75):
      Pi_0 = M F^*(Pi_0[hat Phi]),  Pi_1 = J^{-T} M F^*(Pi_1[hat Phi]),
    where M = V^T and V F_*(Lambda) = hat Lambda. V is the identity apart from the
    Hermite gradient blocks (h_hat_v / h_v) J^T of (5.20) and a (-1)^j on the
    moments of edges parametrised against the reference orientation. Interior
    moments use the transported reference basis, C_0 = F_{-*}(hat C_0).
    """

    def __init__(self, view, order, hermite=False):
        self._ref = PhysicalVEMSpace(view, order, hermite)
        super().__init__(view, order, hermite)

    def bind(self, element_or_vertices):
        self._bind_geometry(element_or_vertices)
        self.M = numpy.eye(self.localDofs)
        if self.hermite:
            for i in range(3):
                self.M[3 * i + 1:3 * i + 3, 3 * i + 1:3 * i + 3] = (self._ref._hV[i] / self._hV[i]) * self.J
        signs = (-1.0) ** numpy.arange(self._ne)
        for i, (edge, ref_edge) in enumerate(zip(self._edges, EDGES)):
            if edge != ref_edge:
                start = 3 * self._nv + i * self._ne
                self.M[start:start + self._ne, start:start + self._ne] = numpy.diag(signs)

    def _moment_basis(self, xhat):
        return self._ref._moment_basis(xhat)

    def evaluateLocal(self, x):
        return self.M @ self._ref.evaluateLocal(x)

    def evaluateLocalGradient(self, x):
        return self.M @ self._ref.evaluateLocalGradient(x) @ self.Jinv

    def localProjectorDofs(self):
        return numpy.linalg.solve(self.M.T, self._ref.localProjectorDofs() @ self.M.T)
