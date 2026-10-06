from ...common.hermite_mapping import build_k4_mapped_transform
from ...common.scaled_monomials import P2_EXPONENTS, scaled_monomial_inverse_pullback_matrix
from .quartic_hermite_physical import QuarticHermitePhysicalVEMSpace


class QuarticHermiteMappedVEMSpace(QuarticHermitePhysicalVEMSpace):
    """
    k=4 mapped Hermite-style VEM on triangles.

    The value and gradient projectors are assembled once by a physical space bound
    to the reference triangle. The value projector is evaluated on each element via
    the Section 4.2.3 surrogate construction. The gradient projector is mapped with
    the covariant Piola factor J^{-T}, independently of Pi_0.
    """

    def __init__(self, view):
        self._ref = QuarticHermitePhysicalVEMSpace(view)
        super().__init__(view)

    def bind(self, element_or_vertices):
        self._bind_geometry(element_or_vertices)

        self._constraint_pullback = scaled_monomial_inverse_pullback_matrix(
            Jinv=self.Jinv,
            h=self.hE,
            h_hat=self.hE_hat,
            exponents=P2_EXPONENTS,
        )

        self.M = build_k4_mapped_transform(
            J=self.J,
            hV_hat=self._reference_vertex_h(),
            hV_local=self._hV_local,
        )

    def _moment_basis_phys(self, x_phys):
        return self._constraint_pullback.dot(self._m2_basis_phys(x_phys))

    def evaluateLocal(self, x):
        return self.M.dot(self._ref.evaluateLocal(x))

    def _evaluateLocalValueProjectionGradient(self, x):
        return self.M.dot(self._ref._evaluateLocalValueProjectionGradient(x)).dot(self.Jinv.T)

    def evaluateLocalGradient(self, x):
        return self.M.dot(self._ref.evaluateLocalGradient(x)).dot(self.Jinv.T)
