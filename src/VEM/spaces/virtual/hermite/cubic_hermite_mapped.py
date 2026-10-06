from ...common.hermite_mapping import build_k3_mapped_transform
from ...common.scaled_monomials import P1_EXPONENTS, scaled_monomial_inverse_pullback_matrix
from .cubic_hermite_physical import CubicHermitePhysicalVEMSpace


class CubicHermiteMappedVEMSpace(CubicHermitePhysicalVEMSpace):
    """
    The value projector Pi_0 and gradient projector Pi_1 are assembled once by a
    physical space bound to the reference triangle. Pi_1 is mapped to each
    physical element according to

    Pi_1^E = F_{mathrm{curl}}(F^{-*}(Pi_1^{hat E})).

    In code, the mapped gradient basis is formed by
    1. evaluating the reference projected gradient basis,
    2. applying the Hermite DOF/basis transform M,
    3. applying the Jacobian factor on the vector side.
    """

    def __init__(self, view):
        self._ref = CubicHermitePhysicalVEMSpace(view)
        super().__init__(view)

    def bind(self, element_or_vertices):
        self._bind_geometry(element_or_vertices)

        self._constraint_pullback = scaled_monomial_inverse_pullback_matrix(
            Jinv=self.Jinv,
            h=self.hE,
            h_hat=self.hE_hat,
            exponents=P1_EXPONENTS,
        )

        self.M = build_k3_mapped_transform(
            J=self.J,
            hV_hat=self._reference_vertex_h(),
            hV_local=self._hV_local,
        )

    def _moment_basis_phys(self, x_phys):
        """
        Physical evaluation of the transported reference M1 basis.

        These are the interior moment functionals appearing in the mapped tuple
        C = F_{-*}(hat C), rather than the raw physical scaled monomials. Keeping the
        mapped space in this transported basis avoids re-expressing the last
        three dofs through an extra block in M.
        """
        return self._constraint_pullback.dot(self._m1_basis_phys(x_phys))

    def evaluateLocal(self, x):
        return self.M.dot(self._ref.evaluateLocal(x))

    def _evaluateLocalValueProjectionGradient(self, x):
        return self.M.dot(self._ref._evaluateLocalValueProjectionGradient(x)).dot(self.Jinv.T)

    def evaluateLocalGradient(self, x):
        """
        Map the reference gradient projector onto the physical element.
        """
        return self.M.dot(self._ref.evaluateLocalGradient(x)).dot(self.Jinv.T)
