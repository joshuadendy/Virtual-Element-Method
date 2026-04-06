import numpy

from ...base import SpaceBase
from ...common.scaled_monomials import P4_EXPONENTS, monomial_gradients, monomials
from ...common.triangle_geometry import bind_affine_triangle


class QuarticHermiteSpace(SpaceBase):
    """
    Quartic Hermite-style triangle (P4) on the reference triangle.

    Local dof ordering (15 dofs):
      0:  u(v0)        1: du/dx(v0)      2: du/dy(v0)
      3:  u(v1)        4: du/dx(v1)      5: du/dy(v1)
      6:  u(v2)        7: du/dx(v2)      8: du/dy(v2)
      9:  u((1/2,0))  10: u((0,1/2))    11: u((1/2,1/2))
      12: u((1/5,1/5)) 13: u((3/5,1/5)) 14: u((1/5,3/5))

    This is a P4 reference element with Hermite vertex data plus six additional value
    nodes. Under affine mapping the derivative 2x2 vertex blocks transform with J,
    while all value nodes map trivially.
    """

    def __init__(self, view):
        self.localDofs = 15
        self.view = view
        self.dim = view.dimension

        self.layout = lambda gt: (3 if gt.dim == 0 else (1 if gt.dim == 1 else (3 if gt.dim == self.dim else 0)))
        self.mapper = view.mapper(self.layout)

        self.points = numpy.array([
            [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
            [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
            [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
            [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
            [0.2, 0.2], [0.6, 0.2], [0.2, 0.6],
        ], dtype=float)

        self.vertices = numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
        self.x0 = numpy.array([0.0, 0.0], dtype=float)
        self.J = numpy.eye(2, dtype=float)
        self.Jinv = numpy.eye(2, dtype=float)
        self.M = numpy.eye(self.localDofs, dtype=float)

        self._invDofMatrix = self._build_inverse_dof_matrix()
        self.bind(numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float))

    def bind(self, element_or_vertices):
        data = bind_affine_triangle(element_or_vertices)
        self.vertices[0] = data["x0"]
        self.vertices[1] = data["e1"]
        self.vertices[2] = data["e2"]
        self.x0 = data["x0"].copy()
        self.J = data["J"].copy()
        self.Jinv = data["Jinv"].copy()
        self.M = numpy.eye(self.localDofs, dtype=float)
        for base in (0, 3, 6):
            sl = slice(base + 1, base + 3)
            self.M[sl, sl] = self.J

    def _build_inverse_dof_matrix(self):
        A = numpy.zeros((self.localDofs, self.localDofs), dtype=float)

        row = 0
        for point in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)):
            A[row, :] = monomials(point, P4_EXPONENTS)
            dx, dy = monomial_gradients(point, P4_EXPONENTS)
            A[row + 1, :] = dx
            A[row + 2, :] = dy
            row += 3

        for point in ((0.5, 0.0), (0.0, 0.5), (0.5, 0.5), (0.2, 0.2), (0.6, 0.2), (0.2, 0.6)):
            A[row, :] = monomials(point, P4_EXPONENTS)
            row += 1

        return numpy.linalg.inv(A)

    def _evaluate_reference_local(self, x):
        return self._invDofMatrix.T.dot(monomials(x, P4_EXPONENTS))

    def evaluateLocal(self, x):
        phi_ref = self._evaluate_reference_local(x)
        return self.M.dot(phi_ref)

    def evaluateLocalGradient(self, x):
        dx, dy = monomial_gradients(x, P4_EXPONENTS)
        grad_hat = self._invDofMatrix.T.dot(numpy.column_stack((dx, dy)))
        return self.M.dot(grad_hat).dot(self.Jinv.T)

    def interpolate(self, gf):
        dofs = numpy.zeros(len(self.mapper), dtype=float)

        if not hasattr(gf, "jacobian"):
            raise NotImplementedError(
                "Quartic Hermite interpolation needs derivatives. "
                "Provide a grid function with gf.jacobian(e,x)."
            )

        ref_vertices = [
            numpy.array([0.0, 0.0]),
            numpy.array([1.0, 0.0]),
            numpy.array([0.0, 1.0]),
        ]
        extra_points = [
            numpy.array([0.5, 0.0]),
            numpy.array([0.0, 0.5]),
            numpy.array([0.5, 0.5]),
            numpy.array([0.2, 0.2]),
            numpy.array([0.6, 0.2]),
            numpy.array([0.2, 0.6]),
        ]

        for e in self.view.elements:
            self.bind(e)
            idx = self.mapper(e)
            local = numpy.zeros(self.localDofs, dtype=float)

            for i, xi in enumerate(ref_vertices):
                base = 3 * i
                local[base] = float(gf(e, xi))
                g_phys = numpy.asarray(gf.jacobian(e, xi)).reshape(-1)
                if g_phys.size < 2:
                    raise ValueError("gf.jacobian(e,x) did not return a 2D gradient.")
                local[base + 1] = float(g_phys[0])
                local[base + 2] = float(g_phys[1])

            for offset, xi in enumerate(extra_points):
                local[9 + offset] = float(gf(e, xi))

            dofs[idx] = local

        return dofs
