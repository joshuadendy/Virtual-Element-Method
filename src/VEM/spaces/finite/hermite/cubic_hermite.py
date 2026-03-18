import numpy

from ...base import SpaceBase
from ...common.hermite_mapping import build_cubic_hermite_transform
from ...common.scaled_monomials import P3_EXPONENTS, monomial_gradients, monomials
from ...common.triangle_geometry import bind_affine_triangle


class CubicHermiteSpace(SpaceBase):
    """
    Cubic Hermite triangle (P3) on the reference triangle with vertices
    (0,0), (1,0), (0,1).

    Local DOF ordering (10 dofs):
      0: u(v0)         1: du/dx(v0)      2: du/dy(v0)
      3: u(v1)         4: du/dx(v1)      5: du/dy(v1)
      6: u(v2)         7: du/dx(v2)      8: du/dy(v2)
      9: u(barycentre)
    """

    def __init__(self, view):
        self.localDofs = 10
        self.view = view
        self.dim = view.dimension

        self.layout = lambda gt: (3 if gt.dim == 0 else (1 if gt.dim == self.dim else 0))
        self.mapper = view.mapper(self.layout)

        self.points = numpy.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0 / 3.0, 1.0 / 3.0],
        ], dtype=float)

        self.vertices = numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
        self.J = numpy.eye(2, dtype=float)
        self.M = numpy.eye(self.localDofs, dtype=float)

        self._invDofMatrix = self._buildInverseDofMatrix()
        self.bind(numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float))

    def bind(self, element_or_vertices):
        data = bind_affine_triangle(element_or_vertices)
        self.vertices[0] = data["x0"]
        self.vertices[1] = data["e1"]
        self.vertices[2] = data["e2"]
        self.J = data["J"].copy()
        self.M = build_cubic_hermite_transform(self.J)

    def _buildInverseDofMatrix(self):
        A = numpy.zeros((10, 10), dtype=float)

        row = 0
        for point in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)):
            A[row, :] = monomials(point, P3_EXPONENTS)
            dx, dy = monomial_gradients(point, P3_EXPONENTS)
            A[row + 1, :] = dx
            A[row + 2, :] = dy
            row += 3

        A[row, :] = monomials((1.0 / 3.0, 1.0 / 3.0), P3_EXPONENTS)
        return numpy.linalg.inv(A)

    def _evaluateReferenceLocal(self, x):
        return self._invDofMatrix.T.dot(monomials(x, P3_EXPONENTS))

    def evaluateLocal(self, x):
        phi_ref = self._evaluateReferenceLocal(x)
        return self.M.dot(phi_ref)

    def interpolate(self, gf):
        dofs = numpy.zeros(len(self.mapper), dtype=float)
        ref_vertices = [
            numpy.array([0.0, 0.0]),
            numpy.array([1.0, 0.0]),
            numpy.array([0.0, 1.0]),
        ]
        barycentre = numpy.array([1.0 / 3.0, 1.0 / 3.0])

        if not hasattr(gf, "jacobian"):
            raise NotImplementedError(
                "Hermite interpolation needs derivatives. "
                "Provide a grid function with gf.jacobian(e,x)."
            )

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

            local[9] = float(gf(e, barycentre))
            dofs[idx] = local

        return dofs