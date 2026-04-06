import numpy

from ...base import SpaceBase
from ...common.scaled_monomials import P1_EXPONENTS, scaled_monomial_gradients, scaled_monomials
from ...common.triangle_geometry import bind_affine_triangle


class LinearLagrangePhysicalVEMSpace(SpaceBase):
    """
    k=1 scalar H1-conforming VEM (value projection only) on triangles.

    evaluateLocal(xhat) returns the VALUE PROJECTION of the (virtual) local
    basis, i.e. [Pi^E_0 phi_i](xhat), not the true virtual basis values
    phi_i(xhat).
    """

    def __init__(self, view):
        self.localDofs = 3
        self.view = view
        self.dim = view.dimension
        self.layout = lambda gt: 1 if gt.dim == 0 else 0
        self.mapper = view.mapper(self.layout)
        self.points = numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        self.vertices = numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)

        self.J = numpy.eye(2)
        self.Jinv = numpy.eye(2)
        self.x0 = numpy.array([0.0, 0.0], dtype=float)
        self.xE = numpy.array([1.0 / 3.0, 1.0 / 3.0], dtype=float)
        self.hE = numpy.sqrt(2.0)

        self._Pi0Coeffs = numpy.zeros((3, 3), dtype=float)
        self.bind(numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float))

    def bind(self, element_or_vertices):
        data = bind_affine_triangle(element_or_vertices)
        self.vertices[0] = data["x0"]
        self.vertices[1] = data["e1"]
        self.vertices[2] = data["e2"]
        self.x0 = data["x0"].copy()
        self.J = data["J"].copy()
        self.Jinv = data["Jinv"].copy()
        self.xE = data["xE"].copy()
        self.hE = float(data["hE"])

        A = numpy.zeros((self.localDofs, 3), dtype=float)
        for i, xv in enumerate((self.x0, self.x0 + self.vertices[1], self.x0 + self.vertices[2])):
            A[i, :] = self._poly_basis_value(xv)

        self._Pi0Coeffs = numpy.linalg.solve(A, numpy.eye(self.localDofs, dtype=float))

    def _poly_basis_value(self, x_phys):
        return scaled_monomials(x_phys, self.xE, self.hE, P1_EXPONENTS)

    def _poly_basis_grad_value(self, x_phys):
        dmx, dmy = scaled_monomial_gradients(x_phys, self.xE, self.hE, P1_EXPONENTS)
        return numpy.column_stack((dmx, dmy))

    def _physical_point(self, xhat):
        return self.x0 + self.J.dot(numpy.asarray(xhat, dtype=float))

    def evaluateLocal(self, x):
        x_phys = self._physical_point(x)
        return self._Pi0Coeffs.T.dot(self._poly_basis_value(x_phys))

    def evaluateLocalGradient(self, x):
        x_phys = self._physical_point(x)
        return self._Pi0Coeffs.T.dot(self._poly_basis_grad_value(x_phys))

    def interpolate(self, gf):
        dofs = numpy.zeros(len(self.mapper))
        ptsT = self.points.transpose()
        for e in self.view.elements:
            dofs[self.mapper(e)] = gf(e, ptsT)
        return dofs