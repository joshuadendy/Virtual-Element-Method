import numpy

from ...base import SpaceBase
from ...common.scaled_monomials import P1_EXPONENTS, scaled_monomials
from ...common.triangle_geometry import bind_affine_triangle


class LinearLagrangeMappedVEMSpace(SpaceBase):
    """
    k=1 scalar H1-conforming VEM (value projection only) on triangles,
    but with the value projection computed ONCE on the reference triangle.
    """

    def __init__(self, view):
        self.localDofs = 3
        self.view = view
        self.dim = view.dimension
        self.layout = lambda gt: 1 if gt.dim == 0 else 0
        self.mapper = view.mapper(self.layout)
        self.points = numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        self.vertices = numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)

        self.x0 = numpy.array([0.0, 0.0], dtype=float)
        self.J = numpy.eye(2, dtype=float)
        self.Jinv = numpy.eye(2, dtype=float)
        self.detJ = 1.0

        self.xE_hat = numpy.array([1.0 / 3.0, 1.0 / 3.0], dtype=float)
        self.hE_hat = numpy.sqrt(2.0)

        Ahat = numpy.zeros((self.localDofs, 3), dtype=float)
        for i, xhat_v in enumerate((
            numpy.array([0.0, 0.0]),
            numpy.array([1.0, 0.0]),
            numpy.array([0.0, 1.0]),
        )):
            Ahat[i, :] = self._poly_basis_ref(xhat_v)
        self._Pi0CoeffsRef = numpy.linalg.solve(Ahat, numpy.eye(self.localDofs, dtype=float))

        self.bind(numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float))

    def bind(self, element_or_vertices):
        data = bind_affine_triangle(element_or_vertices)
        self.vertices[0] = data["x0"]
        self.vertices[1] = data["e1"]
        self.vertices[2] = data["e2"]
        self.x0 = data["x0"].copy()
        self.J = data["J"].copy()
        self.Jinv = data["Jinv"].copy()
        self.detJ = float(data["detJ"])

    def _poly_basis_ref(self, xhat):
        return scaled_monomials(xhat, self.xE_hat, self.hE_hat, P1_EXPONENTS)

    def evaluateLocal(self, x):
        return self._Pi0CoeffsRef.T.dot(self._poly_basis_ref(x))

    def interpolate(self, gf):
        dofs = numpy.zeros(len(self.mapper))
        ptsT = self.points.transpose()
        for e in self.view.elements:
            dofs[self.mapper(e)] = gf(e, ptsT)
        return dofs