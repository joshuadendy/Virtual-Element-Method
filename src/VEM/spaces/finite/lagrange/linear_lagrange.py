import numpy

from ...base import SpaceBase
from ...common.triangle_geometry import bind_affine_triangle


class LinearLagrangeSpace(SpaceBase):
    def __init__(self, view):
        self.localDofs = 3
        self.view = view
        self.dim = view.dimension
        self.layout = lambda gt: 1 if gt.dim == 0 else 0
        self.mapper = view.mapper(self.layout)
        self.points = numpy.array([[0, 0], [1, 0], [0, 1]], dtype=float)

        self.vertices = numpy.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        self.x0 = numpy.array([0.0, 0.0], dtype=float)
        self.J = numpy.eye(2, dtype=float)
        self.Jinv = numpy.eye(2, dtype=float)

    def bind(self, element_or_vertices):
        data = bind_affine_triangle(element_or_vertices)
        self.vertices[0] = data["x0"]
        self.vertices[1] = data["e1"]
        self.vertices[2] = data["e2"]
        self.x0 = data["x0"].copy()
        self.J = data["J"].copy()
        self.Jinv = data["Jinv"].copy()

    def evaluateLocal(self, x):
        bary = 1.0 - x[0] - x[1], x[0], x[1]
        return numpy.array(bary, dtype=float)

    def evaluateLocalGradient(self, x):
        del x
        grad_hat = numpy.array([
            [-1.0, -1.0],
            [ 1.0,  0.0],
            [ 0.0,  1.0],
        ], dtype=float)
        return grad_hat.dot(self.Jinv.T)

    def interpolate(self, gf):
        dofs = numpy.zeros(len(self.mapper))
        for e in self.view.elements:
            dofs[self.mapper(e)] = gf(e, self.points.transpose())
        return dofs