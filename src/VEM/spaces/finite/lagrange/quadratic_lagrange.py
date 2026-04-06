import numpy

from ...base import SpaceBase
from ...common.triangle_geometry import bind_affine_triangle


class QuadraticLagrangeSpace(SpaceBase):
    def __init__(self, view):
        self.localDofs = 6
        self.view = view
        self.dim = view.dimension
        self.layout = lambda gt: 1 if gt.dim == 0 or gt.dim == 1 else 0
        self.mapper = view.mapper(self.layout)
        self.points = numpy.array(
            [[0, 0], [1, 0], [0, 1], [0.5, 0], [0, 0.5], [0.5, 0.5]],
            dtype=float,
        )

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
        return numpy.array(
            [bary[i] * (2.0 * bary[i] - 1.0) for i in range(3)] +
            [4.0 * bary[(3 - j) % 3] * bary[(4 - j) % 3] for j in range(3)],
            dtype=float,
        )

    def evaluateLocalGradient(self, x):
        l1 = 1.0 - x[0] - x[1]
        l2 = float(x[0])
        l3 = float(x[1])

        grad_hat = numpy.array([
            [-(4.0 * l1 - 1.0), -(4.0 * l1 - 1.0)],
            [ (4.0 * l2 - 1.0),  0.0],
            [ 0.0,               (4.0 * l3 - 1.0)],
            [ 4.0 * (l1 - l2),  -4.0 * l2],
            [ -4.0 * l3,         4.0 * (l1 - l3)],
            [ 4.0 * l3,          4.0 * l2],
        ], dtype=float)
        return grad_hat.dot(self.Jinv.T)

    def interpolate(self, gf):
        dofs = numpy.zeros(len(self.mapper))
        points_t = self.points.transpose()
        for e in self.view.elements:
            dofs[self.mapper(e)] = gf(e, points_t)
        return dofs