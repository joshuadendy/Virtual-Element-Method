import numpy

from ...base import SpaceBase
from ...common.scaled_monomials import P1_EXPONENTS, scaled_monomial_gradients, scaled_monomials
from ...common.triangle_geometry import bind_affine_triangle


class LinearLagrangePhysicalVEMSpace(SpaceBase):
    """
    k=1 scalar H1-conforming VEM on triangles.

    evaluateLocal(xhat) returns the VALUE PROJECTION of the (virtual) local
    basis, i.e. [Pi^E_0 phi_i](xhat), not the true virtual basis values
    phi_i(xhat).

    The gradient projection Pi_1 is assembled explicitly on each physical
    element, rather than being inferred as grad(Pi_0).
    """

    def __init__(self, view):
        self.localDofs = 3
        self.gradPolyDim = 2
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
        self.area = 0.5
        self._edge_pairs = ((0, 1), (1, 2), (2, 0))

        # 2-point Gauss-Legendre rule on [0, 1]
        xi = 1.0 / numpy.sqrt(3.0)
        self._edge_quad_r = 0.5 * numpy.array([1.0 - xi, 1.0 + xi], dtype=float)
        self._edge_quad_w = numpy.array([0.5, 0.5], dtype=float)

        self._Pi0Coeffs = numpy.zeros((3, 3), dtype=float)
        self._Pi1Coeffs = numpy.zeros((self.gradPolyDim, self.localDofs), dtype=float)
        self.bind(numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float))

    def bind(self, element_or_vertices):
        data = bind_affine_triangle(element_or_vertices)
        self.vertices[0] = data["x0"]
        self.vertices[1] = data["e1"]
        self.vertices[2] = data["e2"]
        self.x0 = data["x0"].copy()
        self.J = data["J"].copy()
        self.Jinv = data["Jinv"].copy()
        self.area = float(data["area"])
        self.xE = data["xE"].copy()
        self.hE = float(data["hE"])

        A = numpy.zeros((self.localDofs, 3), dtype=float)
        for i, xv in enumerate((self.x0, self.x0 + self.vertices[1], self.x0 + self.vertices[2])):
            A[i, :] = self._poly_basis_value(xv)

        self._Pi0Coeffs = numpy.linalg.solve(A, numpy.eye(self.localDofs, dtype=float))
        self._Pi1Coeffs = self._build_physical_gradient_projector()

    def _poly_basis_value(self, x_phys):
        return scaled_monomials(x_phys, self.xE, self.hE, P1_EXPONENTS)

    def _poly_basis_grad_value(self, x_phys):
        dmx, dmy = scaled_monomial_gradients(x_phys, self.xE, self.hE, P1_EXPONENTS)
        return numpy.column_stack((dmx, dmy))

    def _vector_p0_basis_phys(self):
        basis = numpy.zeros((self.gradPolyDim, 2), dtype=float)
        basis[0, 0] = 1.0
        basis[1, 1] = 1.0
        return basis

    def _physical_vertices(self):
        return (
            self.x0.copy(),
            self.x0 + self.vertices[1],
            self.x0 + self.vertices[2],
        )

    def _edge_geometry_from_vertices(self, verts, ia, ib, x_center):
        xa = numpy.asarray(verts[ia], dtype=float)
        xb = numpy.asarray(verts[ib], dtype=float)
        edge = xb - xa
        length = float(numpy.linalg.norm(edge))
        if length <= 1e-14:
            raise ValueError("Degenerate edge encountered while building Pi_1.")
        tangent = edge / length
        normal = numpy.array([tangent[1], -tangent[0]], dtype=float)
        midpoint = 0.5 * (xa + xb)
        if numpy.dot(normal, x_center - midpoint) > 0.0:
            normal *= -1.0
        return xa, edge, length, normal

    def _edge_trace_from_local_dofs(self, local_dofs, ia, ib, r):
        return (1.0 - float(r)) * float(local_dofs[ia]) + float(r) * float(local_dofs[ib])

    def _solve_dense_system(self, A, B):
        try:
            return numpy.linalg.solve(A, B)
        except numpy.linalg.LinAlgError:
            return numpy.linalg.lstsq(A, B, rcond=None)[0]

    def _physical_point(self, xhat):
        return self.x0 + self.J.dot(numpy.asarray(xhat, dtype=float))

    def _build_physical_gradient_projector(self):
        mass = self.area * numpy.eye(self.gradPolyDim, dtype=float)
        rhs = numpy.zeros((self.gradPolyDim, self.localDofs), dtype=float)

        verts = self._physical_vertices()
        x_center = self.xE

        for ia, ib in self._edge_pairs:
            xa, edge, length, normal = self._edge_geometry_from_vertices(
                verts, ia, ib, x_center
            )

            flux_basis = self._vector_p0_basis_phys().dot(normal)
            for r, wr in zip(self._edge_quad_r, self._edge_quad_w):
                _ = xa + float(r) * edge
                for j in range(self.localDofs):
                    local = numpy.zeros(self.localDofs, dtype=float)
                    local[j] = 1.0
                    trace_val = self._edge_trace_from_local_dofs(local, ia, ib, r)
                    rhs[:, j] += length * float(wr) * flux_basis * trace_val

        return self._solve_dense_system(mass, rhs)

    def evaluateLocal(self, x):
        x_phys = self._physical_point(x)
        return self._Pi0Coeffs.T.dot(self._poly_basis_value(x_phys))

    def evaluateLocalGradient(self, x):
        return self._Pi1Coeffs.T.dot(self._vector_p0_basis_phys())

    def interpolate(self, gf):
        dofs = numpy.zeros(len(self.mapper))
        ptsT = self.points.transpose()
        for e in self.view.elements:
            dofs[self.mapper(e)] = gf(e, ptsT)
        return dofs
