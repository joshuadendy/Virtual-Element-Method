import numpy

from ...base import SpaceBase
from ...common.scaled_monomials import P1_EXPONENTS, scaled_monomial_gradients, scaled_monomials
from ...common.triangle_geometry import bind_affine_triangle


class LinearLagrangeMappedVEMSpace(SpaceBase):
    """
    k=1 scalar H1-conforming VEM on triangles.

    The scalar value projector Pi_0 is assembled once on the reference triangle.
    The gradient projector Pi_1 is also assembled explicitly on the reference
    triangle and then mapped to the physical element with the covariant Piola
    factor J^{-T}.

    For this lowest-order triangular case Pi_1 coincides algebraically with the
    gradient of Pi_0, but we keep a separate projector so that it can be scaled,
    inspected, or replaced independently.
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

        self.x0 = numpy.array([0.0, 0.0], dtype=float)
        self.J = numpy.eye(2, dtype=float)
        self.Jinv = numpy.eye(2, dtype=float)
        self.detJ = 1.0

        self.xE_hat = numpy.array([1.0 / 3.0, 1.0 / 3.0], dtype=float)
        self.hE_hat = numpy.sqrt(2.0)
        self.area_hat = 0.5

        self._ref_vertices = (
            numpy.array([0.0, 0.0], dtype=float),
            numpy.array([1.0, 0.0], dtype=float),
            numpy.array([0.0, 1.0], dtype=float),
        )
        self._edge_pairs = ((0, 1), (1, 2), (2, 0))

        # 2-point Gauss-Legendre rule on [0, 1]
        xi = 1.0 / numpy.sqrt(3.0)
        self._edge_quad_r = 0.5 * numpy.array([1.0 - xi, 1.0 + xi], dtype=float)
        self._edge_quad_w = numpy.array([0.5, 0.5], dtype=float)

        Ahat = numpy.zeros((self.localDofs, 3), dtype=float)
        for i, xhat_v in enumerate(self._ref_vertices):
            Ahat[i, :] = self._poly_basis_ref(xhat_v)
        self._Pi0CoeffsRef = numpy.linalg.solve(Ahat, numpy.eye(self.localDofs, dtype=float))

        # Explicit reference gradient projector Pi_1.
        self._Pi1CoeffsRef = self._build_reference_gradient_projector()

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

    def _poly_basis_grad_ref(self, xhat):
        dxi, deta = scaled_monomial_gradients(xhat, self.xE_hat, self.hE_hat, P1_EXPONENTS)
        return numpy.column_stack((dxi, deta))

    def _vector_p0_basis_ref(self):
        basis = numpy.zeros((self.gradPolyDim, 2), dtype=float)
        basis[0, 0] = 1.0
        basis[1, 1] = 1.0
        return basis

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

    def _build_reference_gradient_projector(self):
        """
        Assemble Pi_1 on the reference element only.

        For k=1, span(B_1) is the space of constant vector fields, so the volume
        term vanishes because div(q)=0 for all q in span(B_1). We still assemble
        the projector from the defining variational identity so it remains an
        explicit, separately scalable object.
        """
        mass = self.area_hat * numpy.eye(self.gradPolyDim, dtype=float)
        rhs = numpy.zeros((self.gradPolyDim, self.localDofs), dtype=float)

        verts = self._ref_vertices
        x_center = self.xE_hat

        for ia, ib in self._edge_pairs:
            xa, edge, length, normal = self._edge_geometry_from_vertices(
                verts, ia, ib, x_center
            )

            flux_basis = self._vector_p0_basis_ref().dot(normal)
            for r, wr in zip(self._edge_quad_r, self._edge_quad_w):
                _ = xa + float(r) * edge
                for j in range(self.localDofs):
                    local = numpy.zeros(self.localDofs, dtype=float)
                    local[j] = 1.0
                    trace_val = self._edge_trace_from_local_dofs(local, ia, ib, r)
                    rhs[:, j] += length * float(wr) * flux_basis * trace_val

        return self._solve_dense_system(mass, rhs)

    def evaluateLocal(self, x):
        return self._Pi0CoeffsRef.T.dot(self._poly_basis_ref(x))

    def evaluateLocalGradient(self, x):
        grad_hat = self._Pi1CoeffsRef.T.dot(self._vector_p0_basis_ref())
        return grad_hat.dot(self.Jinv.T)

    def interpolate(self, gf):
        dofs = numpy.zeros(len(self.mapper))
        ptsT = self.points.transpose()
        for e in self.view.elements:
            dofs[self.mapper(e)] = gf(e, ptsT)
        return dofs
