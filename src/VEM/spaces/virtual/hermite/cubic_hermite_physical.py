import numpy
from dune.geometry import quadratureRule

from ...base import SpaceBase
from ...common.cls_projector import solve_cls_kkt_all_rhs
from ...common.scaled_monomials import (
    P1_EXPONENTS,
    P2_EXPONENTS,
    P3_EXPONENTS,
    scaled_monomial_gradients,
    scaled_monomials,
)
from ...common.triangle_geometry import bind_affine_triangle
from ...common.vertex_scaling import build_vertex_effective_h


class CubicHermitePhysicalVEMSpace(SpaceBase):
    """
    k=3 Hermite-style VEM space assembled directly on each physical element.

    The scalar value projector Pi_0 is still obtained from the constrained least
    squares system. The gradient returned by evaluateLocalGradient is now the
    true Hermite VEM gradient projector Pi_1 rather than grad(Pi_0).
    """

    def __init__(self, view):
        self.view = view
        self.dim = view.dimension
        self.localDofs = 12
        self.polyDim = 10
        self.constraintDim = 3
        self.gradScalarDim = len(P2_EXPONENTS)
        self.gradPolyDim = 2 * self.gradScalarDim
        self.vertices = numpy.array([[0, 0], [1, 0], [0, 1]], dtype=float)

        self.layout = lambda gt: (3 if gt.dim == 0 else (3 if gt.dim == self.dim else 0))
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
            [1.0 / 3.0, 1.0 / 3.0],
            [1.0 / 3.0, 1.0 / 3.0],
        ], dtype=float)

        self.x0 = numpy.array([0.0, 0.0], dtype=float)
        self.J = numpy.eye(2, dtype=float)
        self.Jinv = numpy.eye(2, dtype=float)
        self.detJ = 1.0
        self.area = 0.5

        self.xE = numpy.array([1.0 / 3.0, 1.0 / 3.0], dtype=float)
        self.hE = numpy.sqrt(2.0)

        self.xE_hat = numpy.array([1.0 / 3.0, 1.0 / 3.0], dtype=float)
        self.hE_hat = numpy.sqrt(2.0)
        self._ref_vertices = (
            numpy.array([0.0, 0.0], dtype=float),
            numpy.array([1.0, 0.0], dtype=float),
            numpy.array([0.0, 1.0], dtype=float),
        )
        self._edge_pairs = ((0, 1), (1, 2), (2, 0))
        self._edge_quad_r, self._edge_quad_w = self._build_unit_interval_quadrature(order=6)

        tri_type = None
        for e in self.view.elements:
            tri_type = e.type
            break
        if tri_type is None:
            raise RuntimeError("Grid view appears to have no elements.")

        self._momentQuad = quadratureRule(tri_type, 6)

        self._constraint_rhs_selector = numpy.zeros((self.constraintDim, self.localDofs), dtype=float)
        self._constraint_rhs_selector[0, 9] = 1.0
        self._constraint_rhs_selector[1, 10] = 1.0
        self._constraint_rhs_selector[2, 11] = 1.0

        self._Pi0Coeffs = numpy.zeros((self.polyDim, self.localDofs), dtype=float)
        self._Pi1Coeffs = numpy.zeros((self.gradPolyDim, self.localDofs), dtype=float)
        self._vertex_h = build_vertex_effective_h(
            self.view,
            self.mapper,
            measure="adjacent_edge_average",
        )
        self._hV_local = self._reference_vertex_h()

        self.bind(numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float))

    @staticmethod
    def _build_unit_interval_quadrature(order=6):
        pts, wts = numpy.polynomial.legendre.leggauss(order)
        r = 0.5 * (pts + 1.0)
        w = 0.5 * wts
        return r, w

    @staticmethod
    def _reference_vertex_h():
        e01 = 1.0
        e12 = numpy.sqrt(2.0)
        e20 = 1.0
        return numpy.array([
            0.5 * (e01 + e20),
            0.5 * (e01 + e12),
            0.5 * (e12 + e20),
        ], dtype=float)

    @staticmethod
    def _solve_dense_system(A, B):
        try:
            return numpy.linalg.solve(A, B)
        except numpy.linalg.LinAlgError:
            return numpy.linalg.lstsq(A, B, rcond=None)[0]

    def bind(self, element_or_vertices):
        if hasattr(element_or_vertices, "geometry"):
            idx = self.mapper(element_or_vertices)
            self._hV_local = numpy.array([
                self._vertex_h[int(idx[0])],
                self._vertex_h[int(idx[3])],
                self._vertex_h[int(idx[6])],
            ], dtype=float)
        else:
            self._hV_local = self._reference_vertex_h()

        data = bind_affine_triangle(element_or_vertices)
        self.vertices[0] = data["x0"]
        self.vertices[1] = data["e1"]
        self.vertices[2] = data["e2"]
        self.x0 = data["x0"].copy()
        self.J = data["J"].copy()
        self.Jinv = data["Jinv"].copy()
        self.detJ = float(data["detJ"])
        self.area = float(data["area"])
        self.xE = data["xE"].copy()
        self.hE = float(data["hE"])

        A, C = self._build_physical_A_and_C()
        self._Pi0Coeffs = solve_cls_kkt_all_rhs(A=A, C=C, G=self._constraint_rhs_selector)
        self._Pi1Coeffs = self._build_physical_gradient_projector()

    def _physical_point(self, xhat):
        return self.x0 + self.J.dot(numpy.asarray(xhat, dtype=float))

    def _reference_point(self, xphys):
        return self.Jinv.dot(numpy.asarray(xphys, dtype=float) - self.x0)

    def _physical_vertices(self):
        return (
            self.x0.copy(),
            self.x0 + self.vertices[1],
            self.x0 + self.vertices[2],
        )

    def _p3_basis_phys(self, x_phys):
        return scaled_monomials(x_phys, self.xE, self.hE, P3_EXPONENTS)

    def _p3_basis_grad_phys(self, x_phys):
        dmx, dmy = scaled_monomial_gradients(x_phys, self.xE, self.hE, P3_EXPONENTS)
        return numpy.column_stack((dmx, dmy))

    def _p2_basis_phys(self, x_phys):
        return scaled_monomials(x_phys, self.xE, self.hE, P2_EXPONENTS)

    def _p2_basis_grad_phys(self, x_phys):
        return scaled_monomial_gradients(x_phys, self.xE, self.hE, P2_EXPONENTS)

    def _vector_p2_basis_phys(self, x_phys):
        vals = self._p2_basis_phys(x_phys)
        basis = numpy.zeros((self.gradPolyDim, 2), dtype=float)
        basis[:self.gradScalarDim, 0] = vals
        basis[self.gradScalarDim:, 1] = vals
        return basis

    def _vector_p2_div_phys(self, x_phys):
        dmx, dmy = self._p2_basis_grad_phys(x_phys)
        return numpy.concatenate((dmx, dmy))

    def _m1_basis_phys(self, x_phys):
        return scaled_monomials(x_phys, self.xE, self.hE, P1_EXPONENTS)

    def _build_physical_A_and_C(self):
        A = numpy.zeros((self.localDofs, self.polyDim), dtype=float)
        C = numpy.zeros((self.constraintDim, self.polyDim), dtype=float)

        verts = self._physical_vertices()

        row = 0
        for iv, xv in enumerate(verts):
            A[row, :] = self._p3_basis_phys(xv)
            grad = self._p3_basis_grad_phys(xv)
            h_a = self._hV_local[iv]
            A[row + 1, :] = h_a * grad[:, 0]
            A[row + 2, :] = h_a * grad[:, 1]
            row += 3

        mom_rows = numpy.zeros((3, self.polyDim), dtype=float)
        for p in self._momentQuad:
            xhat = p.position
            x_phys = self._physical_point(xhat)
            mom_rows += (
                abs(self.detJ) * float(p.weight) / self.area
            ) * numpy.outer(self._m1_basis_phys(x_phys), self._p3_basis_phys(x_phys))

        A[9:12, :] = mom_rows
        C[:, :] = mom_rows
        return A, C

    def _edge_geometry_from_vertices(self, verts, ia, ib):
        xa = numpy.asarray(verts[ia], dtype=float)
        xb = numpy.asarray(verts[ib], dtype=float)
        edge = xb - xa
        length = float(numpy.linalg.norm(edge))
        if length <= 1e-14:
            raise ValueError("Degenerate edge encountered while building Hermite projector.")
        tangent = edge / length
        normal = numpy.array([tangent[1], -tangent[0]], dtype=float)
        midpoint = 0.5 * (xa + xb)
        if numpy.dot(normal, self.xE - midpoint) > 0.0:
            normal *= -1.0
        return xa, edge, length, tangent, normal

    def _edge_trace_from_local_dofs(self, local_dofs, verts, ia, ib, r):
        _, _, length, tangent, _ = self._edge_geometry_from_vertices(verts, ia, ib)

        base_a = 3 * ia
        base_b = 3 * ib

        u_a = float(local_dofs[base_a])
        u_b = float(local_dofs[base_b])

        grad_a = numpy.array([
            float(local_dofs[base_a + 1]) / float(self._hV_local[ia]),
            float(local_dofs[base_a + 2]) / float(self._hV_local[ia]),
        ], dtype=float)
        grad_b = numpy.array([
            float(local_dofs[base_b + 1]) / float(self._hV_local[ib]),
            float(local_dofs[base_b + 2]) / float(self._hV_local[ib]),
        ], dtype=float)

        m_a = length * float(numpy.dot(tangent, grad_a))
        m_b = length * float(numpy.dot(tangent, grad_b))

        rr = float(r)
        h00 = 2.0 * rr**3 - 3.0 * rr**2 + 1.0
        h10 = rr**3 - 2.0 * rr**2 + rr
        h01 = -2.0 * rr**3 + 3.0 * rr**2
        h11 = rr**3 - rr**2

        return u_a * h00 + m_a * h10 + u_b * h01 + m_b * h11

    def _build_physical_gradient_projector(self):
        mass = numpy.zeros((self.gradPolyDim, self.gradPolyDim), dtype=float)
        rhs = numpy.zeros((self.gradPolyDim, self.localDofs), dtype=float)
        verts = self._physical_vertices()

        for p in self._momentQuad:
            xhat = p.position
            x_phys = self._physical_point(xhat)
            w = float(abs(self.detJ) * p.weight)
            vec_basis = self._vector_p2_basis_phys(x_phys)
            div_basis = self._vector_p2_div_phys(x_phys)
            pi0_vals = self._Pi0Coeffs.T.dot(self._p3_basis_phys(x_phys))

            mass += w * vec_basis.dot(vec_basis.T)
            rhs -= w * numpy.outer(div_basis, pi0_vals)

        for ia, ib in self._edge_pairs:
            xa, edge, length, _, normal = self._edge_geometry_from_vertices(verts, ia, ib)
            for r, wr in zip(self._edge_quad_r, self._edge_quad_w):
                x_phys = xa + float(r) * edge
                flux_basis = self._vector_p2_basis_phys(x_phys).dot(normal)
                for j in range(self.localDofs):
                    local = numpy.zeros(self.localDofs, dtype=float)
                    local[j] = 1.0
                    trace_val = self._edge_trace_from_local_dofs(
                        local,
                        verts=verts,
                        ia=ia,
                        ib=ib,
                        r=r,
                    )
                    rhs[:, j] += length * float(wr) * flux_basis * trace_val

        return self._solve_dense_system(mass, rhs)

    def evaluateLocal(self, x):
        x_phys = self._physical_point(x)
        return self._Pi0Coeffs.T.dot(self._p3_basis_phys(x_phys))

    def _evaluateLocalValueProjectionGradient(self, x):
        """
        Gradient of the scalar value projection Pi_0.

        This is used only when applying Hermite derivative dofs to the projected
        scalar polynomial inside localProjectorDofs(). The Poisson consistency
        term should continue to use evaluateLocalGradient(), i.e. the separate
        gradient projector Pi_1.
        """
        x_phys = self._physical_point(x)
        dmx, dmy = scaled_monomial_gradients(x_phys, self.xE, self.hE, P3_EXPONENTS)
        grad_basis = numpy.column_stack((dmx, dmy))
        return self._Pi0Coeffs.T.dot(grad_basis)

    def evaluateLocalGradient(self, x):
        x_phys = self._physical_point(x)
        return self._Pi1Coeffs.T.dot(self._vector_p2_basis_phys(x_phys))

    def evaluatePhysical(self, x_phys):
        return self.evaluateLocal(self._reference_point(x_phys))

    def localProjectorDofs(self):
        P = numpy.zeros((self.localDofs, self.localDofs), dtype=float)

        row = 0
        for iv, xhat_v in enumerate(self._ref_vertices):
            P[row, :] = self.evaluateLocal(xhat_v)
            grad = self._evaluateLocalValueProjectionGradient(xhat_v)
            h_a = self._hV_local[iv]
            P[row + 1, :] = h_a * grad[:, 0]
            P[row + 2, :] = h_a * grad[:, 1]
            row += 3

        for p in self._momentQuad:
            xhat = p.position
            w = float(p.weight * abs(self.detJ)) / self.area
            x_phys = self._physical_point(xhat)
            P[9:12, :] += w * numpy.outer(
                self._m1_basis_phys(x_phys),
                self.evaluateLocal(xhat),
            )
        return P

    def interpolate(self, gf):
        dofs = numpy.zeros(len(self.mapper), dtype=float)

        if not hasattr(gf, "jacobian"):
            raise NotImplementedError(
                "Hermite-style VEM interpolation needs derivatives. "
                "Provide gf.jacobian(e,x) returning the physical gradient."
            )

        for e in self.view.elements:
            geo = e.geometry
            self.bind(e)

            idx = self.mapper(e)
            local = numpy.zeros(self.localDofs, dtype=float)

            for i, xhat_v in enumerate(self._ref_vertices):
                base = 3 * i
                local[base] = float(gf(e, xhat_v))

                g_phys = numpy.asarray(gf.jacobian(e, xhat_v)).reshape(-1)
                if g_phys.size < 2:
                    raise ValueError("gf.jacobian(e,x) did not return a 2D gradient.")

                h_a = self._hV_local[i]
                local[base + 1] = h_a * float(g_phys[0])
                local[base + 2] = h_a * float(g_phys[1])

            mom = numpy.zeros(3, dtype=float)
            for p in self._momentQuad:
                xhat = p.position
                w = float(p.weight * geo.integrationElement(xhat))
                x_phys = geo.toGlobal(xhat)
                mom += w * float(gf(e, xhat)) * self._m1_basis_phys(x_phys)

            local[9:12] = mom / self.area
            dofs[idx] = local

        return dofs
