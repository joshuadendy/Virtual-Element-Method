import numpy
from dune.geometry import quadratureRule

from ...base import SpaceBase
from ...common.cls_projector import solve_cls_kkt_all_rhs
from ...common.hermite_mapping import build_k4_mapped_transform
from ...common.scaled_monomials import (
    P2_EXPONENTS,
    P3_EXPONENTS,
    P4_EXPONENTS,
    scaled_monomial_gradients,
    scaled_monomials,
    scaled_monomial_inverse_pullback_matrix,
)
from ...common.triangle_geometry import bind_affine_triangle
from ...common.vertex_scaling import build_vertex_effective_h


class QuarticHermiteMappedVEMSpace(SpaceBase):
    """
    k=4 mapped Hermite-style VEM on triangles.

    The value projector is assembled once on the reference triangle and evaluated on
    each element via the Section 4.2.3 surrogate construction. The gradient projector
    is assembled once on the reference triangle and mapped with the covariant Piola
    factor J^{-T}, independently of Pi_0.
    """

    def __init__(self, view):
        self.view = view
        self.dim = view.dimension
        self.localDofs = 18
        self.polyDim = len(P4_EXPONENTS)
        self.constraintDim = len(P2_EXPONENTS)
        self.gradScalarDim = len(P3_EXPONENTS)
        self.gradPolyDim = 2 * self.gradScalarDim

        self.layout = lambda gt: (3 if gt.dim == 0 else (1 if gt.dim == 1 else (6 if gt.dim == self.dim else 0)))
        self.mapper = view.mapper(self.layout)
        self.vertices = numpy.array([[0, 0], [1, 0], [0, 1]], dtype=float)

        self.points = numpy.array([
            [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
            [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
            [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
            [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
            [1.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 1.0 / 3.0],
            [1.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 1.0 / 3.0],
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
        self._hV_hat = self._reference_vertex_h()
        self._ref_vertices = (
            numpy.array([0.0, 0.0], dtype=float),
            numpy.array([1.0, 0.0], dtype=float),
            numpy.array([0.0, 1.0], dtype=float),
        )
        self._edge_pairs = ((0, 1), (0, 2), (1, 2))
        self._edge_quad_r, self._edge_quad_w = self._build_unit_interval_quadrature(order=8)
        self._quartic_edge_inv = numpy.linalg.inv(numpy.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [1.0, 0.5, 1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0],
        ], dtype=float))

        tri_type = None
        for e in self.view.elements:
            tri_type = e.type
            break
        if tri_type is None:
            raise RuntimeError("Grid view appears to have no elements.")
        self._momentQuad = quadratureRule(tri_type, 10)

        self._constraint_rhs_selector = numpy.zeros((self.constraintDim, self.localDofs), dtype=float)
        for i in range(self.constraintDim):
            self._constraint_rhs_selector[i, 12 + i] = 1.0

        Ahat, Chat = self._build_reference_A_and_C()
        self._Pi0CoeffsRef = solve_cls_kkt_all_rhs(
            A=Ahat,
            C=Chat,
            G=self._constraint_rhs_selector,
        )
        self._Pi1CoeffsRef = self._build_reference_gradient_projector()

        self.M = numpy.eye(self.localDofs, dtype=float)
        self._vertex_h = build_vertex_effective_h(
            self.view,
            self.mapper,
            measure="adjacent_edge_average",
        )
        self._hV_local = self._hV_hat.copy()
        self._constraint_pullback = numpy.eye(self.constraintDim, dtype=float)

        self.bind(numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float))

    @staticmethod
    def _build_unit_interval_quadrature(order=8):
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
            self._hV_local = self._hV_hat.copy()

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

        self._constraint_pullback = scaled_monomial_inverse_pullback_matrix(
            Jinv=self.Jinv,
            h=self.hE,
            h_hat=self.hE_hat,
            exponents=P2_EXPONENTS,
        )

        self.M = build_k4_mapped_transform(
            J=self.J,
            hV_hat=self._hV_hat,
            hV_local=self._hV_local,
        )

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

    def _p4_basis_ref(self, x_hat):
        return scaled_monomials(x_hat, self.xE_hat, self.hE_hat, P4_EXPONENTS)

    def _m2_basis_ref(self, x_hat):
        return scaled_monomials(x_hat, self.xE_hat, self.hE_hat, P2_EXPONENTS)

    def _p3_basis_ref(self, x_hat):
        return scaled_monomials(x_hat, self.xE_hat, self.hE_hat, P3_EXPONENTS)

    def _p3_basis_grad_ref(self, x_hat):
        return scaled_monomial_gradients(x_hat, self.xE_hat, self.hE_hat, P3_EXPONENTS)

    def _m2_basis_phys(self, x_phys):
        return scaled_monomials(x_phys, self.xE, self.hE, P2_EXPONENTS)

    def _m2_basis_mapped_phys(self, x_phys):
        return self._constraint_pullback.dot(self._m2_basis_phys(x_phys))

    def _vector_p3_basis_ref(self, x_hat):
        vals = self._p3_basis_ref(x_hat)
        basis = numpy.zeros((self.gradPolyDim, 2), dtype=float)
        basis[:self.gradScalarDim, 0] = vals
        basis[self.gradScalarDim:, 1] = vals
        return basis

    def _vector_p3_div_ref(self, x_hat):
        dmx, dmy = self._p3_basis_grad_ref(x_hat)
        return numpy.concatenate((dmx, dmy))

    def _edge_geometry_from_vertices(self, verts, ia, ib, x_center):
        xa = numpy.asarray(verts[ia], dtype=float)
        xb = numpy.asarray(verts[ib], dtype=float)
        edge = xb - xa
        length = float(numpy.linalg.norm(edge))
        if length <= 1e-14:
            raise ValueError("Degenerate edge encountered while building Hermite projector.")
        tangent = edge / length
        normal = numpy.array([tangent[1], -tangent[0]], dtype=float)
        midpoint = 0.5 * (xa + xb)
        if numpy.dot(normal, x_center - midpoint) > 0.0:
            normal *= -1.0
        return xa, edge, length, tangent, normal

    def _edge_average_row_ref(self, ia, ib):
        row = numpy.zeros(self.polyDim, dtype=float)
        xa = numpy.asarray(self._ref_vertices[ia], dtype=float)
        xb = numpy.asarray(self._ref_vertices[ib], dtype=float)
        edge = xb - xa
        for r, wr in zip(self._edge_quad_r, self._edge_quad_w):
            xhat = xa + float(r) * edge
            row += float(wr) * self._p4_basis_ref(xhat)
        return row

    def _build_reference_A_and_C(self):
        A = numpy.zeros((self.localDofs, self.polyDim), dtype=float)
        C = numpy.zeros((self.constraintDim, self.polyDim), dtype=float)

        row = 0
        for iv, xhat_v in enumerate(self._ref_vertices):
            A[row, :] = self._p4_basis_ref(xhat_v)
            dmx, dmy = scaled_monomial_gradients(xhat_v, self.xE_hat, self.hE_hat, P4_EXPONENTS)
            grad = numpy.column_stack((dmx, dmy))
            hhat_a = self._hV_hat[iv]
            A[row + 1, :] = hhat_a * grad[:, 0]
            A[row + 2, :] = hhat_a * grad[:, 1]
            row += 3

        for offset, (ia, ib) in enumerate(self._edge_pairs):
            A[9 + offset, :] = self._edge_average_row_ref(ia, ib)

        mom_rows = numpy.zeros((self.constraintDim, self.polyDim), dtype=float)
        for p in self._momentQuad:
            xhat = p.position
            w = float(p.weight) / 0.5
            mom_rows += w * numpy.outer(self._m2_basis_ref(xhat), self._p4_basis_ref(xhat))

        A[12:18, :] = mom_rows
        C[:, :] = mom_rows
        return A, C

    def _edge_trace_from_local_dofs(self, local_dofs, verts, ia, ib, r, hV):
        base_a = 3 * ia
        base_b = 3 * ib
        edge_index = self._edge_pairs.index((ia, ib))

        _, _, length, tangent, _ = self._edge_geometry_from_vertices(verts, ia, ib, numpy.mean(numpy.asarray(verts), axis=0))

        u_a = float(local_dofs[base_a])
        u_b = float(local_dofs[base_b])
        grad_a = numpy.array([
            float(local_dofs[base_a + 1]) / float(hV[ia]),
            float(local_dofs[base_a + 2]) / float(hV[ia]),
        ], dtype=float)
        grad_b = numpy.array([
            float(local_dofs[base_b + 1]) / float(hV[ib]),
            float(local_dofs[base_b + 2]) / float(hV[ib]),
        ], dtype=float)
        m_a = length * float(numpy.dot(tangent, grad_a))
        m_b = length * float(numpy.dot(tangent, grad_b))
        edge_avg = float(local_dofs[9 + edge_index])

        coeffs = self._quartic_edge_inv.dot(numpy.array([u_a, m_a, u_b, m_b, edge_avg], dtype=float))
        rr = float(r)
        powers = numpy.array([1.0, rr, rr ** 2, rr ** 3, rr ** 4], dtype=float)
        return float(coeffs.dot(powers))

    def _build_reference_gradient_projector(self):
        mass = numpy.zeros((self.gradPolyDim, self.gradPolyDim), dtype=float)
        rhs = numpy.zeros((self.gradPolyDim, self.localDofs), dtype=float)

        for p in self._momentQuad:
            xhat = p.position
            w = float(p.weight)
            vec_basis = self._vector_p3_basis_ref(xhat)
            div_basis = self._vector_p3_div_ref(xhat)
            pi0_vals = self._Pi0CoeffsRef.T.dot(self._p4_basis_ref(xhat))
            mass += w * vec_basis.dot(vec_basis.T)
            rhs -= w * numpy.outer(div_basis, pi0_vals)

        x_center = self.xE_hat
        for ia, ib in self._edge_pairs:
            xa, edge, length, _, normal = self._edge_geometry_from_vertices(self._ref_vertices, ia, ib, x_center)
            for r, wr in zip(self._edge_quad_r, self._edge_quad_w):
                xhat = xa + float(r) * edge
                flux_basis = self._vector_p3_basis_ref(xhat).dot(normal)
                for j in range(self.localDofs):
                    local = numpy.zeros(self.localDofs, dtype=float)
                    local[j] = 1.0
                    trace_val = self._edge_trace_from_local_dofs(local, self._ref_vertices, ia, ib, r, self._hV_hat)
                    rhs[:, j] += length * float(wr) * flux_basis * trace_val

        return self._solve_dense_system(mass, rhs)

    def evaluateLocal(self, x):
        phi_ref_proj = self._Pi0CoeffsRef.T.dot(self._p4_basis_ref(x))
        return self.M.dot(phi_ref_proj)

    def evaluateLocalGradient(self, x):
        grad_ref_proj = self._Pi1CoeffsRef.T.dot(self._vector_p3_basis_ref(x))
        return self.M.dot(grad_ref_proj).dot(self.Jinv.T)

    def evaluatePhysical(self, x_phys):
        return self.evaluateLocal(self._reference_point(x_phys))

    def localProjectorDofs(self):
        P = numpy.zeros((self.localDofs, self.localDofs), dtype=float)

        row = 0
        for iv, xhat_v in enumerate(self._ref_vertices):
            P[row, :] = self.evaluateLocal(xhat_v)
            grad = self.evaluateLocalGradient(xhat_v)
            h_a = self._hV_local[iv]
            P[row + 1, :] = h_a * grad[:, 0]
            P[row + 2, :] = h_a * grad[:, 1]
            row += 3

        verts = self._physical_vertices()
        for offset, (ia, ib) in enumerate(self._edge_pairs):
            xa, edge, length, _, _ = self._edge_geometry_from_vertices(verts, ia, ib, self.xE)
            acc = numpy.zeros(self.localDofs, dtype=float)
            for r, wr in zip(self._edge_quad_r, self._edge_quad_w):
                x_phys = xa + float(r) * edge
                xhat = self._reference_point(x_phys)
                acc += float(wr) * self.evaluateLocal(xhat)
            P[9 + offset, :] = acc

        for p in self._momentQuad:
            xhat = p.position
            w = float(p.weight * abs(self.detJ)) / self.area
            x_phys = self._physical_point(xhat)
            P[12:18, :] += w * numpy.outer(
                self._m2_basis_mapped_phys(x_phys),
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

            verts = self._physical_vertices()
            for offset, (ia, ib) in enumerate(self._edge_pairs):
                xa, edge, length, _, _ = self._edge_geometry_from_vertices(verts, ia, ib, self.xE)
                avg = 0.0
                for r, wr in zip(self._edge_quad_r, self._edge_quad_w):
                    x_phys = xa + float(r) * edge
                    xhat = self._reference_point(x_phys)
                    avg += float(wr) * float(gf(e, xhat))
                local[9 + offset] = avg

            mom = numpy.zeros(self.constraintDim, dtype=float)
            for p in self._momentQuad:
                xhat = p.position
                w = float(p.weight * geo.integrationElement(xhat))
                x_phys = geo.toGlobal(xhat)
                mom += w * float(gf(e, xhat)) * self._m2_basis_mapped_phys(x_phys)

            local[12:18] = mom / self.area
            dofs[idx] = local

        return dofs
