import numpy
from dune.geometry import quadratureRule

from ...base import SpaceBase
from ...common.cls_projector import solve_cls_kkt_all_rhs
from ...common.scaled_monomials import (
    P0_EXPONENTS,
    P1_EXPONENTS,
    P2_EXPONENTS,
    scaled_monomial_gradients,
    scaled_monomials,
)
from ...common.triangle_geometry import bind_affine_triangle


class QuadraticLagrangeMappedVEMSpace(SpaceBase):
    """
    k=2 scalar H1-conforming mapped VEM on triangles.

    For the Lagrange family the dof map is trivial, so the scalar basis transform is
    the identity.
    """

    def __init__(self, view):
        self.localDofs = 7
        self.polyDim = len(P2_EXPONENTS)
        self.constraintDim = len(P0_EXPONENTS)
        self.gradScalarDim = len(P1_EXPONENTS)
        self.gradPolyDim = 2 * self.gradScalarDim
        self.view = view
        self.dim = view.dimension
        self.layout = lambda gt: (1 if gt.dim == 0 else (1 if gt.dim == 1 else (1 if gt.dim == self.dim else 0)))
        self.mapper = view.mapper(self.layout)

        self.points = numpy.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
            [1.0 / 3.0, 1.0 / 3.0],
        ], dtype=float)

        self.vertices = numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
        self.x0 = numpy.array([0.0, 0.0], dtype=float)
        self.J = numpy.eye(2, dtype=float)
        self.Jinv = numpy.eye(2, dtype=float)
        self.detJ = 1.0
        self.area = 0.5

        self.xE_hat = numpy.array([1.0 / 3.0, 1.0 / 3.0], dtype=float)
        self.hE_hat = numpy.sqrt(2.0)
        self._ref_vertices = (
            numpy.array([0.0, 0.0], dtype=float),
            numpy.array([1.0, 0.0], dtype=float),
            numpy.array([0.0, 1.0], dtype=float),
        )
        self._edge_pairs = ((0, 1), (0, 2), (1, 2))

        xi = 1.0 / numpy.sqrt(3.0)
        self._edge_quad_r = 0.5 * numpy.array([1.0 - xi, 1.0 + xi], dtype=float)
        self._edge_quad_w = numpy.array([0.5, 0.5], dtype=float)
        self._edge_trace_inv = numpy.linalg.inv(numpy.array([
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.5, 1.0 / 3.0],
        ], dtype=float))

        tri_type = None
        for e in self.view.elements:
            tri_type = e.type
            break
        if tri_type is None:
            raise RuntimeError("Grid view appears to have no elements.")
        self._momentQuad = quadratureRule(tri_type, 6)

        self._constraint_rhs_selector = numpy.zeros((self.constraintDim, self.localDofs), dtype=float)
        self._constraint_rhs_selector[0, 6] = 1.0

        Ahat, Chat = self._build_reference_A_and_C()
        self._Pi0CoeffsRef = solve_cls_kkt_all_rhs(
            A=Ahat,
            C=Chat,
            G=self._constraint_rhs_selector,
        )
        self._Pi1CoeffsRef = self._build_reference_gradient_projector()

        self.bind(numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float))

    @staticmethod
    def _solve_dense_system(A, B):
        try:
            return numpy.linalg.solve(A, B)
        except numpy.linalg.LinAlgError:
            return numpy.linalg.lstsq(A, B, rcond=None)[0]

    def bind(self, element_or_vertices):
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

    def _p2_basis_ref(self, x_hat):
        return scaled_monomials(x_hat, self.xE_hat, self.hE_hat, P2_EXPONENTS)

    def _m0_basis_ref(self, x_hat):
        return scaled_monomials(x_hat, self.xE_hat, self.hE_hat, P0_EXPONENTS)

    def _p1_basis_ref(self, x_hat):
        return scaled_monomials(x_hat, self.xE_hat, self.hE_hat, P1_EXPONENTS)

    def _p1_basis_grad_ref(self, x_hat):
        return scaled_monomial_gradients(x_hat, self.xE_hat, self.hE_hat, P1_EXPONENTS)

    def _vector_p1_basis_ref(self, x_hat):
        vals = self._p1_basis_ref(x_hat)
        basis = numpy.zeros((self.gradPolyDim, 2), dtype=float)
        basis[:self.gradScalarDim, 0] = vals
        basis[self.gradScalarDim:, 1] = vals
        return basis

    def _vector_p1_div_ref(self, x_hat):
        dmx, dmy = self._p1_basis_grad_ref(x_hat)
        return numpy.concatenate((dmx, dmy))

    def _edge_geometry_from_vertices(self, verts, ia, ib, x_center):
        xa = numpy.asarray(verts[ia], dtype=float)
        xb = numpy.asarray(verts[ib], dtype=float)
        edge = xb - xa
        length = float(numpy.linalg.norm(edge))
        if length <= 1e-14:
            raise ValueError("Degenerate edge encountered while building projector.")
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
            x_hat = xa + float(r) * edge
            row += float(wr) * self._p2_basis_ref(x_hat)
        return row

    def _build_reference_A_and_C(self):
        A = numpy.zeros((self.localDofs, self.polyDim), dtype=float)
        C = numpy.zeros((self.constraintDim, self.polyDim), dtype=float)

        for i, x_hat in enumerate(self._ref_vertices):
            A[i, :] = self._p2_basis_ref(x_hat)

        for offset, (ia, ib) in enumerate(self._edge_pairs):
            A[3 + offset, :] = self._edge_average_row_ref(ia, ib)

        mom_row = numpy.zeros((1, self.polyDim), dtype=float)
        for p in self._momentQuad:
            xhat = p.position
            w = float(p.weight) / 0.5
            mom_row += w * numpy.outer(self._m0_basis_ref(xhat), self._p2_basis_ref(xhat))

        A[6:7, :] = mom_row
        C[:, :] = mom_row
        return A, C

    def _edge_trace_from_local_dofs(self, local_dofs, ia, ib, r):
        edge_index = self._edge_pairs.index((ia, ib))
        u_a = float(local_dofs[ia])
        u_b = float(local_dofs[ib])
        edge_avg = float(local_dofs[3 + edge_index])
        coeffs = self._edge_trace_inv.dot(numpy.array([u_a, u_b, edge_avg], dtype=float))
        rr = float(r)
        return coeffs[0] + coeffs[1] * rr + coeffs[2] * (rr ** 2)

    def _build_reference_gradient_projector(self):
        mass = numpy.zeros((self.gradPolyDim, self.gradPolyDim), dtype=float)
        rhs = numpy.zeros((self.gradPolyDim, self.localDofs), dtype=float)

        for p in self._momentQuad:
            xhat = p.position
            w = float(p.weight)
            vec_basis = self._vector_p1_basis_ref(xhat)
            div_basis = self._vector_p1_div_ref(xhat)
            pi0_vals = self._Pi0CoeffsRef.T.dot(self._p2_basis_ref(xhat))
            mass += w * vec_basis.dot(vec_basis.T)
            rhs -= w * numpy.outer(div_basis, pi0_vals)

        x_center = self.xE_hat
        for ia, ib in self._edge_pairs:
            xa, edge, length, _, normal = self._edge_geometry_from_vertices(self._ref_vertices, ia, ib, x_center)
            for r, wr in zip(self._edge_quad_r, self._edge_quad_w):
                xhat = xa + float(r) * edge
                flux_basis = self._vector_p1_basis_ref(xhat).dot(normal)
                for j in range(self.localDofs):
                    local = numpy.zeros(self.localDofs, dtype=float)
                    local[j] = 1.0
                    trace_val = self._edge_trace_from_local_dofs(local, ia, ib, r)
                    rhs[:, j] += length * float(wr) * flux_basis * trace_val

        return self._solve_dense_system(mass, rhs)

    def evaluateLocal(self, x):
        return self._Pi0CoeffsRef.T.dot(self._p2_basis_ref(x))

    def evaluateLocalGradient(self, x):
        grad_ref = self._Pi1CoeffsRef.T.dot(self._vector_p1_basis_ref(x))
        return grad_ref.dot(self.Jinv.T)

    def localProjectorDofs(self):
        P = numpy.zeros((self.localDofs, self.localDofs), dtype=float)
        for i, xhat_v in enumerate(self._ref_vertices):
            P[i, :] = self.evaluateLocal(xhat_v)

        verts = self._physical_vertices()
        for offset, (ia, ib) in enumerate(self._edge_pairs):
            xa, edge, length, _, _ = self._edge_geometry_from_vertices(verts, ia, ib, self.xE)
            acc = numpy.zeros(self.localDofs, dtype=float)
            for r, wr in zip(self._edge_quad_r, self._edge_quad_w):
                x_phys = xa + float(r) * edge
                x_hat = self._reference_point(x_phys)
                acc += float(wr) * self.evaluateLocal(x_hat)
            P[3 + offset, :] = acc

        for p in self._momentQuad:
            xhat = p.position
            w = float(p.weight * abs(self.detJ)) / self.area
            P[6:7, :] += w * numpy.outer(numpy.ones(1, dtype=float), self.evaluateLocal(xhat))
        return P

    def interpolate(self, gf):
        dofs = numpy.zeros(len(self.mapper), dtype=float)

        for e in self.view.elements:
            geo = e.geometry
            self.bind(e)
            idx = self.mapper(e)
            local = numpy.zeros(self.localDofs, dtype=float)
            local[:3] = numpy.asarray(gf(e, self.points[:3].T), dtype=float).reshape(-1)

            verts = self._physical_vertices()
            for offset, (ia, ib) in enumerate(self._edge_pairs):
                xa, edge, length, _, _ = self._edge_geometry_from_vertices(verts, ia, ib, self.xE)
                avg = 0.0
                for r, wr in zip(self._edge_quad_r, self._edge_quad_w):
                    x_phys = xa + float(r) * edge
                    xhat = self._reference_point(x_phys)
                    avg += float(wr) * float(gf(e, xhat))
                local[3 + offset] = avg

            cell_avg = 0.0
            for p in self._momentQuad:
                xhat = p.position
                w = float(p.weight * geo.integrationElement(xhat))
                cell_avg += w * float(gf(e, xhat))
            local[6] = cell_avg / self.area
            dofs[idx] = local

        return dofs
