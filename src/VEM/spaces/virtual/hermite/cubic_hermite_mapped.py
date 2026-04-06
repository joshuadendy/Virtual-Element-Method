import numpy
from dune.geometry import quadratureRule

from ...base import SpaceBase
from ...common.cls_projector import solve_cls_kkt_all_rhs
from ...common.hermite_mapping import build_k3_mapped_transform
from ...common.scaled_monomials import (
    P1_EXPONENTS,
    P2_EXPONENTS,
    P3_EXPONENTS,
    scaled_monomial_gradients,
    scaled_monomials,
)
from ...common.triangle_geometry import bind_affine_triangle
from ...common.vertex_scaling import build_vertex_effective_h


class CubicHermiteMappedVEMSpace(SpaceBase):
    """
    k=3 Hermite-style VEM space with separate value and gradient projectors.

    The value projector Pi_0 is assembled once on the reference triangle. The
    gradient projector Pi_1 is also assembled once on the reference triangle via
    the variational definition, then mapped onto each physical element using

        Pi^E_1 = F_curl( F^{-*}( Pi^{Ê}_1 ) )

    In code, the final mapped gradient basis values are formed by:
      1. evaluating the reference projected gradient basis,
      2. applying the Hermite dof/basis transform M,
      3. applying the Jacobian factor on the vector side.
    """

    def __init__(self, view):
        self.view = view
        self.dim = view.dimension
        self.localDofs = 12
        self.polyDim = 10
        self.constraintDim = 3
        self.gradScalarDim = len(P2_EXPONENTS)
        self.gradPolyDim = 2 * self.gradScalarDim

        self.layout = lambda gt: (3 if gt.dim == 0 else (3 if gt.dim == self.dim else 0))
        self.mapper = view.mapper(self.layout)
        self.vertices = numpy.array([[0, 0], [1, 0], [0, 1]], dtype=float)

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
        self._hV_hat = numpy.array([self.hE_hat, self.hE_hat, self.hE_hat], dtype=float)

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

        # Reference CLS value projector Pi_0 on the reference triangle.
        Ahat, Chat = self._build_reference_A_and_C()
        self._Pi0CoeffsRef = solve_cls_kkt_all_rhs(
            A=Ahat,
            C=Chat,
            G=self._constraint_rhs_selector,
        )

        # Reference gradient projector Pi_1 on the reference triangle.
        self._Pi1CoeffsRef = self._build_reference_gradient_projector()

        # Elementwise scalar Hermite basis transform.
        self.M = numpy.eye(self.localDofs, dtype=float)

        # Use the vertex scale from the paper rather than diameter.
        self._vertex_h = build_vertex_effective_h(
            self.view,
            self.mapper,
            measure="adjacent_edge_average",
        )
        self._hV_local = self._hV_hat.copy()

        self.bind(numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float))

    @staticmethod
    def _build_unit_interval_quadrature(order=6):
        pts, wts = numpy.polynomial.legendre.leggauss(order)
        r = 0.5 * (pts + 1.0)
        w = 0.5 * wts
        return r, w

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

        self.M = build_k3_mapped_transform(
            J=self.J,
            Jinv=self.Jinv,
            hV_hat=self._hV_hat,
            hV_local=self._hV_local,
            hE=self.hE,
            hE_hat=self.hE_hat,
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

    # -------------------------------------------------------------------------
    # Scalar polynomial bases
    # -------------------------------------------------------------------------

    def _p3_basis_ref(self, x_hat):
        return scaled_monomials(x_hat, self.xE_hat, self.hE_hat, P3_EXPONENTS)

    def _m1_basis_ref(self, x_hat):
        return scaled_monomials(x_hat, self.xE_hat, self.hE_hat, P1_EXPONENTS)

    def _p2_basis_ref(self, x_hat):
        return scaled_monomials(x_hat, self.xE_hat, self.hE_hat, P2_EXPONENTS)

    def _p2_basis_grad_ref(self, x_hat):
        return scaled_monomial_gradients(x_hat, self.xE_hat, self.hE_hat, P2_EXPONENTS)

    def _m1_basis_phys(self, x_phys):
        return scaled_monomials(x_phys, self.xE, self.hE, P1_EXPONENTS)

    # -------------------------------------------------------------------------
    # Vector P2 basis on the reference triangle for Pi_1
    # -------------------------------------------------------------------------

    def _vector_p2_basis_ref(self, x_hat):
        vals = self._p2_basis_ref(x_hat)
        basis = numpy.zeros((self.gradPolyDim, 2), dtype=float)
        basis[:self.gradScalarDim, 0] = vals
        basis[self.gradScalarDim:, 1] = vals
        return basis

    def _vector_p2_div_ref(self, x_hat):
        dmx, dmy = self._p2_basis_grad_ref(x_hat)
        return numpy.concatenate((dmx, dmy))

    # -------------------------------------------------------------------------
    # Reference Pi_0
    # -------------------------------------------------------------------------

    def _build_reference_A_and_C(self):
        A = numpy.zeros((self.localDofs, self.polyDim), dtype=float)
        C = numpy.zeros((self.constraintDim, self.polyDim), dtype=float)

        row = 0
        for iv, xhat_v in enumerate(self._ref_vertices):
            A[row, :] = self._p3_basis_ref(xhat_v)

            dxi, deta = scaled_monomial_gradients(
                xhat_v, self.xE_hat, self.hE_hat, P3_EXPONENTS
            )
            grad_hat = numpy.column_stack((dxi, deta))

            hhat_a = self._hV_hat[iv]
            A[row + 1, :] = hhat_a * grad_hat[:, 0]
            A[row + 2, :] = hhat_a * grad_hat[:, 1]
            row += 3

        mom_rows = numpy.zeros((3, self.polyDim), dtype=float)
        for p in self._momentQuad:
            xhat = p.position
            mom_rows += 2.0 * float(p.weight) * numpy.outer(
                self._m1_basis_ref(xhat),
                self._p3_basis_ref(xhat),
            )

        A[9:12, :] = mom_rows
        C[:, :] = mom_rows
        return A, C

    # -------------------------------------------------------------------------
    # Edge helpers used in the reference Pi_1 assembly
    # -------------------------------------------------------------------------

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

    def _edge_trace_from_local_dofs(self, local_dofs, verts, ia, ib, r, hV):
        _, _, length, tangent, _ = self._edge_geometry_from_vertices(
            verts, ia, ib, numpy.mean(numpy.asarray(verts), axis=0)
        )

        base_a = 3 * ia
        base_b = 3 * ib

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

        rr = float(r)
        h00 = 2.0 * rr**3 - 3.0 * rr**2 + 1.0
        h10 = rr**3 - 2.0 * rr**2 + rr
        h01 = -2.0 * rr**3 + 3.0 * rr**2
        h11 = rr**3 - rr**2

        return u_a * h00 + m_a * h10 + u_b * h01 + m_b * h11

    # -------------------------------------------------------------------------
    # Reference Pi_1
    # -------------------------------------------------------------------------

    def _build_reference_gradient_projector(self):
        """
        Assemble Pi_1 on the reference element only.
        """
        mass = numpy.zeros((self.gradPolyDim, self.gradPolyDim), dtype=float)
        rhs = numpy.zeros((self.gradPolyDim, self.localDofs), dtype=float)

        verts = self._ref_vertices
        x_center = self.xE_hat

        # Volume term:
        #   ∫_Ê Pi_1[φ_j] · q̂
        # = -∫_Ê Pi_0[φ_j] div q̂ + Σ_edges ∫_ê Pi^ê_0[φ_j] (n̂·q̂)
        for p in self._momentQuad:
            xhat = p.position
            w = float(p.weight)

            vec_basis = self._vector_p2_basis_ref(xhat)
            div_basis = self._vector_p2_div_ref(xhat)
            pi0_vals = self._Pi0CoeffsRef.T.dot(self._p3_basis_ref(xhat))

            mass += w * vec_basis.dot(vec_basis.T)
            rhs -= w * numpy.outer(div_basis, pi0_vals)

        # Edge term
        for ia, ib in self._edge_pairs:
            xa, edge, length, _, normal = self._edge_geometry_from_vertices(
                verts, ia, ib, x_center
            )

            for r, wr in zip(self._edge_quad_r, self._edge_quad_w):
                xhat = xa + float(r) * edge
                flux_basis = self._vector_p2_basis_ref(xhat).dot(normal)

                for j in range(self.localDofs):
                    local = numpy.zeros(self.localDofs, dtype=float)
                    local[j] = 1.0
                    trace_val = self._edge_trace_from_local_dofs(
                        local_dofs=local,
                        verts=verts,
                        ia=ia,
                        ib=ib,
                        r=r,
                        hV=self._hV_hat,
                    )
                    rhs[:, j] += length * float(wr) * flux_basis * trace_val

        return self._solve_dense_system(mass, rhs)

    # -------------------------------------------------------------------------
    # Public evaluation
    # -------------------------------------------------------------------------

    def evaluateLocal(self, x):
        phi_ref_proj = self._Pi0CoeffsRef.T.dot(self._p3_basis_ref(x))
        return self.M.dot(phi_ref_proj)

    def evaluateLocalGradient(self, x):
        """
        Map the reference gradient projector onto the physical element.
        """
        grad_ref_proj = self._Pi1CoeffsRef.T.dot(self._vector_p2_basis_ref(x))
        return self.M.dot(grad_ref_proj).dot(self.Jinv.T)

    def evaluatePhysical(self, x_phys):
        return self.evaluateLocal(self._reference_point(x_phys))

    # -------------------------------------------------------------------------
    # Projector-on-dofs matrix used by stabilization
    # -------------------------------------------------------------------------

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

        for p in self._momentQuad:
            xhat = p.position
            w = float(p.weight * abs(self.detJ)) / self.area
            x_phys = self._physical_point(xhat)
            P[9:12, :] += w * numpy.outer(
                self._m1_basis_phys(x_phys),
                self.evaluateLocal(xhat),
            )
        return P

    # -------------------------------------------------------------------------
    # Interpolation into the physical mapped Hermite VEM dofs
    # -------------------------------------------------------------------------

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