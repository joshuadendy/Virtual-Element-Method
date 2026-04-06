import numpy
from dune.geometry import quadratureRule
from ...base import SpaceBase
from ...common.cls_projector import solve_cls_kkt_all_rhs

class CubicHermitePhysicalVEMSpace(SpaceBase):
    """
    k=3 Hermite-style VEM value-projection space (physical CLS on every element).

    Local DOFs (12), ordered as:
      0: u(v0)
      1: du/dx(v0)
      2: du/dy(v0)
      3: u(v1)
      4: du/dx(v1)
      5: du/dy(v1)
      6: u(v2)
      7: du/dx(v2)
      8: du/dy(v2)
      9:  (1/|E|) int_E u q0
      10: (1/|E|) int_E u q1
      11: (1/|E|) int_E u q2


    where {q0,q1,q2} is a basis of M_1(E), chosen as
    [1, (x-xE)_x/hE, (x-xE)_y/hE].

    Value projection basis B0 = P3(E), represented with scaled monomials
    centered at xE. Constraints C0 are the interior moments against M_1(E)
    (3 constraints).
    """

    def __init__(self, view):
        self.view = view
        self.dim = view.dimension
        self.localDofs = 12
        self.polyDim = 10
        self.constraintDim = 3
        self.vertices = numpy.array( [ [0,0],[1,0],[0,1] ], dtype=float )

        # 3 DOFs per vertex, 3 DOFs per cell (interior moments)
        self.layout = lambda gt: (3 if gt.dim == 0 else
                                  (3 if gt.dim == self.dim else 0))
        self.mapper = view.mapper(self.layout)

        # "Points" are only meaningful for vertex-based DOFs; for moment DOFs, placeholders are used.
        self.points = numpy.array([
            [0.0, 0.0],      # v0 value
            [0.0, 0.0],      # v0 dx
            [0.0, 0.0],      # v0 dy
            [1.0, 0.0],      # v1 value
            [1.0, 0.0],      # v1 dx
            [1.0, 0.0],      # v1 dy
            [0.0, 1.0],      # v2 value
            [0.0, 1.0],      # v2 dx
            [0.0, 1.0],      # v2 dy
            [1.0/3.0, 1.0/3.0],  # moment q0 placeholder
            [1.0/3.0, 1.0/3.0],  # moment q1 placeholder
            [1.0/3.0, 1.0/3.0],  # moment q2 placeholder
        ], dtype=float)

        self.x0 = numpy.array([0.0, 0.0], dtype=float)
        self.J = numpy.eye(2, dtype=float)
        self.Jinv = numpy.eye(2, dtype=float)
        self.detJ = 1.0
        self.area = 0.5

        self.xE = numpy.array([1.0/3.0, 1.0/3.0], dtype=float)
        self.hE = numpy.sqrt(2.0)

        # Reference geometry data (used for mapping block consistency /
        # interpolation helpers)
        self.xE_hat = numpy.array([1.0/3.0, 1.0/3.0], dtype=float)
        self.hE_hat = numpy.sqrt(2.0)
        self.area_hat = 0.5

        # P3 scaled monomial exponents (10 monomials)
        self._exponents = [
            (0, 0),
            (1, 0), (0, 1),
            (2, 0), (1, 1), (0, 2),
            (3, 0), (2, 1), (1, 2), (0, 3),
        ]

        # Triangle type and quadrature
        self._triType = None
        for e in self.view.elements:
            self._triType = e.type
            break
        if self._triType is None:
            raise RuntimeError("Grid view appears to have no elements.")

        self._momentQuad = quadratureRule(self._triType, 6)

        # Constraint RHS selector for canonical local basis functions
        self._constraint_rhs_selector = numpy.zeros((self.constraintDim, self.localDofs), dtype=float)
        self._constraint_rhs_selector[0, 9]  = 1.0
        self._constraint_rhs_selector[1, 10] = 1.0
        self._constraint_rhs_selector[2, 11] = 1.0

        # Hermite-style basis transform matrix (kept for consistency /
        # diagnostics)
        self.M = numpy.eye(self.localDofs, dtype=float)

        # Storage for projected basis coefficients on current physical element
        self._Pi0Coeffs = numpy.zeros((self.polyDim, self.localDofs),
                                      dtype=float)

        # Set up scaling constants for derivative dofs (characteristic lengths
        # associated with each vertex)
        self._build_vertex_effective_h()

        # Initialize on reference triangle
        self.bind(numpy.array([[0., 0.], [1., 0.], [0., 1.]], dtype=float))

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def bind(self, element_or_vertices):
        if hasattr(element_or_vertices, "geometry"):
            e = element_or_vertices
            geo = e.geometry
            verts = numpy.array([geo.corner(i) for i in range(3)], dtype=float)
            idx = self.mapper(e)
            self._hV_local = numpy.array([
                self._vertex_h[int(idx[0])],
                self._vertex_h[int(idx[3])],
                self._vertex_h[int(idx[6])]
            ], dtype=float)
        else:
            verts = numpy.asarray(element_or_vertices, dtype=float)
            # fallback for reference bind / legacy calls
            # choose something sane; for reference cell use hE_hat
            self._hV_local = numpy.array([self.hE_hat, self.hE_hat, self.hE_hat], dtype=float)

        self.vertices[0] = verts[0]
        self.vertices[1] = verts[1] - verts[0]
        self.vertices[2] = verts[2] - verts[0]

        self.x0 = self.vertices[0].copy()
        self.J = numpy.column_stack((self.vertices[1], self.vertices[2]))
        self.detJ = float(numpy.linalg.det(self.J))
        if abs(self.detJ) <= 1e-14:
            raise ValueError("Degenerate element (detJ=0)")
        self.Jinv = numpy.linalg.inv(self.J)
        self.area = 0.5 * abs(self.detJ)

        # Barycentre and diameter for physical scaled monomials
        v0 = self.x0
        v1 = self.x0 + self.vertices[1]
        v2 = self.x0 + self.vertices[2]
        verts_phys = numpy.array([v0, v1, v2])
        self.xE = verts_phys.mean(axis=0)

        e01 = numpy.linalg.norm(verts_phys[1] - verts_phys[0])
        e12 = numpy.linalg.norm(verts_phys[2] - verts_phys[1])
        e20 = numpy.linalg.norm(verts_phys[0] - verts_phys[2])
        self.hE = max(e01, e12, e20)
        if self.hE <= 0:
            raise ValueError("Degenerate element with non-positive diameter")

        # Physical CLS solve on every element
        A, C = self._build_physical_A_and_C()
        self._Pi0Coeffs = solve_cls_kkt_all_rhs(
            A=A, C=C, G=self._constraint_rhs_selector
        )

    # ------------------------------------------------------------------
    # Affine maps
    # ------------------------------------------------------------------
    def _physical_point(self, xhat):
        xhat = numpy.asarray(xhat, dtype=float)
        return self.x0 + self.J.dot(xhat)

    def _reference_point(self, xphys):
        xphys = numpy.asarray(xphys, dtype=float)
        return self.Jinv.dot(xphys - self.x0)

    # ------------------------------------------------------------------
    # Scaled monomial bases
    # ------------------------------------------------------------------
    def _scaled_coords_phys(self, x_phys):
        y = (numpy.asarray(x_phys, dtype=float) - self.xE) / self.hE
        return float(y[0]), float(y[1])

    def _scaled_coords_ref(self, x_hat):
        y = (numpy.asarray(x_hat, dtype=float) - self.xE_hat) / self.hE_hat
        return float(y[0]), float(y[1])

    def _p3_basis_phys(self, x_phys):
        sx, sy = self._scaled_coords_phys(x_phys)
        vals = []
        for a, b in self._exponents:
            vals.append((sx ** a) * (sy ** b))
        return numpy.array(vals, dtype=float)

    def _p3_basis_ref(self, x_hat):
        sx, sy = self._scaled_coords_ref(x_hat)
        vals = []
        for a, b in self._exponents:
            vals.append((sx ** a) * (sy ** b))
        return numpy.array(vals, dtype=float)

    def _p3_basis_grad_phys(self, x_phys):
        """
        Returns (d/dx basis, d/dy basis) for P3 scaled monomials on the current
        physical element.
        """
        sx, sy = self._scaled_coords_phys(x_phys)
        dx = numpy.zeros(self.polyDim, dtype=float)
        dy = numpy.zeros(self.polyDim, dtype=float)
        invh = 1.0 / self.hE

        for i, (a, b) in enumerate(self._exponents):
            if a > 0:
                dx[i] = a * (sx ** (a-1)) * (sy ** b) * invh
            if b > 0:
                dy[i] = b * (sx ** a) * (sy ** (b-1)) * invh
        return dx, dy

    def _p3_basis_grad_ref(self, x_hat):
        """
        Returns (d/dxi basis, d/deta basis) for P3 scaled monomials on the
        reference element.
        """
        sx, sy = self._scaled_coords_ref(x_hat)
        dxi = numpy.zeros(self.polyDim, dtype=float)
        deta = numpy.zeros(self.polyDim, dtype=float)
        invh = 1.0 / self.hE_hat

        for i, (a, b) in enumerate(self._exponents):
            if a > 0:
                dxi[i] = a * (sx ** (a-1)) * (sy ** b) * invh
            if b > 0:
                deta[i] = b * (sx ** a) * (sy ** (b-1)) * invh
        return dxi, deta

    def _m1_basis_phys(self, x_phys):
        sx, sy = self._scaled_coords_phys(x_phys)
        return numpy.array([1.0, sx, sy], dtype=float)

    def _m1_basis_ref(self, x_hat):
        sx, sy = self._scaled_coords_ref(x_hat)
        return numpy.array([1.0, sx, sy], dtype=float)

    # ------------------------------------------------------------------
    # Physical CLS matrices
    # ------------------------------------------------------------------
    def _build_physical_A_and_C(self):
        """
        Build:
          A[i,alpha] = lambda_i(m_alpha),  i=0..11, alpha=0..9
          C[r,alpha] = c_r(m_alpha),       r=0..2
        on the current physical element.
        """
        A = numpy.zeros((self.localDofs, self.polyDim), dtype=float)
        C = numpy.zeros((self.constraintDim, self.polyDim), dtype=float)

        # Physical vertices
        v0 = self.x0
        v1 = self.x0 + self.vertices[1]
        v2 = self.x0 + self.vertices[2]
        verts = [v0, v1, v2]

        # Vertex value + gradient DOFs
        row = 0
        for iv, xv in enumerate(verts):
            A[row, :] = self._p3_basis_phys(xv)
            dmx, dmy = self._p3_basis_grad_phys(xv)
            h_a = self._hV_local[iv]
            A[row+1, :] = h_a * dmx
            A[row+2, :] = h_a * dmy
            row += 3

        # Interior moments and constraints (same functionals for k=3)
        mom_rows = numpy.zeros((3, self.polyDim), dtype=float)
        for p in self._momentQuad:
            xhat = p.position
            wref = p.weight
            x_phys = self._physical_point(xhat)
            pvals = self._p3_basis_phys(x_phys)
            qvals = self._m1_basis_phys(x_phys)
            # (1/|E|)\int_E = 1/A * \sum |detJ| w_ref (...)
            # (scale reference weights by |detJ| to obtain physical weights)
            mom_rows += 1/self.area * abs(self.detJ) * wref * numpy.outer(qvals, pvals)

        A[9:12, :] = mom_rows
        C[:, :] = mom_rows
        return A, C

    # ------------------------------------------------------------------
    # Characteristic length associated to vertices
    # ------------------------------------------------------------------
    def _element_diameter_from_vertices(self, verts):
        e01 = numpy.linalg.norm(verts[1] - verts[0])
        e12 = numpy.linalg.norm(verts[2] - verts[1])
        e20 = numpy.linalg.norm(verts[0] - verts[2])
        hE = max(e01, e12, e20)
        if hE <= 0:
            raise ValueError("Degenerate element with non-positive diameter")
        return hE

    def _build_vertex_effective_h(self):
        sums = {}
        counts = {}

        for e in self.view.elements:
            geo = e.geometry
            verts = numpy.array([geo.corner(i) for i in range(3)], dtype=float)
            hE = self._element_diameter_from_vertices(verts)
            idx = self.mapper(e)   # length 12, local ordering fixed

            # value dof indices identify the vertex blocks globally
            for base in (0, 3, 6):
                vid = int(idx[base])   # use value-dof global index as vertex key
                sums[vid] = sums.get(vid, 0.0) + hE
                counts[vid] = counts.get(vid, 0) + 1

        self._vertex_h = {vid: sums[vid] / counts[vid] for vid in sums}

    # ------------------------------------------------------------------
    # Projected basis evaluation
    # ------------------------------------------------------------------
    def evaluateLocal(self, x):
        """
        Evaluate [Pi0^E phi_j](xhat) for all local basis functions j=0..11 at
        reference point xhat. Returns shape (12,).
        """
        x_phys = self._physical_point(x)
        m = self._p3_basis_phys(x_phys)
        return self._Pi0Coeffs.T.dot(m)

    def evaluatePhysical(self, x_phys):
        return self.evaluateLocal(self._reference_point(x_phys))

    # ------------------------------------------------------------------
    # Interpolation into the 12 Hermite-style DOFs
    # ------------------------------------------------------------------
    def interpolate(self, gf):
        """
        Interpolate a scalar grid function into the Hermite-style VEM k=3 DOFs:
          - vertex values
          - vertex physical gradients (ux, uy)
          - interior moments against M1(E)

        Requires gf.jacobian(e,x) for derivative DOFs.
        """
        dofs = numpy.zeros(len(self.mapper), dtype=float)

        if not hasattr(gf, "jacobian"):
            raise NotImplementedError(
                "Hermite-style VEM interpolation needs derivatives. "
                "Provide gf.jacobian(e,x) returning the physical gradient."
            )

        ref_vertices = [numpy.array([0.0, 0.0]),
                        numpy.array([1.0, 0.0]),
                        numpy.array([0.0, 1.0])]

        for e in self.view.elements:
            geo = e.geometry
            self.bind(e)

            idx = self.mapper(e)
            local = numpy.zeros(self.localDofs, dtype=float)

            # Vertex value + gradient DOFs
            for i, xhat_v in enumerate(ref_vertices):
                base = 3 * i
                local[base] = float(gf(e, xhat_v))

                g_phys = numpy.asarray(gf.jacobian(e, xhat_v)).reshape(-1)
                if g_phys.size < 2:
                    raise ValueError("gf.jacobian(e,x) did not return a 2D " \
                    "gradient.")

                h_a = self._hV_local[i]
                local[base+1] = h_a * float(g_phys[0])   # scaled du/dx at vertex i
                local[base+2] = h_a * float(g_phys[1])   # scaled du/dy at vertex i

            # Interior moments (1/|E|) \int_E u q_i
            mom = numpy.zeros(3, dtype=float)
            for p in self._momentQuad:
                xhat = p.position
                w = p.weight * geo.integrationElement(xhat)
                x_phys = geo.toGlobal(xhat)
                q = self._m1_basis_phys(x_phys)
                mom += w * float(gf(e, xhat)) * q

            mom /= self.area
            local[9:12] = mom

            dofs[idx] = local

        return dofs
