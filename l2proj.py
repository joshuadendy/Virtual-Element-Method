# # A projection into a finite element space
# The following requires assembly a finite element
# matrix (the mass matrix) and a right hand side.
# We use linear Lagrange shape functions.
#
# So we are looking for the $L^2$ projection
# \begin{align*}
#    u_h(x) = \sum_k u_k\varphi_k(x)
# \end{align*}
# which is the solution of
# \begin{align*}
#   \int_\Omega u_h\varphi_i &= \int_\Omega u\varphi_i, && \text{for all $i$}
# \end{align*}
# We assume that on an element $E$ we have
# \begin{align*}
#   \varphi_{g_E(k)}(x) = \hat\varphi_k(F_E^{-1}(x))
# \end{align*}
# for $k=0,1,2$ and where $g_E$ denotes the local to global dof mapper
# and $F_E$ is the reference mapping.
#
# So we need to compute
# \begin{align*}
#   M^E_{kl} := \int_{\hat{E}} |DF|\hat\varphi_k\hat\varphi_l~, &&
#   b^E_l := \int_E u\varphi_l~,
# \end{align*}
# and distribute these into a global matrix.

import numpy
import scipy.sparse
import scipy.sparse.linalg
from dune.geometry import quadratureRules, quadratureRule
from dune.grid import cartesianDomain, gridFunction


def _coerce_triangle_vertices(element_or_vertices):
    """Return triangle vertices as a (3,2) float array."""
    if hasattr(element_or_vertices, "geometry"):
        geo = element_or_vertices.geometry
        return numpy.array([geo.corner(i) for i in range(3)], dtype=float)

    verts = numpy.asarray(element_or_vertices, dtype=float)
    if verts.shape != (3, 2):
        raise ValueError("triangle vertices must have shape (3, 2)")
    return verts


class SpaceBase:
    """Shared interface for all local spaces in this single-file pass."""

    def bind(self, element_or_vertices):
        raise NotImplementedError

    def evaluateLocal(self, x):
        raise NotImplementedError

    def interpolate(self, gf):
        raise NotImplementedError

# We will use a triangular grid for this exercise
from dune.alugrid import aluConformGrid

# ## The shape functions
# We use a simple class here to collect all required
# information about the finite element space, i.e.,
# how to evaluate the shape functions on the reference
# element (together with their derivatives). We also
# setup a mapper to attach the degrees of freedom to
# the entities of the grid.

class LinearLagrangeSpace(SpaceBase):
    def __init__(self,view):
        self.localDofs = 3
        self.view   = view
        self.dim    = view.dimension
        self.layout = lambda gt: 1 if gt.dim == 0 else 0
        self.mapper = view.mapper(self.layout)
        self.points = numpy.array( [ [0,0],[1,0],[0,1] ] )
        # by default this is the reference element
        self.vertices = numpy.array( [ [0,0],[1,0],[0,1] ], dtype=float )
    # bind to a different physical triangles
    # - store origin and coordinate axes
    # Use the vertices when setting up the CLS so that we can easily
    # switch between a physical VEM and a reference VEM
    def bind(self, element_or_vertices):
        vertices = _coerce_triangle_vertices(element_or_vertices)
        self.vertices[0] = vertices[0]
        self.vertices[1] = vertices[1] - vertices[0]
        self.vertices[2] = vertices[2] - vertices[0]
    # evaluate basis function in local coordinates so
    # (0,0) is vertex[0], (1,0) is vertex[1], (0,1) is vertex[2]
    def evaluateLocal(self, x):
        bary = 1.-x[0]-x[1], x[0], x[1]
        return numpy.array( bary )
    def interpolate(self,gf):
        dofs = numpy.zeros(len(self.mapper))
        for e in self.view.elements:
            dofs[self.mapper(e)] = gf(e,self.points.transpose())
        return dofs

class QuadraticLagrangeSpace(SpaceBase):
    def __init__(self,view):
        self.localDofs = 6
        self.view   = view
        self.dim    = view.dimension
        self.layout = lambda gt: 1 if gt.dim == 0 or gt.dim == 1 else 0
        self.mapper = view.mapper(self.layout)
        self.points = numpy.array( [ [0,0],[1,0],[0,1],
                                   [0.5,0],[0,0.5],[0.5,0.5] ] )
        # by default this is the reference element
        self.vertices = numpy.array( [ [0,0],[1,0],[0,1] ], dtype=float )
    # bind to a different physical triangles
    # - store origin and coordinate axes
    def bind(self, element_or_vertices):
        vertices = _coerce_triangle_vertices(element_or_vertices)
        self.vertices[0] = vertices[0]
        self.vertices[1] = vertices[1] - vertices[0]
        self.vertices[2] = vertices[2] - vertices[0]
    def evaluateLocal(self, x):
        bary = 1.-x[0]-x[1], x[0], x[1]
        return numpy.array([ bary[i]*(2.*bary[i]-1.) for i in range(3) ] +\
               [ 4.*bary[(3-j)%3]*bary[(4-j)%3] for j in range(3) ])
    def interpolate(self, gf):
        dofs = numpy.zeros(len(self.mapper))
        points_t = self.points.transpose()
        for e in self.view.elements:
            dofs[self.mapper(e)] = gf(e, points_t)
        return dofs


class CubicHermiteTriangleSpace(SpaceBase):
    """
    Cubic Hermite triangle (P3) on the reference triangle with vertices
    (0,0), (1,0), (0,1).

    Local DOF ordering (10 dofs):
      0: u(v0)         1: du/dx(v0)      2: du/dy(v0)
      3: u(v1)         4: du/dx(v1)      5: du/dy(v1)
      6: u(v2)         7: du/dx(v2)      8: du/dy(v2)
      9: u(barycentre)
    """
    def __init__(self, view):
        self.localDofs = 10
        self.view = view
        self.dim = view.dimension

        # 3 dofs per vertex, 1 dof per cell interior (barycentre)
        self.layout = lambda gt: ( 3 if gt.dim == 0 else
                                  (1 if gt.dim == self.dim else 0))
        self.mapper = view.mapper(self.layout)

        # Points corresponding to local DOFs (repeated for derivative DOFs)
        self.points = numpy.array([
            [0.0, 0.0],  # v0 value
            [0.0, 0.0],  # v0 dx
            [0.0, 0.0],  # v0 dy
            [1.0, 0.0],  # v1 value
            [1.0, 0.0],  # v1 dx
            [1.0, 0.0],  # v1 dy
            [0.0, 1.0],  # v2 value
            [0.0, 1.0],  # v2 dx
            [0.0, 1.0],  # v2 dy
            [1.0/3.0, 1.0/3.0],  # barycentre value
        ], dtype=float)

        # by default: reference element
        self.vertices = numpy.array([[0., 0.], [1., 0.], [0., 1.]],
                                    dtype=float)

        # Precompute reference Hermite basis on monomial basis:
        self._invDofMatrix = self._buildInverseDofMatrix()

        # Initialize geometric data + transformation matrices for reference
        # cell (J = I)
        self.bind(numpy.array([[0., 0.], [1., 0.], [0., 1.]], dtype=float))

    def bind(self, element_or_vertices):
        """
        Bind to a physical triangle.
        element_or_vertices may be either a grid element or a (3,2) vertex
        array.

        Stores:
          J      = d x / d x_hat   (reference -> physical)
          JinvT  = (d x / d x_hat)^(-T)
          M      = basis transform, M = V^T
        """
        vertices = _coerce_triangle_vertices(element_or_vertices)
        self.vertices[0] = vertices[0]
        self.vertices[1] = vertices[1] - vertices[0]
        self.vertices[2] = vertices[2] - vertices[0]

        # Jacobian of affine map x = x0 + J x_hat (reference -> physical)
        self.J = numpy.column_stack((self.vertices[1], self.vertices[2]))

        # Construct mapping matrix
        self.M = numpy.eye(self.localDofs, dtype=float)

        for base in (0, 3, 6):
            # derivative block indices are [base+1 : base+3]
            sl = slice(base+1, base+3)
            self.M[sl, sl] = self.J      # basis transform M = V^T

    @staticmethod
    def _monomials(x, y):
        # Basis for P3: [1, x, y, x^2, xy, y^2, x^3, x^2 y, x y^2, y^3]
        return numpy.array([
            1.0,
            x, y,
            x*x, x*y, y*y,
            x*x*x, x*x*y, x*y*y, y*y*y
        ], dtype=float)

    @staticmethod
    def _dmonomials_dx(x, y):
        return numpy.array([
            0.0,
            1.0, 0.0,
            2.0*x, y, 0.0,
            3.0*x*x, 2.0*x*y, y*y, 0.0
        ], dtype=float)

    @staticmethod
    def _dmonomials_dy(x, y):
        return numpy.array([
            0.0,
            0.0, 1.0,
            0.0, x, 2.0*y,
            0.0, x*x, 2.0*x*y, 3.0*y*y
        ], dtype=float)

    def _buildInverseDofMatrix(self):
        # Rows = reference DOFs, columns = monomials
        # Reference derivative DOFs here are d/dxi and d/deta.
        A = numpy.zeros((10, 10), dtype=float)

        v0 = (0.0, 0.0)
        v1 = (1.0, 0.0)
        v2 = (0.0, 1.0)
        bc = (1.0/3.0, 1.0/3.0)

        row = 0
        for (x, y) in (v0, v1, v2):
            A[row,   :] = self._monomials(x, y)      # value
            A[row+1, :] = self._dmonomials_dx(x, y)  # d/dxi on reference
            A[row+2, :] = self._dmonomials_dy(x, y)  # d/deta on reference
            row += 3

        A[row, :] = self._monomials(*bc)  # barycentre value

        return numpy.linalg.inv(A)

    def _evaluateReferenceLocal(self, x):
        """
        Evaluate reference Hermite basis (reference-node basis with d/dxi,
        d/deta DOFs) at local coordinates x=(xi,eta) on the reference triangle.
        """
        xx = float(x[0])
        yy = float(x[1])
        m = self._monomials(xx, yy)
        return self._invDofMatrix.T.dot(m)

    def evaluateLocal(self, x):
        """
        Evaluate all 10 PHYSICAL Hermite basis functions at local coordinates
        x=(xi,eta), pulled back to the reference point x.

        This applies the Kirby nodal-basis transform:
            Psi = M F^*(Psi_hat)
        using the current element geometry from bind(...).
        """
        phi_ref = self._evaluateReferenceLocal(x)
        return self.M.dot(phi_ref)

    def interpolate(self, gf):
        """
        Interpolate a scalar grid function into Hermite DOFs.
        Kirby-aligned DOFs are stored as PHYSICAL derivatives (ux, uy) at
        vertices.

        Expects gf.jacobian(e,x) to return the physical Cartesian gradient at
        local point x.
        """
        dofs = numpy.zeros(len(self.mapper), dtype=float)
        verts = [numpy.array([0.0, 0.0]), numpy.array([1.0, 0.0]),
                 numpy.array([0.0, 1.0])]
        bc = numpy.array([1.0/3.0, 1.0/3.0])

        if not hasattr(gf, "jacobian"):
            raise NotImplementedError(
                "Hermite interpolation needs derivatives. "
                "Provide a grid function with gf.jacobian(e,x)."
            )

        for e in self.view.elements:
            geo = e.geometry
            vertices = numpy.array([geo.corner(i) for i in range(3)])
            self.bind(vertices)

            idx = self.mapper(e)
            local = numpy.zeros(self.localDofs, dtype=float)

            for i, xi in enumerate(verts):
                base = 3*i
                local[base] = float(gf(e, xi))  # value DOF

                # Physical derivatives (Kirby 3.2 style)
                g_phys = numpy.asarray(gf.jacobian(e, xi)).reshape(-1)
                if g_phys.size < 2:
                    raise ValueError("gf.jacobian(e,x) did not return a 2D " \
                    "gradient.")
                local[base+1] = float(g_phys[0])  # du/dx
                local[base+2] = float(g_phys[1])  # du/dy

            local[9] = float(gf(e, bc))  # barycentre value
            dofs[idx] = local

        return dofs


class LagrangePhysicalVEMSpace(SpaceBase):
    """
    k=1 scalar H1-conforming VEM (value projection only) on triangles.

    evaluateLocal(xhat) returns the VALUE PROJECTION of the (virtual) local
    basis, i.e. [Pi^E_0 phi_i](xhat), not the true virtual basis values
    phi_i(xhat).

    k=1 tuple:
      - local shape function space: virtual
      - dofs Λ: vertex values only
      - value projection basis B0: P1(E)
      - constraints C0: none (since k-2 = -1)

    For triangles, this reproduces the linear FE basis after projection,
    but it is computed through the VEM CLS machinery element-by-element.
    """

    def __init__(self, view):
        self.localDofs = 3
        self.view = view
        self.dim = view.dimension
        self.layout = lambda gt: 1 if gt.dim == 0 else 0
        self.mapper = view.mapper(self.layout)
        self.points = numpy.array( [ [0,0],[1,0],[0,1] ] )
        self.vertices = numpy.array( [ [0,0],[1,0],[0,1] ], dtype=float )

        # Physical element data (updated in bind)
        self.J = numpy.eye(2)
        self.x0 = numpy.array([0.0, 0.0], dtype=float)
        self.xE = numpy.array([1/3, 1/3], dtype=float)
        self.hE = 1.0

        # Coefficients of projected local basis in B0 = P1(E):
        # columns = local basis functions, rows = polynomial basis coefficients
        #   Pi0phi_k(x) = sum_alpha self._Pi0Coeffs[alpha,k] * m_alpha(x)
        self._Pi0Coeffs = numpy.zeros((3, 3), dtype=float)

        # Bind to reference triangle by default
        self.bind(numpy.array([[0., 0.], [1., 0.], [0., 1.]], dtype=float))

    # ----------------------------
    # Geometry handling
    # ----------------------------
    def bind(self, element_or_vertices):
        """
        Bind space to one physical triangle and build the local value
        projection matrix.

        element_or_vertices may be either a grid element or a (3,2) vertex
        array.
        """
        verts = _coerce_triangle_vertices(element_or_vertices)

        # store in your existing convention
        self.vertices[0] = verts[0]
        self.vertices[1] = verts[1] - verts[0]
        self.vertices[2] = verts[2] - verts[0]

        self.x0 = self.vertices[0].copy()
        self.J = numpy.column_stack((self.vertices[1], self.vertices[2]))

        # Element barycentre and a simple diameter scale (for scaled monomials)
        self.xE = verts.mean(axis=0)
        e01 = numpy.linalg.norm(verts[1] - verts[0])
        e12 = numpy.linalg.norm(verts[2] - verts[1])
        e20 = numpy.linalg.norm(verts[0] - verts[2])
        self.hE = max(e01, e12, e20)
        if self.hE <= 0:
            raise ValueError("Degenerate element with non-positive diameter")

        # Build the k=1 value projection matrix on this element:
        #   min || A c - e_k ||^2  subject to no constraints
        # for each local basis dof k.
        #
        # A_{i,alpha} = lambda_i(m_alpha), with lambda_i = vertex value dofs.
        A = self._build_value_projection_dof_matrix()

        # RHS matrix = [e_0, e_1, e_2] (nodal basis property λ_i(φ_j)=δ_ij)
        I = numpy.eye(self.localDofs, dtype=float)

        # No constraints for k=1 (M_{-1} = empty)
        self._Pi0Coeffs = numpy.linalg.solve(A, I)

    # ----------------------------
    # Polynomial basis for Pi0 (B0 = P1(E))
    # ----------------------------
    def _poly_basis_value(self, x_phys):
        """
        Basis B0 = P1(E), represented here as scaled monomials centered at xE:
          m0 = 1
          m1 = (x - xE)_x / hE
          m2 = (x - xE)_y / hE
        """
        y = (numpy.asarray(x_phys, dtype=float) - self.xE) / self.hE
        return numpy.array([1.0, y[0], y[1]], dtype=float)

    def _physical_point(self, xhat):
        """Map local reference coordinates xhat=(xi,eta) to physical x."""
        xhat = numpy.asarray(xhat, dtype=float)
        return self.x0 + self.J.dot(xhat)

    def _build_value_projection_dof_matrix(self):
        """
        Build A with entries A[i,alpha] = λ_i(m_alpha), where λ_i are vertex-
        value dofs.
        For k=1, i=0,1,2 correspond to the 3 vertices of the current triangle.
        """
        A = numpy.zeros((self.localDofs, 3), dtype=float)

        # Physical triangle vertices (reconstruct from x0 and edge vectors)
        v0 = self.x0
        v1 = self.x0 + self.vertices[1]
        v2 = self.x0 + self.vertices[2]
        verts = [v0, v1, v2]

        for i, xv in enumerate(verts):
            A[i, :] = self._poly_basis_value(xv)  # vertex value dof = point evaluation

        return A

    # ----------------------------
    # Projected basis evaluation (what assembly needs for VEM)
    # ----------------------------
    def evaluateLocal(self, x):
        """
        Evaluate the projected local basis values [Pi^E_0 phi_i](xhat), i=0,1,2,
        at local coordinates xhat=(xi,eta) on the reference triangle.
        """
        x_phys = self._physical_point(x)
        m = self._poly_basis_value(x_phys)       # shape (3,)
        return self._Pi0Coeffs.T.dot(m)          # shape (3,)

    # ----------------------------
    # Interpolation into global dofs (vertex values)
    # ----------------------------
    def interpolate(self, gf):
        dofs = numpy.zeros(len(self.mapper))
        ptsT = self.points.transpose()  # shape (2,3) as in your existing code
        for e in self.view.elements:
            dofs[self.mapper(e)] = gf(e, ptsT)
        return dofs


class LagrangeReferenceMappedVEMSpace(SpaceBase):
    """
    k=1 scalar H1-conforming VEM (value projection only) on triangles,
    but with the value projection computed ONCE on the reference triangle.

    For value-only DOFs and affine maps F, we use
        Pi0^E u = (Pi0^hat (u o F)) o F^{-1}.

    This means:
      - no element-wise CLS solve in bind(...)
      - evaluateLocal(xhat) uses the reference projection directly
    """

    def __init__(self, view):
        self.localDofs = 3
        self.view = view
        self.dim = view.dimension
        self.layout = lambda gt: 1 if gt.dim == 0 else 0
        self.mapper = view.mapper(self.layout)
        self.points = numpy.array( [ [0,0],[1,0],[0,1] ] )
        self.vertices = numpy.array( [ [0,0],[1,0],[0,1] ], dtype=float )

        self.x0 = numpy.array([0.0, 0.0], dtype=float)
        self.J = numpy.eye(2, dtype=float)
        self.Jinv = numpy.eye(2, dtype=float)
        self.detJ = 1.0

        # Reference-element polynomial basis data (for B0 = P1(\hat E))
        self.xE_hat = numpy.array([1.0/3.0, 1.0/3.0], dtype=float)

        # Use the "diameter = max edge length" convention on the reference
        # triangle
        self.hE_hat = numpy.sqrt(2.0)

        # Precompute the reference CLS projection matrix once:
        #   Pi0_hat phi_k = sum_alpha coeffs_ref[alpha,k] mhat_alpha
        self._Pi0CoeffsRef = self._build_reference_value_projection()

        # Bind to reference by default
        self.bind(numpy.array([[0., 0.], [1., 0.], [0., 1.]], dtype=float))

    # ----------------------------
    # Geometry handling
    # ----------------------------
    def bind(self, element_or_vertices):
        """
        Bind to a physical triangle. No CLS solve. Just store F and F^{-1}
        data. element_or_vertices may be either a grid element or a (3,2)
        vertex array.
        """
        verts = _coerce_triangle_vertices(element_or_vertices)

        self.vertices[0] = verts[0]
        self.vertices[1] = verts[1] - verts[0]
        self.vertices[2] = verts[2] - verts[0]

        self.x0 = self.vertices[0].copy()
        self.J = numpy.column_stack((self.vertices[1], self.vertices[2]))
        self.detJ = float(numpy.linalg.det(self.J))
        if abs(self.detJ) <= 1e-14:
            raise ValueError("Degenerate element (detJ=0)")
        self.Jinv = numpy.linalg.inv(self.J)

    # ----------------------------
    # Reference basis B0 = P1(\hat E)
    # ----------------------------
    def _poly_basis_ref(self, xhat):
        """
        Scaled monomial basis on reference triangle:
          m0 = 1
          m1 = (xi  - xi_E)/hhat
          m2 = (eta - eta_E)/hhat
        """
        y = (numpy.asarray(xhat, dtype=float) - self.xE_hat) / self.hE_hat
        return numpy.array([1.0, y[0], y[1]], dtype=float)

    def _build_reference_value_projection_dof_matrix(self):
        """
        Ahat[i,alpha] = hat lambda_i(mhat_alpha), with
            hat lambda_i = vertex value dofs.
        """
        Ahat = numpy.zeros((self.localDofs, 3), dtype=float)
        ref_vertices = [numpy.array([0.0, 0.0]),
                        numpy.array([1.0, 0.0]),
                        numpy.array([0.0, 1.0])]
        for i, xhat_v in enumerate(ref_vertices):
            Ahat[i, :] = self._poly_basis_ref(xhat_v)
        return Ahat

    def _build_reference_value_projection(self):
        """
        k=1 CLS on reference triangle.
        For k=1 there are no constraints, and the system is square.
        """
        Ahat = self._build_reference_value_projection_dof_matrix()
        I = numpy.eye(self.localDofs, dtype=float)  # nodal-basis property
        return numpy.linalg.solve(Ahat, I)

    # ----------------------------
    # Projected basis evaluation
    # ----------------------------
    def evaluateLocal(self, x):
        """
        Evaluate [Pi0^E phi_i](xhat).

        Since x is already a reference quadrature point, and
            Pi0^E phi_i = (Pi0^hat hat phi_i) o F^{-1},
        we can evaluate directly with the reference projection at xhat = x.
        """
        mhat = self._poly_basis_ref(x)
        return self._Pi0CoeffsRef.T.dot(mhat)

    # ----------------------------
    # Interpolation into global dofs (vertex values)
    # ----------------------------
    def interpolate(self, gf):
        dofs = numpy.zeros(len(self.mapper))
        ptsT = self.points.transpose()
        for e in self.view.elements:
            dofs[self.mapper(e)] = gf(e, ptsT)
        return dofs


class HermiteK3PhysicalVEMSpace(SpaceBase):
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
        self._Pi0Coeffs = self._solve_cls_kkt_all_rhs(
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
    # CLS solve (KKT form)
    # ------------------------------------------------------------------
    def _solve_cls_kkt_all_rhs(self, A, C, G):
        """
        Solve, for all local basis functions (columns j), the constrained LS:
            min ||A c_j - e_j||^2  s.t.  C c_j = G[:,j]
        Returns X of shape (polyDim, localDofs).
        """
        ATA = A.T.dot(A)
        K = numpy.block([
            [ATA, C.T],
            [C, numpy.zeros((C.shape[0], C.shape[0]), dtype=float)]
        ])
        RHS = numpy.vstack([
            A.T.dot(numpy.eye(self.localDofs, dtype=float)),
            G
        ])

        try:
            sol = numpy.linalg.solve(K, RHS)
        except numpy.linalg.LinAlgError:
            sol = numpy.linalg.lstsq(K, RHS, rcond=None)[0]

        return sol[:self.polyDim, :]

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


class HermiteK3MappedVEMSpace(SpaceBase):
    """
    k=3 Hermite-style VEM value-projection space.

    This version solves the constrained least-squares projection ONCE on the
    reference element, then evaluates by composition with F^{-1}, and applies a
    Hermite-style basis transform block.

    IMPORTANT:
      This is NOT mathematically equivalent to HermiteK3PhysicalVEMSpace in
      general.
    """

    def __init__(self, view):
        self.view = view
        self.dim = view.dimension
        self.localDofs = 12
        self.polyDim = 10
        self.constraintDim = 3

        # 3 DOFs per vertex, 3 DOFs per cell (interior moments)
        self.layout = lambda gt: (3 if gt.dim == 0 else
                                  (3 if gt.dim == self.dim else 0))
        self.mapper = view.mapper(self.layout)
        self.vertices = numpy.array( [ [0,0],[1,0],[0,1] ], dtype=float )

        # "Points" are only meaningful for vertex-based DOFs; for moment DOFs,
        # placeholders are used.
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

        # Reference geometry data
        self.xE_hat = numpy.array([1.0/3.0, 1.0/3.0], dtype=float)
        self.hE_hat = numpy.sqrt(2.0)
        self.area_hat = 0.5

        # Reference vertex derivative scales (shared on reference cell)
        self._hV_hat = numpy.array([self.hE_hat, self.hE_hat, self.hE_hat],
                                   dtype=float)

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
        self._constraint_rhs_selector = numpy.zeros((self.constraintDim,
                                                     self.localDofs),
                                                     dtype=float)
        self._constraint_rhs_selector[0, 9]  = 1.0
        self._constraint_rhs_selector[1, 10] = 1.0
        self._constraint_rhs_selector[2, 11] = 1.0

        # Hermite-style mapped basis transform (derivative blocks only)
        self.M = numpy.eye(self.localDofs, dtype=float)

        # Precompute reference CLS projector once
        Ahat, Chat = self._build_reference_A_and_C()
        self._Pi0CoeffsRef = self._solve_cls_kkt_all_rhs(
            A=Ahat, C=Chat, G=self._constraint_rhs_selector
        )

        # Set up scaling constants for derivative dofs (characteristic lengths
        # associated with each vertex)
        self._build_vertex_effective_h()

        # Initialize geometry on reference triangle
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
            self._hV_local = numpy.array([self.hE_hat, self.hE_hat, self.hE_hat],
                                         dtype=float)

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

        # Barycentre and diameter for physical scaled monomials (used in M
        # block and interpolation moments)
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

        # Hermite-style derivative DOF basis transform
        self.M.fill(0.0)
        numpy.fill_diagonal(self.M, 1.0)
        # Mapped gradients scale with J and characteristic length
        for i, base in enumerate((0, 3, 6)):
            sl = slice(base+1, base+3)
            self.M[sl, sl] = (self._hV_hat[i] / self._hV_local[i]) * self.J

        # Constant moment block scales trivially and linear moments scale with
        # const*J^{-T}
        self.M[9, 9] = 1.0
        self.M[10:12, 10:12] = (self.hE / self.hE_hat) * self.Jinv.T

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
    # CLS solve (KKT form)
    # ------------------------------------------------------------------
    def _solve_cls_kkt_all_rhs(self, A, C, G):
        """
        Solve, for all local basis functions (columns j), the constrained LS:
            min ||A c_j - e_j||^2  s.t.  C c_j = G[:,j]
        Returns X of shape (polyDim, localDofs).
        """
        ATA = A.T.dot(A)
        K = numpy.block([
            [ATA, C.T],
            [C, numpy.zeros((C.shape[0], C.shape[0]), dtype=float)]
        ])
        RHS = numpy.vstack([
            A.T.dot(numpy.eye(self.localDofs, dtype=float)),
            G
        ])

        try:
            sol = numpy.linalg.solve(K, RHS)
        except numpy.linalg.LinAlgError:
            sol = numpy.linalg.lstsq(K, RHS, rcond=None)[0]

        return sol[:self.polyDim, :]

    # ------------------------------------------------------------------
    # Reference CLS matrices
    # ------------------------------------------------------------------
    def _build_reference_A_and_C(self):
        """
        Build reference CLS matrices using:
          - reference derivatives d/dxi, d/deta at reference vertices
          - reference interior moments against M1(hat E)
        """
        A = numpy.zeros((self.localDofs, self.polyDim), dtype=float)
        C = numpy.zeros((self.constraintDim, self.polyDim), dtype=float)

        ref_verts = [numpy.array([0.0, 0.0]),
                     numpy.array([1.0, 0.0]),
                     numpy.array([0.0, 1.0])]

        row = 0
        for iv, xhat_v in enumerate(ref_verts):
            A[row, :] = self._p3_basis_ref(xhat_v)
            dxi, deta = self._p3_basis_grad_ref(xhat_v)
            hhat_a = self._hV_hat[iv]
            A[row+1, :] = hhat_a * dxi
            A[row+2, :] = hhat_a * deta
            row += 3

        mom_rows = numpy.zeros((3, self.polyDim), dtype=float)
        for p in self._momentQuad:
            xhat = p.position
            wref = p.weight
            pvals = self._p3_basis_ref(xhat)
            qvals = self._m1_basis_ref(xhat)
            # # (1/|Ehat|)\int_E = 2 * \sum w_ref (...) since |Ehat| = 0.5
            mom_rows += 2.0 * wref * numpy.outer(qvals, pvals)

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
        Experimental mapped evaluation:
          1) evaluate reference projected basis at xhat
          2) apply Hermite-style basis transform M (values, derivatives, and
             linear moments)
        Returns shape (12,).
        """
        mhat = self._p3_basis_ref(x)
        phi_ref_proj = self._Pi0CoeffsRef.T.dot(mhat)
        return self.M.dot(phi_ref_proj)

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


def assemble(space, force, quad_order):
    """Assemble the global mass matrix and right-hand side for a space."""
    rhs = numpy.zeros(len(space.mapper))

    local_entries = space.localDofs
    local_matrix = numpy.zeros((local_entries, local_entries), dtype=float)

    global_entries = local_entries**2 * space.view.size(0)
    value = numpy.zeros(global_entries, dtype=float)
    row_index = numpy.zeros(global_entries, int)
    col_index = numpy.zeros(global_entries, int)

    start = 0
    for e in space.view.elements:
        geo = e.geometry
        space.bind(e)

        indices = space.mapper(e)
        local_matrix.fill(0.0)
        for p in quadratureRule(e.type, quad_order):
            x = p.position
            w = p.weight * geo.integrationElement(x)
            phi_vals = numpy.asarray(space.evaluateLocal(x), dtype=float)

            rhs[indices] += w * force(e, x) * phi_vals
            local_matrix += w * numpy.outer(phi_vals, phi_vals)

        for i in range(local_entries):
            for j in range(local_entries):
                entry = start + i * local_entries + j
                value[entry] = local_matrix[i, j]
                row_index[entry] = indices[i]
                col_index[entry] = indices[j]
        start += local_entries**2

    matrix = scipy.sparse.csr_matrix(
        (value, (row_index, col_index)),
        shape=(len(space.mapper), len(space.mapper)),
    )
    return rhs, matrix


def error(view, u, uh, quad_order=5):
    rules = quadratureRules(quad_order)
    l2 = 0.0
    for e in view.elements:
        geometry = e.geometry
        for p in rules(e.type):
            hatx = p.position
            weight = p.weight * geometry.integrationElement(hatx)
            l2 += (uh(e, hatx) - u(e, hatx))**2 * weight
    return [numpy.sqrt(l2)]


def compare_projectors(spaceA, spaceB, quad_order=8, print_per_element=False,
                       compare_local_mass=True):
    """
    Compare two projector-based spaces on the same grid by evaluating their
    projected local basis vectors phi(xhat) = evaluateLocal(xhat) on each
    element.

    Metrics are quadrature-based over the physical mesh:
      - global L2/Frobenius norm of projector difference
      - relative difference (against spaceA)
      - per-local-basis-column L2 differences
      - max pointwise absolute basis-value discrepancy at quadrature points
      - (optional) local mass-matrix discrepancy induced by projected basis
        values

    Parameters
    ----------
    spaceA, spaceB : spaces with methods
        - bind(e)
        - evaluateLocal(xhat) -> shape (localDofs,)
        - localDofs
        - view
    quad_order : int
        Triangle quadrature order for the comparison integrals.
    print_per_element : bool
        If True, print a short line of metrics per element.
    compare_local_mass : bool
        If True, also compare local matrices int phi_i phi_j on each element.

    Returns
    -------
    stats : dict
        Summary dictionary with global metrics and (optionally) per-element
        data.
    """
    if spaceA.localDofs != spaceB.localDofs:
        raise ValueError(f"localDofs mismatch: {spaceA.localDofs} vs {spaceB.localDofs}")
    if len(spaceA.mapper) != len(spaceB.mapper):
        raise ValueError(f"global dof count mismatch: {len(spaceA.mapper)} vs"\
                         f"{len(spaceB.mapper)}")

    nloc = spaceA.localDofs

    total_diff_sq = 0.0
    total_A_sq = 0.0
    total_B_sq = 0.0
    total_area = 0.0
    global_max_abs = 0.0

    col_diff_sq = numpy.zeros(nloc, dtype=float)
    col_A_sq = numpy.zeros(nloc, dtype=float)
    col_B_sq = numpy.zeros(nloc, dtype=float)

    mass_diff_sq_sum = 0.0   # sum over elements of ||M_A^E - M_B^E||_F^2
    mass_A_sq_sum = 0.0      # sum over elements of ||M_A^E||_F^2

    element_stats = []

    # Iterate over elements (same grid assumed)
    for elem_id, e in enumerate(spaceA.view.elements):
        geo = e.geometry

        spaceA.bind(e)
        spaceB.bind(e)

        ediff_sq = 0.0
        eA_sq = 0.0
        eB_sq = 0.0
        emax_abs = 0.0
        earea = 0.0

        ecol_diff_sq = numpy.zeros(nloc, dtype=float)
        ecol_A_sq = numpy.zeros(nloc, dtype=float)
        ecol_B_sq = numpy.zeros(nloc, dtype=float)

        if compare_local_mass:
            MA = numpy.zeros((nloc, nloc), dtype=float)
            MB = numpy.zeros((nloc, nloc), dtype=float)

        for p in quadratureRule(e.type, quad_order):
            xhat = p.position
            w = float(p.weight * geo.integrationElement(xhat))
            earea += w

            phiA = numpy.asarray(spaceA.evaluateLocal(xhat),
                                 dtype=float).reshape(-1)
            phiB = numpy.asarray(spaceB.evaluateLocal(xhat),
                                 dtype=float).reshape(-1)

            if phiA.shape[0] != nloc or phiB.shape[0] != nloc:
                raise ValueError(
                    f"evaluateLocal shape mismatch on element {elem_id}: "
                    f"{phiA.shape} vs {phiB.shape}, expected ({nloc},)"
                )

            d = phiA - phiB

            ediff_sq += w * float(numpy.dot(d, d))
            eA_sq += w * float(numpy.dot(phiA, phiA))
            eB_sq += w * float(numpy.dot(phiB, phiB))

            ecol_diff_sq += w * (d * d)
            ecol_A_sq += w * (phiA * phiA)
            ecol_B_sq += w * (phiB * phiB)

            pt_max = float(numpy.max(numpy.abs(d)))
            if pt_max > emax_abs:
                emax_abs = pt_max

            if compare_local_mass:
                MA += w * numpy.outer(phiA, phiA)
                MB += w * numpy.outer(phiB, phiB)

        total_diff_sq += ediff_sq
        total_A_sq += eA_sq
        total_B_sq += eB_sq
        total_area += earea
        global_max_abs = max(global_max_abs, emax_abs)

        col_diff_sq += ecol_diff_sq
        col_A_sq += ecol_A_sq
        col_B_sq += ecol_B_sq

        estats = {
            "element_id": elem_id,
            "area": earea,
            "L2_fro_diff": numpy.sqrt(ediff_sq),
            "L2_fro_A": numpy.sqrt(eA_sq),
            "L2_fro_B": numpy.sqrt(eB_sq),
            "rel_to_A": (numpy.sqrt(ediff_sq) / max(numpy.sqrt(eA_sq), 1e-16)),
            "max_abs": emax_abs,
        }

        if compare_local_mass:
            m_diff = numpy.linalg.norm(MA - MB, ord="fro")
            m_A = numpy.linalg.norm(MA, ord="fro")
            estats["local_mass_fro_diff"] = m_diff
            estats["local_mass_rel_to_A"] = m_diff / max(m_A, 1e-16)

            mass_diff_sq_sum += m_diff * m_diff
            mass_A_sq_sum += m_A * m_A

        element_stats.append(estats)

        if print_per_element:
            msg = (f"[elem {elem_id:4d}] "
                   f"L2proj diff={estats['L2_fro_diff']:.3e} "
                   f"(rel {estats['rel_to_A']:.3e}), "
                   f"maxabs={estats['max_abs']:.3e}")
            if compare_local_mass:
                msg += (f",  Mdiff={estats['local_mass_fro_diff']:.3e} "
                        f"(rel {estats['local_mass_rel_to_A']:.3e})")
            print(msg)

    # Global summaries
    global_L2_diff = numpy.sqrt(total_diff_sq)
    global_L2_A = numpy.sqrt(total_A_sq)
    global_L2_B = numpy.sqrt(total_B_sq)

    per_col_L2_diff = numpy.sqrt(col_diff_sq)
    per_col_L2_A = numpy.sqrt(col_A_sq)
    per_col_rel = per_col_L2_diff / numpy.maximum(per_col_L2_A, 1e-16)

    stats = {
        "num_elements": len(element_stats),
        "num_global_dofs_A": len(spaceA.mapper),
        "num_global_dofs_B": len(spaceB.mapper),
        "local_dofs": nloc,
        "total_area": total_area,
        "global_L2_fro_diff": global_L2_diff,
        "global_L2_fro_A": global_L2_A,
        "global_L2_fro_B": global_L2_B,
        "global_rel_to_A": global_L2_diff / max(global_L2_A, 1e-16),
        "global_max_abs": global_max_abs,
        "per_local_basis_L2_diff": per_col_L2_diff,
        "per_local_basis_L2_A": per_col_L2_A,
        "per_local_basis_rel_to_A": per_col_rel,
        "element_stats": element_stats,
    }

    if compare_local_mass:
        stats["global_local_mass_fro_diff_l2sum"] = numpy.sqrt(mass_diff_sq_sum)
        stats["global_local_mass_fro_A_l2sum"] = numpy.sqrt(mass_A_sq_sum)
        stats["global_local_mass_rel_to_A_l2sum"] = (
            stats["global_local_mass_fro_diff_l2sum"] /
            max(stats["global_local_mass_fro_A_l2sum"], 1e-16)
        )

    # Pretty print summary
    print("\n=== Projector comparison summary ===")
    print(f"elements                : {stats['num_elements']}")
    print(f"local dofs              : {stats['local_dofs']}")
    print(f"global dofs (A, B)      : {stats['num_global_dofs_A']}, {stats['num_global_dofs_B']}")
    print(f"mesh area (quadrature)  : {stats['total_area']:.12g}")
    print(f"global ||PiA-PiB||      : {stats['global_L2_fro_diff']:.6e}")
    print(f"global ||PiA||          : {stats['global_L2_fro_A']:.6e}")
    print(f"global relative diff    : {stats['global_rel_to_A']:.6e}")
    print(f"global max abs diff     : {stats['global_max_abs']:.6e}")

    if compare_local_mass:
        print(f"mass diff (l2-sum elems): {stats['global_local_mass_fro_diff_l2sum']:.6e}")
        print(f"mass rel  (l2-sum elems): {stats['global_local_mass_rel_to_A_l2sum']:.6e}")

    print("per-local-basis relative L2 diff:")
    for i, r in enumerate(stats["per_local_basis_rel_to_A"]):
        print(f"  basis {i:2d}: {r:.6e}")

    return stats


# ## The main part of the code
# Construct the grid and a grid function for the
# right hand side, compute the projection and plot
# on a sequence of global grid refinements.


def build_demo_view(nx=10, ny=10):
    domain = cartesianDomain([0, 0], [1, 1], [nx, ny])
    return domain, aluConformGrid(domain)


def make_test_function(view):
    @gridFunction(view)
    def u(p):
        x, y = p
        return numpy.cos(2 * numpy.pi * x) * numpy.cos(2 * numpy.pi * y)

    return u


def make_projected_function(view, space, dofs):
    @gridFunction(view)
    def uh(e, x):
        space.bind(e)
        indices = space.mapper(e)
        phi_vals = numpy.asarray(space.evaluateLocal(x), dtype=float)
        local_dofs = dofs[indices]
        return numpy.dot(local_dofs, phi_vals)

    return uh


def run_projection_demo(space_factory=HermiteK3PhysicalVEMSpace, refinements=3,
                        plot=True, compare_mapped=True):
    _, view = build_demo_view()
    u = make_test_function(view)

    if plot:
        u.plot(level=3)

    old_pde_error = None
    for _ in range(refinements):
        space = space_factory(view)
        quad_order = 4 if space.localDofs == 3 else 6

        print("number of elements:", view.size(0), "number of dofs:", len(space.mapper))

        rhs, matrix = assemble(space, u, quad_order=quad_order)
        dofs = scipy.sparse.linalg.spsolve(matrix, rhs)
        uh = make_projected_function(view, space, dofs)

        if plot:
            uh.plot(level=1)

        err = error(view, u, uh)
        if old_pde_error is not None:
            eoc = [numpy.log(old_e / e) / numpy.log(2) for old_e, e in zip(old_pde_error, err)]
        else:
            eoc = None
        print("  pde problem:", err, eoc)
        old_pde_error = err

        view.hierarchicalGrid.globalRefine(2)

    if compare_mapped:
        _, compare_view = build_demo_view()
        space_global = HermiteK3PhysicalVEMSpace(compare_view)
        space_mapped = HermiteK3MappedVEMSpace(compare_view)
        return compare_projectors(
            space_global,
            space_mapped,
            quad_order=8,
            print_per_element=False,
            compare_local_mass=True,
        )

    return None


if __name__ == "__main__":
    run_projection_demo(space_factory=HermiteK3MappedVEMSpace, plot=False)
