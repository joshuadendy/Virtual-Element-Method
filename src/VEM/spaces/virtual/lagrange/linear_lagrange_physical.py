import numpy
from ...base import SpaceBase
from ...common.triangle_geometry import coerce_triangle_vertices

class LinearLagrangePhysicalVEMSpace(SpaceBase):
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
        verts = coerce_triangle_vertices(element_or_vertices)

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
