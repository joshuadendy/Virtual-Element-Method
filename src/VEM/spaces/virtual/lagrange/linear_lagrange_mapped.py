import numpy
from ...base import SpaceBase
from ...common.triangle_geometry import coerce_triangle_vertices

class LinearLagrangeMappedVEMSpace(SpaceBase):
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
        verts = coerce_triangle_vertices(element_or_vertices)

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
