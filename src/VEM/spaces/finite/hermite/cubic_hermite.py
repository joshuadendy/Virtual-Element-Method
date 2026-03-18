import numpy
from ...base import SpaceBase
from ...common.triangle_geometry import coerce_triangle_vertices

class CubicHermiteSpace(SpaceBase):
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
        vertices = coerce_triangle_vertices(element_or_vertices)
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
