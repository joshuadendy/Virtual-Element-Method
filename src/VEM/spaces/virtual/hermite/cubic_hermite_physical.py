import numpy
from dune.geometry import quadratureRule

from ...base import SpaceBase
from ...common.cls_projector import solve_cls_kkt_all_rhs
from ...common.scaled_monomials import (
    P1_EXPONENTS,
    P3_EXPONENTS,
    scaled_monomial_gradients,
    scaled_monomials,
)
from ...common.triangle_geometry import bind_affine_triangle
from ...common.vertex_scaling import build_vertex_effective_h


class CubicHermitePhysicalVEMSpace(SpaceBase):
    """
    k=3 Hermite-style VEM value-projection space (physical CLS on every element).
    """

    def __init__(self, view):
        self.view = view
        self.dim = view.dimension
        self.localDofs = 12
        self.polyDim = 10
        self.constraintDim = 3
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
        self._vertex_h = build_vertex_effective_h(self.view, self.mapper, measure="diameter")
        self._hV_local = numpy.array([self.hE_hat, self.hE_hat, self.hE_hat], dtype=float)

        self.bind(numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float))

    def bind(self, element_or_vertices):
        if hasattr(element_or_vertices, "geometry"):
            idx = self.mapper(element_or_vertices)
            self._hV_local = numpy.array([
                self._vertex_h[int(idx[0])],
                self._vertex_h[int(idx[3])],
                self._vertex_h[int(idx[6])],
            ], dtype=float)
        else:
            self._hV_local = numpy.array([self.hE_hat, self.hE_hat, self.hE_hat], dtype=float)

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

    def _physical_point(self, xhat):
        return self.x0 + self.J.dot(numpy.asarray(xhat, dtype=float))

    def _reference_point(self, xphys):
        return self.Jinv.dot(numpy.asarray(xphys, dtype=float) - self.x0)

    def _p3_basis_phys(self, x_phys):
        return scaled_monomials(x_phys, self.xE, self.hE, P3_EXPONENTS)

    def _p3_basis_grad_phys(self, x_phys):
        return scaled_monomial_gradients(x_phys, self.xE, self.hE, P3_EXPONENTS)

    def _m1_basis_phys(self, x_phys):
        return scaled_monomials(x_phys, self.xE, self.hE, P1_EXPONENTS)

    def _build_physical_A_and_C(self):
        A = numpy.zeros((self.localDofs, self.polyDim), dtype=float)
        C = numpy.zeros((self.constraintDim, self.polyDim), dtype=float)

        verts = [self.x0, self.x0 + self.vertices[1], self.x0 + self.vertices[2]]

        row = 0
        for iv, xv in enumerate(verts):
            A[row, :] = self._p3_basis_phys(xv)
            dmx, dmy = self._p3_basis_grad_phys(xv)
            h_a = self._hV_local[iv]
            A[row + 1, :] = h_a * dmx
            A[row + 2, :] = h_a * dmy
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

    def evaluateLocal(self, x):
        x_phys = self._physical_point(x)
        return self._Pi0Coeffs.T.dot(self._p3_basis_phys(x_phys))

    def evaluatePhysical(self, x_phys):
        return self.evaluateLocal(self._reference_point(x_phys))

    def interpolate(self, gf):
        dofs = numpy.zeros(len(self.mapper), dtype=float)

        if not hasattr(gf, "jacobian"):
            raise NotImplementedError(
                "Hermite-style VEM interpolation needs derivatives. "
                "Provide gf.jacobian(e,x) returning the physical gradient."
            )

        ref_vertices = [
            numpy.array([0.0, 0.0]),
            numpy.array([1.0, 0.0]),
            numpy.array([0.0, 1.0]),
        ]

        for e in self.view.elements:
            geo = e.geometry
            self.bind(e)

            idx = self.mapper(e)
            local = numpy.zeros(self.localDofs, dtype=float)

            for i, xhat_v in enumerate(ref_vertices):
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