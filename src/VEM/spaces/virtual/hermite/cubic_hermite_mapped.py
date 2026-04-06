import numpy
from dune.geometry import quadratureRule

from ...base import SpaceBase
from ...common.cls_projector import solve_cls_kkt_all_rhs
from ...common.hermite_mapping import build_k3_mapped_transform
from ...common.scaled_monomials import (
    P1_EXPONENTS,
    P3_EXPONENTS,
    scaled_monomial_gradients,
    scaled_monomials,
)
from ...common.triangle_geometry import bind_affine_triangle
from ...common.vertex_scaling import build_vertex_effective_h


class CubicHermiteMappedVEMSpace(SpaceBase):
    """
    k=3 Hermite-style VEM value-projection space.

    This version solves the constrained least-squares projection ONCE on the
    reference element, then evaluates by composition with F^{-1}, and applies a
    Hermite-style basis transform block.
    """

    def __init__(self, view):
        self.view = view
        self.dim = view.dimension
        self.localDofs = 12
        self.polyDim = 10
        self.constraintDim = 3

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

        Ahat, Chat = self._build_reference_A_and_C()
        self._Pi0CoeffsRef = solve_cls_kkt_all_rhs(A=Ahat, C=Chat, G=self._constraint_rhs_selector)

        self.M = numpy.eye(self.localDofs, dtype=float)
        self._vertex_h = build_vertex_effective_h(self.view, self.mapper, measure="diameter")
        self._hV_local = self._hV_hat.copy()

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

    def _p3_basis_ref(self, x_hat):
        return scaled_monomials(x_hat, self.xE_hat, self.hE_hat, P3_EXPONENTS)

    def _p3_basis_grad_ref(self, x_hat):
        return scaled_monomial_gradients(x_hat, self.xE_hat, self.hE_hat, P3_EXPONENTS)

    def _m1_basis_ref(self, x_hat):
        return scaled_monomials(x_hat, self.xE_hat, self.hE_hat, P1_EXPONENTS)

    def _m1_basis_phys(self, x_phys):
        return scaled_monomials(x_phys, self.xE, self.hE, P1_EXPONENTS)

    def _build_reference_A_and_C(self):
        A = numpy.zeros((self.localDofs, self.polyDim), dtype=float)
        C = numpy.zeros((self.constraintDim, self.polyDim), dtype=float)

        ref_vertices = [
            numpy.array([0.0, 0.0]),
            numpy.array([1.0, 0.0]),
            numpy.array([0.0, 1.0]),
        ]

        row = 0
        for iv, xhat_v in enumerate(ref_vertices):
            A[row, :] = self._p3_basis_ref(xhat_v)
            dxi, deta = self._p3_basis_grad_ref(xhat_v)
            hhat_a = self._hV_hat[iv]
            A[row + 1, :] = hhat_a * dxi
            A[row + 2, :] = hhat_a * deta
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

    def evaluateLocal(self, x):
        phi_ref_proj = self._Pi0CoeffsRef.T.dot(self._p3_basis_ref(x))
        return self.M.dot(phi_ref_proj)

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