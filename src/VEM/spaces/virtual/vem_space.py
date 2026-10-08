import numpy
from dune.geometry import quadratureRule

from ..base import SpaceBase
from ..common.cls_projector import solve_cls_kkt_all_rhs
from ..common.scaled_monomials import scaled_monomial_gradients, scaled_monomials, total_degree_exponents
from ..common.triangle_geometry import EDGES, REFERENCE_TRIANGLE_VERTICES, bind_affine_triangle
from ..common.vertex_scaling import build_vertex_effective_h


class VEMSpace(SpaceBase):
    """
    Order-k H1-conforming VEM on triangles with the CLS value projection (4.9)
    and gradient projection (4.43).

    element="lagrange" gives the Lagrange-type space of Section 5.1 (k >= 1) and
    element="hermite" the Hermite-type space of Section 5.2 (k >= 3). Both use
    B_0 = M_k(E), C_0 = interior moments and B_1 = [M_{k-1}(E)]^2.

    By default the projections are assembled on each physical element. With
    mapped=True they are assembled once on the reference triangle and evaluated
    as the surrogates (4.73) and (4.75):
      Pi_0 = M F^*(Pi_0[hat Phi]),  Pi_1 = J^{-T} M F^*(Pi_1[hat Phi]),
    where M = V^T and V F_*(Lambda) = hat Lambda. V is the identity apart from the
    Hermite gradient blocks (h_hat_v / h_v) J^T of (5.20) and a (-1)^j on the
    moments of edges parametrised against the reference orientation. Interior
    moments then use the transported reference basis, C_0 = F_{-*}(hat C_0).

    Local dof ordering (matching the DUNE mapper):
      vertices : u, or [u, h_v u_x, h_v u_y] for Hermite
      edges    : moments against M_{k-2}(e), or M_{k-4}(e) for Hermite, with each
                 edge parametrised from its lower global vertex index
      interior : moments against M_{k-2}(E)

    evaluateLocal returns Pi_0 of the virtual basis and evaluateLocalGradient Pi_1.
    """

    def __init__(self, view, order, element="lagrange", mapped=False):
        if element not in ("lagrange", "hermite"):
            raise ValueError(f"Unknown element type {element!r}.")
        hermite = element == "hermite"
        if order < (3 if hermite else 1):
            raise ValueError(f"order {order} is too low for a {element} VEM space.")
        self.view = view
        self.dim = view.dimension
        self.order = order
        self.element = element
        self._hermite = hermite
        self._ref = VEMSpace(view, order, element) if mapped else None
        self._nv = 3 if hermite else 1
        self._ne = order - 3 if hermite else order - 1
        self._ni = order * (order - 1) // 2
        self.localDofs = 3 * (self._nv + self._ne) + self._ni
        self.layout = lambda gt: (self._nv, self._ne, self._ni)[gt.dim]
        self.mapper = view.mapper(self.layout)

        ref = REFERENCE_TRIANGLE_VERTICES
        nodes = numpy.vstack([ref, [0.5 * (ref[a] + ref[b]) for a, b in EDGES], ref.mean(axis=0)])
        self.points = numpy.repeat(nodes, [self._nv] * 3 + [self._ne] * 3 + [self._ni], axis=0)

        self._value_exps = total_degree_exponents(order)
        self._grad_exps = total_degree_exponents(order - 1)
        self._moment_exps = total_degree_exponents(order - 2)

        r, w = numpy.polynomial.legendre.leggauss(order + 1)
        self._edge_r = 0.5 * (r + 1.0)
        self._edge_w = 0.5 * w
        self._edge_moments = self._edge_w * (self._edge_r - 0.5) ** numpy.arange(self._ne)[:, None]

        # Edge value projection: the degree-k polynomial in r fixed by the edge dofs
        # [u(0), u(1), (u'(0), u'(1)), moments], evaluated at the edge quadrature points.
        powers = numpy.arange(order + 1)
        rows = [powers == 0, numpy.ones(order + 1)]
        if hermite:
            rows += [powers == 1, powers]
        rows += list(self._edge_moments @ self._edge_r[:, None] ** powers)
        self._edge_trace = (self._edge_r[:, None] ** powers) @ numpy.linalg.inv(numpy.array(rows, dtype=float))

        quad = quadratureRule(next(iter(view.elements)).type, 2 * order + 2)
        self._cell_x = numpy.array([p.position for p in quad], dtype=float)
        self._cell_w = numpy.array([p.weight for p in quad], dtype=float)

        if hermite:
            self._vertex_h = build_vertex_effective_h(view, self.mapper, measure="adjacent_edge_average")
        self.bind(ref)

    def _bind_geometry(self, element_or_vertices):
        data = bind_affine_triangle(element_or_vertices)
        self.x0, self.J, self.Jinv = data["x0"], data["J"], data["Jinv"]
        self.detJ, self.area = data["detJ"], data["area"]
        self.xE, self.hE = data["xE"], data["hE"]
        verts = data["verts"]

        if hasattr(element_or_vertices, "geometry"):
            idx = self.mapper(element_or_vertices)
            gv = [int(idx[self._nv * i]) for i in range(3)]
            if self._hermite:
                self._hV = self._vertex_h[gv]
        else:
            gv = [0, 1, 2]
            if self._hermite:
                lengths = {ab: numpy.linalg.norm(verts[ab[1]] - verts[ab[0]]) for ab in EDGES}
                self._hV = numpy.array([numpy.mean([l for ab, l in lengths.items() if i in ab]) for i in range(3)])
        self._edges = [(a, b) if gv[a] < gv[b] else (b, a) for a, b in EDGES]
        self._cell_moments = 2.0 * self._cell_w * self._moment_basis(self._cell_x).T

    def _mono(self, xhat, exps):
        return scaled_monomials(self.x0 + xhat @ self.J.T, self.xE, self.hE, exps)

    def _mono_grad(self, xhat, exps):
        return scaled_monomial_gradients(self.x0 + xhat @ self.J.T, self.xE, self.hE, exps)

    def _moment_basis(self, xhat):
        if self._ref is not None:
            return self._ref._moment_basis(xhat)
        return self._mono(xhat, self._moment_exps)

    def _apply_dofs(self, f, df):
        """Local dofs of f(xhat) -> (n, m), with df(xhat) -> (n, m, 2) its physical gradient."""
        ref = REFERENCE_TRIANGLE_VERTICES
        fv = f(ref)
        dfv = df(ref) if self._hermite else None
        rows = []
        for i in range(3):
            rows.append(fv[i])
            if self._hermite:
                rows.extend(self._hV[i] * dfv[i].T)
        for a, b in self._edges:
            rows.extend(self._edge_moments @ f(ref[a] + self._edge_r[:, None] * (ref[b] - ref[a])))
        rows.extend(self._cell_moments @ f(self._cell_x))
        return numpy.array(rows)

    def bind(self, element_or_vertices):
        self._bind_geometry(element_or_vertices)
        if self._ref is not None:
            self._bind_mapping()
            return
        self._A = self._apply_dofs(
            lambda x: self._mono(x, self._value_exps),
            lambda x: self._mono_grad(x, self._value_exps),
        )
        interior = slice(self.localDofs - self._ni, self.localDofs)
        self._Pi0 = solve_cls_kkt_all_rhs(self._A, self._A[interior], numpy.eye(self.localDofs)[interior])
        self._Pi1 = self._build_gradient_projector()

    def _bind_mapping(self):
        self.M = numpy.eye(self.localDofs)
        if self._hermite:
            for i in range(3):
                self.M[3 * i + 1:3 * i + 3, 3 * i + 1:3 * i + 3] = (self._ref._hV[i] / self._hV[i]) * self.J
        signs = (-1.0) ** numpy.arange(self._ne)
        for i, (edge, ref_edge) in enumerate(zip(self._edges, EDGES)):
            if edge != ref_edge:
                start = 3 * self._nv + i * self._ne
                self.M[start:start + self._ne, start:start + self._ne] = numpy.diag(signs)

    def _build_gradient_projector(self):
        """Solve (4.43) for Pi_1, stored as (2, dim M_{k-1}, localDofs)."""
        w = abs(self.detJ) * self._cell_w
        G = self._mono(self._cell_x, self._grad_exps)
        mass = G.T @ (w[:, None] * G)
        pi0 = self._mono(self._cell_x, self._value_exps) @ self._Pi0
        rhs = -numpy.einsum("q,qgd,qj->dgj", w, self._mono_grad(self._cell_x, self._grad_exps), pi0)

        ref = REFERENCE_TRIANGLE_VERTICES
        k, nv, ne = self.order, self._nv, self._ne
        for i, (s, t) in enumerate(self._edges):
            ev = self.J @ (ref[t] - ref[s])
            normal = numpy.array([ev[1], -ev[0]])
            if normal @ (self.xE - self.x0 - self.J @ ref[s]) > 0.0:
                normal = -normal

            S = numpy.zeros((k + 1, self.localDofs))
            S[0, nv * s] = S[1, nv * t] = 1.0
            if self._hermite:
                S[2, nv * s + 1:nv * s + 3] = ev / self._hV[s]
                S[3, nv * t + 1:nv * t + 3] = ev / self._hV[t]
            S[k + 1 - ne:, 3 * nv + i * ne:3 * nv + (i + 1) * ne] = numpy.eye(ne)

            xe = ref[s] + self._edge_r[:, None] * (ref[t] - ref[s])
            flux = self._edge_w[:, None] * self._mono(xe, self._grad_exps)
            rhs += normal[:, None, None] * (flux.T @ self._edge_trace @ S)

        return numpy.linalg.solve(mass, rhs)

    def evaluateLocal(self, x):
        if self._ref is not None:
            return self.M @ self._ref.evaluateLocal(x)
        return self._mono(numpy.asarray(x, dtype=float)[None], self._value_exps)[0] @ self._Pi0

    def evaluateLocalGradient(self, x):
        if self._ref is not None:
            return self.M @ self._ref.evaluateLocalGradient(x) @ self.Jinv
        return (self._mono(numpy.asarray(x, dtype=float)[None], self._grad_exps)[0] @ self._Pi1).T

    def localProjectorDofs(self):
        if self._ref is not None:
            return numpy.linalg.solve(self.M.T, self._ref.localProjectorDofs() @ self.M.T)
        return self._A @ self._Pi0

    def interpolate(self, gf):
        if self._hermite and not hasattr(gf, "jacobian"):
            raise NotImplementedError(
                "Hermite-type VEM interpolation needs gf.jacobian(e, x) returning the physical gradient."
            )
        dofs = numpy.zeros(len(self.mapper))
        for e in self.view.elements:
            self._bind_geometry(e)
            dofs[self.mapper(e)] = self._apply_dofs(
                lambda xs: numpy.array([[float(gf(e, x))] for x in xs]),
                lambda xs: numpy.array([numpy.asarray(gf.jacobian(e, x), dtype=float).reshape(1, 2) for x in xs]),
            )[:, 0]
        return dofs
