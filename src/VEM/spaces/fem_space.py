import numpy

from .base import SpaceBase
from .common.scaled_monomials import scaled_monomial_gradients, scaled_monomials, total_degree_exponents
from .common.triangle_geometry import EDGES, REFERENCE_TRIANGLE_VERTICES, bind_affine_triangle


class FEMSpace(SpaceBase):
    """
    Order-k C0 finite element on triangles with a nodal basis of P_k.

    element="lagrange" (k >= 1): values at the order-k lattice points.
    element="hermite" (k >= 3): u and grad u at the vertices, k-3 equispaced
    values per edge and (k-1)(k-2)/2 interior values.

    The interior nodes are the P_{k-3} principal lattice of an inner triangle,
    at barycentric coordinates (1 + s alpha) / (3 + s(k-3)) with |alpha| = k-3:
    s = 1 gives the interior order-k lattice (Lagrange), and s = 2 gives the
    barycentre at k = 3 and (1/5, 1/5), (3/5, 1/5), (1/5, 3/5) at k = 4 (Hermite).

    The nodal basis is built once on the reference triangle and mapped by
    Phi = M F^*(hat Phi) with M = V^T (Section 3.3). V is the identity apart from
    the J^T vertex gradient blocks and a reversal of the nodes on edges that are
    parametrised against the reference orientation.

    Local dof ordering (matching the DUNE mapper): vertices, edges (0,1), (0,2),
    (1,2) with nodes ordered from the lower global vertex index, then interior.
    """

    def __init__(self, view, order, element="lagrange"):
        if element not in ("lagrange", "hermite"):
            raise ValueError(f"Unknown element type {element!r}.")
        self._hermite = element == "hermite"
        if order < (3 if self._hermite else 1):
            raise ValueError(f"order {order} is too low for a {element} finite element.")
        self.view = view
        self.dim = view.dimension
        self.order = order
        self.element = element
        self._nv = 3 if self._hermite else 1
        self._ne = order - 3 if self._hermite else order - 1
        self._ni = (order - 1) * (order - 2) // 2
        self.localDofs = 3 * (self._nv + self._ne) + self._ni
        self.layout = lambda gt: (self._nv, self._ne, self._ni)[gt.dim]
        self.mapper = view.mapper(self.layout)

        ref = REFERENCE_TRIANGLE_VERTICES
        self._edge_r = numpy.arange(1, self._ne + 1) / (self._ne + 1)
        s, n = (2 if self._hermite else 1), order - 3
        self._interior = numpy.array(
            [[1 + s * a, 1 + s * b] for b in range(n + 1) for a in range(n + 1 - b)], dtype=float
        ).reshape(-1, 2) / (3 + s * n)
        self.points = numpy.vstack([
            numpy.repeat(ref, self._nv, axis=0),
            *(ref[a] + self._edge_r[:, None] * (ref[b] - ref[a]) for a, b in EDGES),
            self._interior,
        ])

        self._exps = total_degree_exponents(order)
        self._coeffs = numpy.linalg.inv(self._apply_dofs(
            lambda x: scaled_monomials(x, 0.0, 1.0, self._exps),
            lambda x: scaled_monomial_gradients(x, 0.0, 1.0, self._exps),
            EDGES,
        ))
        self.bind(ref)

    def _apply_dofs(self, f, df, edges):
        """Local dofs of f(xhat) -> (n, m), with df(xhat) -> (n, m, 2) its physical gradient."""
        ref = REFERENCE_TRIANGLE_VERTICES
        fv = f(ref)
        dfv = df(ref) if self._hermite else None
        rows = []
        for i in range(3):
            rows.append(fv[i])
            if self._hermite:
                rows.extend(dfv[i].T)
        for a, b in edges:
            rows.extend(f(ref[a] + self._edge_r[:, None] * (ref[b] - ref[a])))
        rows.extend(f(self._interior))
        return numpy.array(rows)

    def bind(self, element_or_vertices):
        data = bind_affine_triangle(element_or_vertices)
        self.x0, self.J, self.Jinv = data["x0"], data["J"], data["Jinv"]

        if hasattr(element_or_vertices, "geometry"):
            idx = self.mapper(element_or_vertices)
            gv = [int(idx[self._nv * i]) for i in range(3)]
        else:
            gv = [0, 1, 2]
        self._edges = [(a, b) if gv[a] < gv[b] else (b, a) for a, b in EDGES]

        self.M = numpy.eye(self.localDofs)
        if self._hermite:
            for i in range(3):
                self.M[3 * i + 1:3 * i + 3, 3 * i + 1:3 * i + 3] = self.J
        for i, (edge, ref_edge) in enumerate(zip(self._edges, EDGES)):
            if edge != ref_edge:
                start = 3 * self._nv + i * self._ne
                self.M[start:start + self._ne, start:start + self._ne] = numpy.eye(self._ne)[::-1]

    def evaluateLocal(self, x):
        return self.M @ (scaled_monomials(x, 0.0, 1.0, self._exps) @ self._coeffs)

    def evaluateLocalGradient(self, x):
        return self.M @ (self._coeffs.T @ scaled_monomial_gradients(x, 0.0, 1.0, self._exps)) @ self.Jinv

    def interpolate(self, gf):
        if self._hermite and not hasattr(gf, "jacobian"):
            raise NotImplementedError(
                "Hermite interpolation needs gf.jacobian(e, x) returning the physical gradient."
            )
        dofs = numpy.zeros(len(self.mapper))
        for e in self.view.elements:
            self.bind(e)
            dofs[self.mapper(e)] = self._apply_dofs(
                lambda xs: numpy.array([[float(gf(e, x))] for x in xs]),
                lambda xs: numpy.array([numpy.asarray(gf.jacobian(e, x), dtype=float).reshape(1, 2) for x in xs]),
                self._edges,
            )[:, 0]
        return dofs
