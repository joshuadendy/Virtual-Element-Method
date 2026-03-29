import numpy
import scipy.sparse
from dune.geometry import quadratureRule


def assemble_poisson(space, force, quad_order=6, stabilization="auto", stabilization_scale=1.0):
    """
    Assemble a projected Poisson/Laplace operator using the space's local value
    projector and local projected gradients.

    Parameters
    ----------
    space :
        Space with methods bind(e), evaluateLocal(xhat), evaluateLocalGradient(xhat),
        mapper, localDofs, and view.
    force :
        Grid function or callable evaluated as force(e, xhat).
    quad_order : int
        Triangle quadrature order.
    stabilization : {"auto", "none"} or bool
        In "auto" mode, a dof-space stabilization is added whenever the space
        provides localProjectorDofs().
    stabilization_scale : float
        Scalar multiplier on the stabilization term.

    Returns
    -------
    rhs, matrix : ndarray, csr_matrix
    """
    rhs = numpy.zeros(len(space.mapper), dtype=float)

    local_entries = space.localDofs
    global_entries = local_entries ** 2 * space.view.size(0)

    value = numpy.zeros(global_entries, dtype=float)
    row_index = numpy.zeros(global_entries, dtype=int)
    col_index = numpy.zeros(global_entries, dtype=int)

    local_matrix = numpy.zeros((local_entries, local_entries), dtype=float)
    start = 0

    for e in space.view.elements:
        geo = e.geometry
        space.bind(e)
        indices = numpy.asarray(space.mapper(e), dtype=int)

        local_matrix.fill(0.0)
        for p in quadratureRule(e.type, quad_order):
            xhat = p.position
            w = float(p.weight * geo.integrationElement(xhat))

            phi_vals = numpy.asarray(space.evaluateLocal(xhat), dtype=float).reshape(-1)
            grad_vals = numpy.asarray(space.evaluateLocalGradient(xhat), dtype=float)

            rhs[indices] += w * float(force(e, xhat)) * phi_vals
            local_matrix += w * grad_vals.dot(grad_vals.T)

        stab = _build_local_stabilization(
            space,
            local_matrix,
            mode=stabilization,
            scale=stabilization_scale,
        )
        if stab is not None:
            local_matrix += stab

        for i in range(local_entries):
            for j in range(local_entries):
                entry = start + i * local_entries + j
                value[entry] = local_matrix[i, j]
                row_index[entry] = indices[i]
                col_index[entry] = indices[j]
        start += local_entries ** 2

    matrix = scipy.sparse.csr_matrix(
        (value, (row_index, col_index)),
        shape=(len(space.mapper), len(space.mapper)),
    )
    return rhs, matrix


def _build_local_stabilization(space, local_consistency, mode="auto", scale=1.0):
    if mode in (False, None, "none"):
        return None

    if mode != "auto":
        raise ValueError(f"Unknown stabilization mode {mode!r}")

    if not hasattr(space, "localProjectorDofs"):
        return None

    P = numpy.asarray(space.localProjectorDofs(), dtype=float)
    I = numpy.eye(space.localDofs, dtype=float)

    alpha = float(numpy.trace(local_consistency)) / max(space.localDofs, 1)
    if abs(alpha) < 1e-14:
        alpha = 1.0

    return scale * alpha * (I - P).T.dot(I - P)


def apply_dirichlet(matrix, rhs, dof_ids, values):
    """
    Apply strong Dirichlet conditions by symmetric row/column elimination.
    """
    dof_ids = numpy.asarray(dof_ids, dtype=int).reshape(-1)
    values = numpy.asarray(values, dtype=float).reshape(-1)
    if dof_ids.size != values.size:
        raise ValueError("dof_ids and values must have the same length.")

    A = matrix.tolil(copy=True)
    b = numpy.asarray(rhs, dtype=float).copy()

    for dof, val in zip(dof_ids, values):
        column = A[:, dof].toarray().reshape(-1)
        b -= column * val

    for dof, val in zip(dof_ids, values):
        A[:, dof] = 0.0
        A[dof, :] = 0.0
        A[dof, dof] = 1.0
        b[dof] = val

    return b, A.tocsr()