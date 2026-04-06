import numpy
from dune.geometry import quadratureRule
import scipy.sparse

def assemble_l2_projection(space, force, quad_order):
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