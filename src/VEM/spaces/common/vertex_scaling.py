"""Helpers for vertex-associated effective length scales."""

import numpy

from .triangle_geometry import coerce_triangle_vertices


def build_vertex_effective_h(view, mapper, value_dof_offsets=(0, 3, 6), measure="diameter"):
    """
    Average a characteristic element length onto the global vertex-value dofs.

    Use the same reference-local vertex ordering as bind_affine_triangle so the
    Hermite vertex blocks and their associated h_v are aligned with the mapped
    basis transform.
    """
    values = numpy.zeros(len(mapper), dtype=float)
    counts = numpy.zeros(len(mapper), dtype=float)

    for e in view.elements:
        verts = coerce_triangle_vertices(e)

        e01 = numpy.linalg.norm(verts[1] - verts[0])
        e12 = numpy.linalg.norm(verts[2] - verts[1])
        e20 = numpy.linalg.norm(verts[0] - verts[2])

        if measure == "diameter":
            local_h = numpy.array([max(e01, e12, e20)] * 3, dtype=float)
        elif measure == "adjacent_edge_average":
            local_h = numpy.array([
                0.5 * (e01 + e20),
                0.5 * (e01 + e12),
                0.5 * (e12 + e20),
            ], dtype=float)
        else:
            raise ValueError(f"Unknown measure '{measure}'.")

        idx = mapper(e)
        for lv, offset in enumerate(value_dof_offsets):
            gid = int(idx[offset])
            values[gid] += local_h[lv]
            counts[gid] += 1.0

    mask = counts > 0
    values[mask] /= counts[mask]
    values[~mask] = 1.0
    return values