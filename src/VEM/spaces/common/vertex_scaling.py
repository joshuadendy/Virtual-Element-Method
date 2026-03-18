"""Helpers for vertex-associated effective length scales."""

import numpy


def build_vertex_effective_h(view, mapper, value_dof_offsets=(0, 3, 6)):
    """
    Average a simple edge-based characteristic length onto global vertex dofs.

    This matches the usage pattern in the Hermite spaces, where the mapper
    indices at offsets (0,3,6) correspond to the three vertex-value dofs.
    """
    values = numpy.zeros(len(mapper), dtype=float)
    counts = numpy.zeros(len(mapper), dtype=float)

    for e in view.elements:
        geo = e.geometry
        verts = numpy.array([geo.corner(i) for i in range(3)], dtype=float)

        e01 = numpy.linalg.norm(verts[1] - verts[0])
        e12 = numpy.linalg.norm(verts[2] - verts[1])
        e20 = numpy.linalg.norm(verts[0] - verts[2])

        local_h = numpy.array([
            0.5 * (e01 + e20),
            0.5 * (e01 + e12),
            0.5 * (e12 + e20),
        ], dtype=float)

        idx = mapper(e)
        for lv, offset in enumerate(value_dof_offsets):
            gid = int(idx[offset])
            values[gid] += local_h[lv]
            counts[gid] += 1.0

    mask = counts > 0
    values[mask] /= counts[mask]
    values[~mask] = 1.0
    return values