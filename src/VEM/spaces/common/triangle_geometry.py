"""Triangle geometry helpers."""

import numpy


REFERENCE_TRIANGLE_VERTICES = numpy.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    dtype=float,
)


def coerce_triangle_vertices(element_or_vertices):
    """
    Accept either a DUNE element or a raw (3,2) vertex array.
    Return the physical images of the REFERENCE triangle vertices in the
    reference-local order (0,0), (1,0), (0,1).

    This is important for any code that later applies J^{-T} to mapped
    gradients: the columns of J must correspond to the reference coordinate
    directions, not just an arbitrary corner ordering.
    """
    if hasattr(element_or_vertices, "geometry"):
        geo = element_or_vertices.geometry
        return numpy.array(
            [geo.toGlobal(xhat) for xhat in REFERENCE_TRIANGLE_VERTICES],
            dtype=float,
        )

    verts = numpy.asarray(element_or_vertices, dtype=float)
    if verts.shape != (3, 2):
        raise ValueError("triangle vertices must have shape (3, 2)")
    return verts


def triangle_barycentre(vertices):
    verts = coerce_triangle_vertices(vertices)
    return verts.mean(axis=0)


def triangle_diameter(vertices):
    verts = coerce_triangle_vertices(vertices)
    e01 = numpy.linalg.norm(verts[1] - verts[0])
    e12 = numpy.linalg.norm(verts[2] - verts[1])
    e20 = numpy.linalg.norm(verts[0] - verts[2])
    return max(e01, e12, e20)


def triangle_area(vertices):
    verts = coerce_triangle_vertices(vertices)
    j = numpy.column_stack((verts[1] - verts[0], verts[2] - verts[0]))
    return 0.5 * abs(numpy.linalg.det(j))


def bind_affine_triangle(element_or_vertices):
    """
    Return the standard affine-triangle data used throughout the codebase.
    """
    verts = coerce_triangle_vertices(element_or_vertices)

    x0 = verts[0].copy()
    e1 = verts[1] - verts[0]
    e2 = verts[2] - verts[0]

    j = numpy.column_stack((e1, e2))
    detj = float(numpy.linalg.det(j))
    if abs(detj) <= 1e-14:
        raise ValueError("Degenerate triangle (detJ=0)")

    jinv = numpy.linalg.inv(j)

    return {
        "verts": verts,
        "x0": x0,
        "e1": e1,
        "e2": e2,
        "J": j,
        "Jinv": jinv,
        "detJ": detj,
        "area": 0.5 * abs(detj),
        "xE": triangle_barycentre(verts),
        "hE": triangle_diameter(verts),
    }