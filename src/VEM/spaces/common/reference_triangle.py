"""Reference-triangle constants and interpolation helpers."""

import numpy

REFERENCE_TRIANGLE_VERTICES = numpy.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    dtype=float,
)
REFERENCE_TRIANGLE_BARYCENTRE = numpy.array([1.0 / 3.0, 1.0 / 3.0], dtype=float)
REFERENCE_TRIANGLE_DIAMETER = float(numpy.sqrt(2.0))

LINEAR_LAGRANGE_POINTS = REFERENCE_TRIANGLE_VERTICES.copy()
QUADRATIC_LAGRANGE_POINTS = numpy.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.0, 0.5],
        [0.5, 0.5],
    ],
    dtype=float,
)
CUBIC_HERMITE_POINTS = numpy.array(
    [
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
    ],
    dtype=float,
)
K3_HERMITE_VEM_POINTS = numpy.array(
    [
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
    ],
    dtype=float,
)


def barycentric_coords(x):
    """Return barycentric coordinates on the reference triangle."""
    xi = float(x[0])
    eta = float(x[1])
    return 1.0 - xi - eta, xi, eta


def interpolate_at_local_points(view, mapper, points, gf):
    """Interpolate a grid function by evaluating it at local points."""
    dofs = numpy.zeros(len(mapper), dtype=float)
    points_t = numpy.asarray(points, dtype=float).T
    for element in view.elements:
        dofs[mapper(element)] = gf(element, points_t)
    return dofs