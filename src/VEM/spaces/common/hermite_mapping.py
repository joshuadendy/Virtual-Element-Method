"""Hermite mapping helpers."""

import numpy


def build_cubic_hermite_transform(J):
    """
    Transform matrix for the cubic Hermite triangle where only the
    derivative 2x2 vertex blocks change under affine mapping.
    """
    M = numpy.eye(10, dtype=float)
    for base in (0, 3, 6):
        sl = slice(base + 1, base + 3)
        M[sl, sl] = J
    return M

