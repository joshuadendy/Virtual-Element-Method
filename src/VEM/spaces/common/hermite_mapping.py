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


def build_k3_mapped_transform(J, Jinv, hV_hat, hV_local, hE, hE_hat):
    """
    Transform matrix for the k=3 mapped Hermite-VEM basis.
    """
    M = numpy.eye(12, dtype=float)

    for i, base in enumerate((0, 3, 6)):
        sl = slice(base + 1, base + 3)
        M[sl, sl] = (hV_hat[i] / hV_local[i]) * J

    M[9, 9] = 1.0
    M[10:12, 10:12] = (hE / hE_hat) * Jinv.T
    return M