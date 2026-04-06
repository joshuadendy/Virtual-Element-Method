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


def build_k3_mapped_transform(J, hV_hat, hV_local, constraint_transform=None):
    """
    Transform matrix for the k=3 mapped Hermite-VEM basis.

    The first nine slots correspond to the standard cubic Hermite vertex value and
    scaled-gradient dofs. The final three slots correspond to the interior moment
    functionals.
    """
    M = numpy.eye(12, dtype=float)

    for i, base in enumerate((0, 3, 6)):
        sl = slice(base + 1, base + 3)
        M[sl, sl] = (hV_hat[i] / hV_local[i]) * J

    if constraint_transform is not None:
        block = numpy.asarray(constraint_transform, dtype=float)
        if block.shape != (3, 3):
            raise ValueError("constraint_transform must have shape (3, 3).")
        M[9:12, 9:12] = block

    return M
