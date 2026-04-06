"""Constrained least-squares projector helpers."""

import numpy


def solve_cls_kkt_all_rhs(A, C, G):
    """
    Solve, for all right-hand sides at once,

    \min_{c_j} |A c_j - e_j|^2
    subject to
    C c_j = G_{:,j},

    returning the coefficient matrix X whose columns are the c_j.
    """
    ATA = A.T.dot(A)
    K = numpy.block([
        [ATA, C.T],
        [C, numpy.zeros((C.shape[0], C.shape[0]), dtype=float)],
    ])
    RHS = numpy.vstack([
        A.T.dot(numpy.eye(A.shape[0], dtype=float)),
        G,
    ])

    try:
        sol = numpy.linalg.solve(K, RHS)
    except numpy.linalg.LinAlgError:
        sol = numpy.linalg.lstsq(K, RHS, rcond=None)[0]

    return sol[:A.shape[1], :]