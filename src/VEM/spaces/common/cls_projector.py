"""Constrained least-squares projector helpers."""

import numpy


def solve_cls(A, C, G):
    """
    Solve, for all right-hand sides at once,

    min_{c_j} |A c_j - e_j|^2
    subject to
    C c_j = G_{:,j},

    returning the coefficient matrix X whose columns are the c_j.

    This is the null-space method: with C^T = Q R, the constraints fix the
    component in range(Q_1) and the least-squares problem is solved over
    range(Q_2) = ker(C). Unlike the KKT system (4.41) it never forms A^T A, so
    the solve keeps the conditioning of A rather than squaring it.
    """
    m = C.shape[0]
    Q, R = numpy.linalg.qr(C.T, mode="complete")
    Q1, Q2 = Q[:, :m], Q[:, m:]
    X0 = Q1 @ numpy.linalg.solve(R[:m].T, G)
    Z = numpy.linalg.lstsq(A @ Q2, numpy.eye(A.shape[0]) - A @ X0, rcond=None)[0]
    return X0 + Q2 @ Z
