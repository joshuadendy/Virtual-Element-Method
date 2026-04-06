"""Scaled monomial helpers."""

from math import comb

import numpy

P1_EXPONENTS = [
    (0, 0),
    (1, 0), (0, 1),
]

P2_EXPONENTS = [
    (0, 0),
    (1, 0), (0, 1),
    (2, 0), (1, 1), (0, 2),
]

P3_EXPONENTS = [
    (0, 0),
    (1, 0), (0, 1),
    (2, 0), (1, 1), (0, 2),
    (3, 0), (2, 1), (1, 2), (0, 3),
]


def monomials(x, exponents):
    xx = float(x[0])
    yy = float(x[1])
    return numpy.array([(xx ** a) * (yy ** b) for a, b in exponents], dtype=float)


def monomial_gradients(x, exponents):
    xx = float(x[0])
    yy = float(x[1])
    dx = numpy.zeros(len(exponents), dtype=float)
    dy = numpy.zeros(len(exponents), dtype=float)

    for i, (a, b) in enumerate(exponents):
        if a > 0:
            dx[i] = a * (xx ** (a - 1)) * (yy ** b)
        if b > 0:
            dy[i] = b * (xx ** a) * (yy ** (b - 1))

    return dx, dy


def scaled_coords(x, x_center, h):
    y = (numpy.asarray(x, dtype=float) - numpy.asarray(x_center, dtype=float)) / h
    return float(y[0]), float(y[1])


def scaled_monomials(x, x_center, h, exponents):
    sx, sy = scaled_coords(x, x_center, h)
    return numpy.array([(sx ** a) * (sy ** b) for a, b in exponents], dtype=float)


def scaled_monomial_gradients(x, x_center, h, exponents):
    sx, sy = scaled_coords(x, x_center, h)
    dx = numpy.zeros(len(exponents), dtype=float)
    dy = numpy.zeros(len(exponents), dtype=float)
    invh = 1.0 / h

    for i, (a, b) in enumerate(exponents):
        if a > 0:
            dx[i] = a * (sx ** (a - 1)) * (sy ** b) * invh
        if b > 0:
            dy[i] = b * (sx ** a) * (sy ** (b - 1)) * invh

    return dx, dy


def monomial_linear_transform_matrix(B, exponents):
    """
    Return the matrix T such that, for y = B x,

        monomials(y, exponents) = T @ monomials(x, exponents).

    The exponent list must be closed under the linear change of variables, which is
    the case for the hierarchical total-degree monomial sets used in this project.
    """
    B = numpy.asarray(B, dtype=float)
    if B.shape != (2, 2):
        raise ValueError("B must have shape (2, 2).")

    exp_to_row = {tuple(exp): i for i, exp in enumerate(exponents)}
    T = numpy.zeros((len(exponents), len(exponents)), dtype=float)

    b00 = float(B[0, 0])
    b01 = float(B[0, 1])
    b10 = float(B[1, 0])
    b11 = float(B[1, 1])

    for row, (a, b) in enumerate(exponents):
        for i in range(a + 1):
            coeff_x = comb(a, i) * (b00 ** i) * (b01 ** (a - i))
            for j in range(b + 1):
                coeff = coeff_x * comb(b, j) * (b10 ** j) * (b11 ** (b - j))
                out_exp = (i + j, (a - i) + (b - j))
                try:
                    col = exp_to_row[out_exp]
                except KeyError as exc:
                    raise ValueError(
                        "Exponent list is not closed under the requested transform; "
                        f"missing exponent {out_exp}."
                    ) from exc
                T[row, col] += coeff

    return T


def scaled_monomial_inverse_pullback_matrix(Jinv, h, h_hat, exponents):
    """
    Return the matrix T such that

        scaled_monomials(F^{-1}(x), xE_hat, h_hat, exponents)
        = T @ scaled_monomials(x, xE, h, exponents)

    for an affine map F(hat x) = J hat x + b. Since the scaled coordinates are
    barycentre-based, the transform is purely linear and does not mix polynomial
    degrees.
    """
    B = (float(h) / float(h_hat)) * numpy.asarray(Jinv, dtype=float)
    return monomial_linear_transform_matrix(B, exponents)
