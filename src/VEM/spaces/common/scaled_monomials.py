"""Scaled monomial helpers."""

import numpy

P1_EXPONENTS = [
    (0, 0),
    (1, 0), (0, 1),
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