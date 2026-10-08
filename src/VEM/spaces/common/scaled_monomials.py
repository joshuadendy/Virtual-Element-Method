"""Scaled monomial helpers."""

import numpy


def total_degree_exponents(order):
    """Return hierarchical total-degree exponents up to the given order."""
    exponents = []
    for degree in range(order + 1):
        for a in range(degree, -1, -1):
            b = degree - a
            exponents.append((a, b))
    return exponents


P0_EXPONENTS = total_degree_exponents(0)
P1_EXPONENTS = total_degree_exponents(1)
P2_EXPONENTS = total_degree_exponents(2)
P3_EXPONENTS = total_degree_exponents(3)
P4_EXPONENTS = total_degree_exponents(4)


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


def scaled_monomials(x, x_center, h, exponents):
    """Scaled monomials ((x - x_center) / h)^alpha for points x of shape (..., 2)."""
    y = (numpy.asarray(x, dtype=float) - x_center) / h
    E = numpy.array(exponents, dtype=int).reshape(-1, 2)
    return numpy.prod(y[..., None, :] ** E, axis=-1)


def scaled_monomial_gradients(x, x_center, h, exponents):
    """Gradients of scaled_monomials, with shape (..., len(exponents), 2)."""
    y = (numpy.asarray(x, dtype=float) - x_center) / h
    E = numpy.array(exponents, dtype=int).reshape(-1, 2)
    grads = []
    for d in range(2):
        Ed = E.copy()
        Ed[:, d] = numpy.maximum(E[:, d] - 1, 0)
        grads.append(E[:, d] * numpy.prod(y[..., None, :] ** Ed, axis=-1) / h)
    return numpy.stack(grads, axis=-1)
