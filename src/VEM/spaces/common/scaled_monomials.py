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
