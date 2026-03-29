import numpy
from dune.geometry import quadratureRules


def error(view, u, uh, quad_order=5):
    rules = quadratureRules(quad_order)
    l2 = 0.0
    for e in view.elements:
        geometry = e.geometry
        for p in rules(e.type):
            hatx = p.position
            weight = p.weight * geometry.integrationElement(hatx)
            l2 += (uh(e, hatx) - u(e, hatx))**2 * weight
    return [numpy.sqrt(l2)]


def projected_error(space, dofs, u, quad_order=6):
    """
    Compute errors against an exact solution using the projected basis values and
    projected basis gradients supplied by the space.

    Returns [L2_error, H1_seminorm_error].
    """
    if not hasattr(u, "jacobian"):
        raise NotImplementedError(
            "projected_error needs u.jacobian(e,x) for the exact gradient."
        )

    rules = quadratureRules(quad_order)
    l2 = 0.0
    h1 = 0.0

    for e in space.view.elements:
        geometry = e.geometry
        space.bind(e)
        indices = space.mapper(e)
        local_dofs = numpy.asarray(dofs[indices], dtype=float)

        for p in rules(e.type):
            hatx = p.position
            weight = float(p.weight * geometry.integrationElement(hatx))

            phi_vals = numpy.asarray(space.evaluateLocal(hatx), dtype=float).reshape(-1)
            grad_vals = numpy.asarray(space.evaluateLocalGradient(hatx), dtype=float)

            uh = float(local_dofs.dot(phi_vals))
            guh = local_dofs.dot(grad_vals)

            uex = float(u(e, hatx))
            guex = numpy.asarray(u.jacobian(e, hatx), dtype=float).reshape(-1)

            l2 += (uh - uex) ** 2 * weight
            h1 += float(numpy.dot(guh - guex, guh - guex)) * weight

    return [numpy.sqrt(l2), numpy.sqrt(h1)]