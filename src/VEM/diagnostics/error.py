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