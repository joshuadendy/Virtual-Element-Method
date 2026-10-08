import numpy


def unit_square_boundary_dofs(space, tol=1e-12):
    """
    Return the sorted Dirichlet dof ids of a space on the unit square.

    For pure value spaces:
      all boundary-associated local dofs are clamped.

    For Hermite spaces on the unit square:
      - clamp boundary vertex value dofs
      - clamp tangential derivative dofs on boundary edges
        * horizontal edges -> dx dof
        * vertical edges   -> dy dof
      - clamp boundary edge-value / edge-average dofs when present

    This is correct for the unit square because the boundary tangents are
    coordinate-aligned. On a general polygon, tangential constraints would be
    linear combinations of dx/dy dofs instead.
    """
    def on_boundary(x, y):
        return abs(x) < tol or abs(x - 1.0) < tol or abs(y) < tol or abs(y - 1.0) < tol

    ids = set()

    if space.element == "hermite":
        edge_slots = range(9, space.localDofs)

        for e in space.view.elements:
            idx = numpy.asarray(space.mapper(e), dtype=int)

            for base in (0, 3, 6):
                x, y = e.geometry.toGlobal(numpy.asarray(space.points[base], dtype=float))

                if on_boundary(x, y):
                    ids.add(int(idx[base]))

                if abs(y) < tol or abs(y - 1.0) < tol:
                    ids.add(int(idx[base + 1]))

                if abs(x) < tol or abs(x - 1.0) < tol:
                    ids.add(int(idx[base + 2]))

            for slot in edge_slots:
                x, y = e.geometry.toGlobal(numpy.asarray(space.points[slot], dtype=float))
                if on_boundary(x, y):
                    ids.add(int(idx[slot]))

    else:
        for e in space.view.elements:
            idx = numpy.asarray(space.mapper(e), dtype=int)
            for ldof, xhat in enumerate(space.points):
                x, y = e.geometry.toGlobal(numpy.asarray(xhat, dtype=float))
                if on_boundary(x, y):
                    ids.add(int(idx[ldof]))

    return numpy.array(sorted(ids), dtype=int)
