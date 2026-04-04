import numpy
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import time
from dune.grid import cartesianDomain, gridFunction
from dune.alugrid import aluConformGrid

from VEM import (
    CubicHermiteMappedVEMSpace,
    CubicHermitePhysicalVEMSpace,
    CubicHermiteSpace,
    QuarticHermiteMappedVEMSpace,
    QuarticHermitePhysicalVEMSpace,
    QuarticHermiteSpace,
    LinearLagrangeMappedVEMSpace,
    LinearLagrangePhysicalVEMSpace,
    LinearLagrangeSpace,
    QuadraticLagrangeMappedVEMSpace,
    QuadraticLagrangePhysicalVEMSpace,
    QuadraticLagrangeSpace,
    apply_dirichlet,
    assemble_poisson,
    compare_gradient_projectors,
    mesh_size,
    plot_eoc_curves,
    projected_error,
)


def run_poisson_demo(
    spaces=(LinearLagrangeMappedVEMSpace,),
    refinements=3,
    plot=False,
    stabilisation="auto",
    stabilisation_scale=1.0,
    compare_mapped=True,
    plot_eoc=False,
    show_reference_slope=True,
    nx0=8,
    ny0=8,
):
    def build_demo_view(level, nx0=nx0, ny0=ny0):
        """
        Rebuild a fresh structured grid on each level.
        """
        nx = nx0 * (2 ** level)
        ny = ny0 * (2 ** level)
        domain = cartesianDomain([0, 0], [1, 1], [nx, ny])
        return domain, aluConformGrid(domain)

    def make_exact_solution(view):
        @gridFunction(view)
        def u(p):
            x, y = p
            return numpy.sin(numpy.pi * x) * numpy.sin(numpy.pi * y)

        def jacobian(e, xhat):
            x, y = e.geometry.toGlobal(xhat)
            return numpy.array([
                numpy.pi * numpy.cos(numpy.pi * x) * numpy.sin(numpy.pi * y),
                numpy.pi * numpy.sin(numpy.pi * x) * numpy.cos(numpy.pi * y),
            ], dtype=float)

        u.jacobian = jacobian
        return u

    def make_rhs(view):
        @gridFunction(view)
        def f(p):
            x, y = p
            return 2.0 * (numpy.pi ** 2) * numpy.sin(numpy.pi * x) * numpy.sin(numpy.pi * y)

        return f

    def make_projected_function(view, space, dofs):
        @gridFunction(view)
        def uh(e, x):
            space.bind(e)
            indices = numpy.asarray(space.mapper(e), dtype=int)
            phi_vals = numpy.asarray(space.evaluateLocal(x), dtype=float).reshape(-1)
            return float(dofs[indices].dot(phi_vals))

        return uh

    def boundary_dofs_and_values(space, exact_dofs, tol=1e-12):
        """
        Return Dirichlet dof ids and values.

        For pure value spaces:
          all boundary-associated local dofs are clamped.

        For Hermite spaces on the unit square:
          - clamp boundary vertex value dofs
          - clamp tangential derivative dofs on boundary edges
            * horizontal edges -> dx dof
            * vertical edges   -> dy dof
          - clamp boundary edge-value / edge-average dofs when present

        This is correct for the current square-domain demo because the boundary
        tangents are coordinate-aligned. On a general polygon, tangential
        constraints would be linear combinations of dx/dy dofs instead.
        """
        ids = set()

        if space.localDofs in (10, 12, 15, 18):
            edge_slots = []
            if space.localDofs == 15:
                edge_slots = [9, 10, 11]
            elif space.localDofs == 18:
                edge_slots = [9, 10, 11]

            for e in space.view.elements:
                idx = numpy.asarray(space.mapper(e), dtype=int)

                for base in (0, 3, 6):
                    xhat = numpy.asarray(space.points[base], dtype=float)
                    x, y = e.geometry.toGlobal(xhat)

                    on_left = abs(x) < tol
                    on_right = abs(x - 1.0) < tol
                    on_bottom = abs(y) < tol
                    on_top = abs(y - 1.0) < tol

                    if on_left or on_right or on_bottom or on_top:
                        ids.add(int(idx[base]))

                    if on_bottom or on_top:
                        ids.add(int(idx[base + 1]))

                    if on_left or on_right:
                        ids.add(int(idx[base + 2]))

                for slot in edge_slots:
                    xhat = numpy.asarray(space.points[slot], dtype=float)
                    x, y = e.geometry.toGlobal(xhat)
                    if (
                        abs(x) < tol
                        or abs(x - 1.0) < tol
                        or abs(y) < tol
                        or abs(y - 1.0) < tol
                    ):
                        ids.add(int(idx[slot]))

        else:
            for e in space.view.elements:
                idx = numpy.asarray(space.mapper(e), dtype=int)
                for ldof, xhat in enumerate(space.points):
                    x, y = e.geometry.toGlobal(numpy.asarray(xhat, dtype=float))
                    if (
                        abs(x) < tol
                        or abs(x - 1.0) < tol
                        or abs(y) < tol
                        or abs(y - 1.0) < tol
                    ):
                        ids.add(int(idx[ldof]))

        ids = numpy.array(sorted(ids), dtype=int)
        vals = numpy.asarray(exact_dofs[ids], dtype=float)
        return ids, vals

    histories = {}

    for space_type in spaces:
        print("Testing space:", space_type.__name__)
        space_start = time.perf_counter()
        old_err = None
        history = []

        for level in range(refinements):
            level_start = time.perf_counter()
            _, view = build_demo_view(level)
            u = make_exact_solution(view)
            f = make_rhs(view)
            space = space_type(view)
            h = mesh_size(view)

            if space.localDofs >= 15:
                quad_order = 10
            elif space.localDofs >= 10:
                quad_order = 8
            elif space.localDofs > 3:
                quad_order = 6
            else:
                quad_order = 4

            print(
                "level ", level, ":",
                "number of elements:", view.size(0),
                "number of dofs:", len(space.mapper),
                "mesh size h:", h,
            )

            rhs, matrix = assemble_poisson(
                space,
                f,
                quad_order=quad_order,
                stabilisation=stabilisation,
                stabilisation_scale=stabilisation_scale,
            )

            exact_dofs = space.interpolate(u)
            bdy_ids, bdy_vals = boundary_dofs_and_values(space, exact_dofs)

            rhs_bc, matrix_bc = apply_dirichlet(matrix, rhs, bdy_ids, bdy_vals)
            dofs = scipy.sparse.linalg.spsolve(matrix_bc, rhs_bc)

            uh = make_projected_function(view, space, dofs)
            if plot:
                uh.plot(level=1)

            err = projected_error(space, dofs, u, quad_order=max(quad_order, 6))
            history.append({"h": h, "errors": err})

            if old_err is not None:
                eoc = [
                    numpy.log(old / new) / numpy.log(2.0)
                    for old, new in zip(old_err, err)
                ]
            else:
                eoc = None

            elapsed = time.perf_counter() - level_start
            print("  projected [L2, H1-semi]:", err, eoc)
            print(f"  runtime: {elapsed:.3f} s")
            old_err = err

        total_elapsed = time.perf_counter() - space_start
        histories[space_type.__name__] = history
        print(f"Total runtime for {space_type.__name__}: {total_elapsed:.3f} s")
        print()

    if plot_eoc:
        plot_eoc_curves(
            histories,
            component_names=("L2", "H1-semi"),
            title_prefix="Poisson convergence",
            show_reference=False if show_reference_slope else False,
        )

    if compare_mapped:
        _, compare_view = build_demo_view(level=0)
        space_physical = QuarticHermitePhysicalVEMSpace(compare_view)
        space_mapped = QuarticHermiteMappedVEMSpace(compare_view)
        return compare_gradient_projectors(
            space_physical,
            space_mapped,
            quad_order=10,
            print_per_element=False,
        )

    return histories


if __name__ == "__main__":
    run_poisson_demo(
        spaces=(
            LinearLagrangeSpace,
            QuadraticLagrangeSpace,
            CubicHermiteSpace,
            QuarticHermiteSpace,
            LinearLagrangePhysicalVEMSpace,
            LinearLagrangeMappedVEMSpace,
            QuadraticLagrangePhysicalVEMSpace,
            QuadraticLagrangeMappedVEMSpace,
            CubicHermitePhysicalVEMSpace,
            CubicHermiteMappedVEMSpace,
            QuarticHermitePhysicalVEMSpace,
            QuarticHermiteMappedVEMSpace,
        ),
        refinements=4,
        plot=False,
        compare_mapped=False,
        plot_eoc=False,
    )
    plt.show()
