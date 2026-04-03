import numpy
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import time
from dune.grid import cartesianDomain, gridFunction
from VEM.assembly import assemble_l2_projection
from VEM import compare_projectors, error, mesh_size, plot_eoc_curves
from VEM import (
    LinearLagrangeSpace,
    QuadraticLagrangeSpace,
    CubicHermiteSpace,
    QuarticHermiteSpace,
    CubicHermiteMappedVEMSpace,
    CubicHermitePhysicalVEMSpace,
    QuarticHermiteMappedVEMSpace,
    QuarticHermitePhysicalVEMSpace,
    LinearLagrangeMappedVEMSpace,
    LinearLagrangePhysicalVEMSpace,
    QuadraticLagrangeMappedVEMSpace,
    QuadraticLagrangePhysicalVEMSpace,
)

# Use a triangular grid for demo
from dune.alugrid import aluConformGrid


def run_projection_demo(
    spaces=(QuarticHermiteMappedVEMSpace,),
    refinements=3,
    plot=True,
    compare_mapped=True,
    plot_eoc=False,
    show_reference_slope=True,
):
    def build_demo_view(nx=10, ny=10):
        domain = cartesianDomain([0, 0], [1, 1], [nx, ny])
        return domain, aluConformGrid(domain)

    def make_test_function(view):
        @gridFunction(view)
        def u(p):
            x, y = p
            return numpy.cos(2 * numpy.pi * x) * numpy.cos(2 * numpy.pi * y)

        return u

    def make_projected_function(view, space, dofs):
        @gridFunction(view)
        def uh(e, x):
            space.bind(e)
            indices = space.mapper(e)
            phi_vals = numpy.asarray(space.evaluateLocal(x), dtype=float)
            local_dofs = dofs[indices]
            return numpy.dot(local_dofs, phi_vals)

        return uh

    histories = {}

    for space_type in spaces:
        print("Testing space:", space_type.__name__)
        space_start = time.perf_counter()

        _, view = build_demo_view()
        u = make_test_function(view)

        if plot:
            u.plot(level=3)

        old_pde_error = None
        history = []

        for level in range(refinements):
            level_start = time.perf_counter()
            space = space_type(view)
            if space.localDofs >= 15:
                quad_order = 10
            elif space.localDofs >= 10:
                quad_order = 8
            elif space.localDofs > 3:
                quad_order = 6
            else:
                quad_order = 4
            h = mesh_size(view)

            print(
                "level ", level, ":",
                "number of elements:", view.size(0),
                "number of dofs:", len(space.mapper),
                "mesh size h:", h,
            )

            rhs, matrix = assemble_l2_projection(space, u, quad_order=quad_order)
            dofs = scipy.sparse.linalg.spsolve(matrix, rhs)
            uh = make_projected_function(view, space, dofs)

            if plot:
                uh.plot(level=1)

            err = error(view, u, uh)
            history.append({"h": h, "errors": err})

            if old_pde_error is not None:
                eoc = [numpy.log(old_e / e) / numpy.log(2)
                       for old_e, e in zip(old_pde_error, err)]
            else:
                eoc = None
            elapsed = time.perf_counter() - level_start
            print("  pde problem:", err, eoc)
            print(f"  runtime: {elapsed:.3f} s")
            old_pde_error = err

            view.hierarchicalGrid.globalRefine(2)

        total_elapsed = time.perf_counter() - space_start
        histories[space_type.__name__] = history
        print(f"Total runtime for {space_type.__name__}: {total_elapsed:.3f} s")
        print()

    if plot_eoc:
        plot_eoc_curves(
            histories,
            component_names=("L2",),
            title_prefix="L2 projection convergence",
            show_reference=False if show_reference_slope else False,
        )

    if compare_mapped:
        _, compare_view = build_demo_view()
        space_global = QuarticHermitePhysicalVEMSpace(compare_view)
        space_mapped = QuarticHermiteMappedVEMSpace(compare_view)
        return compare_projectors(
            space_global,
            space_mapped,
            quad_order=10,
            print_per_element=False,
            compare_local_mass=True,
        )

    return histories


if __name__ == "__main__":
    run_projection_demo(
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
        compare_mapped=False,
        refinements=3,
        plot=False,
        plot_eoc=False,
    )
    plt.show()
