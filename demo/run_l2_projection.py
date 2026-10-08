from functools import partial
import numpy
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import time
from dune.grid import cartesianDomain, gridFunction
from VEM.assembly import assemble_l2_projection
from VEM import compare_projectors, error, mesh_size, plot_eoc_curves
from VEM import (
    FEMSpace,
    VEMSpace,
)

# Use a triangular grid for demo
from dune.alugrid import aluConformGrid


def run_projection_demo(
    spaces={"hermite k=4 mapped VEM": partial(VEMSpace, order=4, element="hermite", mapped=True)},
    refinements=3,
    plot=True,
    plot_true_solution=False,
    plot_error=False,
    compare_mapped=True,
    plot_eoc=False,
    show_reference_slope=True,
    nx0=8,
    ny0=8,
):
    def build_demo_view(nx=nx0, ny=ny0):
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
            indices = numpy.asarray(space.mapper(e), dtype=int)
            phi_vals = numpy.asarray(space.evaluateLocal(x), dtype=float).reshape(-1)
            local_dofs = dofs[indices]
            return float(local_dofs.dot(phi_vals))

        return uh

    def make_error_functions(view, u, uh):
        @gridFunction(view)
        def err(e, x):
            xg = e.geometry.toGlobal(x)
            return u(xg) - uh(e, x)

        @gridFunction(view)
        def abs_err(e, x):
            return abs(err(e, x))

        return err, abs_err

    histories = {}

    for name, make_space in spaces.items():
        print("Testing space:", name)
        space_start = time.perf_counter()

        _, view = build_demo_view()
        u = make_test_function(view)

        old_pde_error = None
        history = []

        for level in range(refinements):
            level_start = time.perf_counter()
            space = make_space(view)
            quad_order = 2 * space.order + 2
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

            if plot_true_solution:
                fig1 = plt.figure(figsize=(7, 5))
                u.plot(level=2, figure=fig1)
                fig1.suptitle("Exact solution")
                plt.show()

            if plot:
                fig2 = plt.figure(figsize=(7, 5))
                uh.plot(level=2, figure=fig2)
                fig2.suptitle("Approximate solution")
                plt.show()

            if plot_error:
                err, abs_err = make_error_functions(view, u, uh)
                fig3 = plt.figure(figsize=(7, 5))
                abs_err.plot(level=2, figure=fig3)
                fig3.suptitle("Absolute error")
                plt.show()

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
        histories[name] = history
        print(f"Total runtime for {name}: {total_elapsed:.3f} s")
        print()

    if plot_eoc:
        plot_eoc_curves(
            histories,
            component_names=("L2",),
            title_prefix="L2 projection convergence",
            show_reference=show_reference_slope,
        )

    if compare_mapped:
        _, compare_view = build_demo_view()
        space_global = VEMSpace(compare_view, 4, element="hermite")
        space_mapped = VEMSpace(compare_view, 4, element="hermite", mapped=True)
        return compare_projectors(
            space_global,
            space_mapped,
            quad_order=10,
            print_per_element=False,
            compare_local_mass=True,
        )

    return histories


if __name__ == "__main__":
    SPACES = (("lagrange", 1), ("lagrange", 2), ("hermite", 3), ("hermite", 4))
    run_projection_demo(
        spaces={
            **{
                f"{element} k={k} FEM": partial(FEMSpace, order=k, element=element)
                for element, k in SPACES
            },
            **{
                f"{element} k={k} {'mapped' if mapped else 'physical'} VEM":
                    partial(VEMSpace, order=k, element=element, mapped=mapped)
                for element, k in SPACES
                for mapped in (False, True)
            },
        },
        compare_mapped=False,
        refinements=3,
        plot=False,
        plot_true_solution=False,
        plot_error=False,
        plot_eoc=False,
    )
    plt.show()
