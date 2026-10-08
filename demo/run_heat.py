from functools import partial
import numpy
import matplotlib.pyplot as plt
import time
from dune.grid import cartesianDomain, gridFunction
from dune.alugrid import aluConformGrid

from VEM import (
    FEMSpace,
    VEMSpace,
    assemble_l2_projection,
    assemble_poisson,
    mesh_size,
    plot_eoc_curves,
    projected_error,
    solve_heat,
    unit_square_boundary_dofs,
)


def run_heat_demo(
    spaces={"lagrange k=1 mapped VEM": partial(VEMSpace, order=1, mapped=True)},
    refinements=3,
    final_time=1.0,
    steps=8,
    theta=0.5,
    stabilisation_scale=1.0,
    plot=False,
    plot_eoc=False,
    nx0=8,
    ny0=8,
):
    """
    Solve u_t - Laplace u = f on the unit square up to final_time with the
    manufactured solution u = (1 + t) sin(pi x) sin(pi y).

    The solution is linear in t, which the theta-scheme integrates exactly, so
    the errors at final_time show the spatial convergence rates even for a few
    steps; the only time error comes from the O(h^{k+1}) transient left by
    interpolating the initial data.
    """
    def build_demo_view(level):
        domain = cartesianDomain([0, 0], [1, 1], [nx0 * 2 ** level, ny0 * 2 ** level])
        return aluConformGrid(domain)

    def make_profile(view, scale=1.0):
        """scale * sin(pi x) sin(pi y) with its gradient attached as jacobian."""
        @gridFunction(view)
        def s(p):
            x, y = p
            return scale * numpy.sin(numpy.pi * x) * numpy.sin(numpy.pi * y)

        def jacobian(e, xhat):
            x, y = e.geometry.toGlobal(xhat)
            return scale * numpy.pi * numpy.array([
                numpy.cos(numpy.pi * x) * numpy.sin(numpy.pi * y),
                numpy.sin(numpy.pi * x) * numpy.cos(numpy.pi * y),
            ])

        s.jacobian = jacobian
        return s

    def make_projected_function(view, space, dofs):
        @gridFunction(view)
        def uh(e, x):
            space.bind(e)
            phi_vals = numpy.asarray(space.evaluateLocal(x), dtype=float).reshape(-1)
            return float(dofs[space.mapper(e)].dot(phi_vals))

        return uh

    dt = final_time / steps
    histories = {}

    for name, make_space in spaces.items():
        print("Testing space:", name)
        space_start = time.perf_counter()
        old_err = None
        history = []

        for level in range(refinements):
            level_start = time.perf_counter()
            view = build_demo_view(level)
            s = make_profile(view)
            space = make_space(view)
            h = mesh_size(view)
            quad_order = 2 * space.order + 2

            print(
                "level ", level, ":",
                "number of elements:", view.size(0),
                "number of dofs:", len(space.mapper),
                "mesh size h:", h,
            )

            # f = (1 + 2 pi^2 (1 + t)) s and u = (1 + t) s, so the load vector and
            # the Dirichlet values are time multiples of those of s.
            load_s, stiffness = assemble_poisson(
                space, s, quad_order=quad_order, stabilisation_scale=stabilisation_scale,
            )
            _, mass = assemble_l2_projection(
                space, s, quad_order, stabilisation="auto", stabilisation_scale=stabilisation_scale,
            )
            s_dofs = space.interpolate(s)
            bdy_ids = unit_square_boundary_dofs(space)

            dofs = solve_heat(
                mass,
                stiffness,
                s_dofs,
                load=lambda t: (1.0 + 2.0 * numpy.pi ** 2 * (1.0 + t)) * load_s,
                dirichlet=lambda t: (1.0 + t) * s_dofs[bdy_ids],
                dirichlet_ids=bdy_ids,
                dt=dt,
                steps=steps,
                theta=theta,
            )

            if plot:
                fig = plt.figure(figsize=(7, 5))
                make_projected_function(view, space, dofs).plot(level=2, figure=fig)
                fig.suptitle(f"Approximate solution at t = {final_time}")
                plt.show()

            err = projected_error(space, dofs, make_profile(view, 1.0 + final_time), quad_order=max(quad_order, 6))
            history.append({"h": h, "errors": err})

            eoc = None if old_err is None else [
                numpy.log(old / new) / numpy.log(2.0) for old, new in zip(old_err, err)
            ]
            print("  projected [L2, H1-semi] at final time:", err, eoc)
            print(f"  runtime: {time.perf_counter() - level_start:.3f} s")
            old_err = err

        histories[name] = history
        print(f"Total runtime for {name}: {time.perf_counter() - space_start:.3f} s")
        print()

    if plot_eoc:
        plot_eoc_curves(histories, component_names=("L2", "H1-semi"), title_prefix="Heat equation convergence")

    return histories


if __name__ == "__main__":
    run_heat_demo(
        spaces={
            **{
                f"{element} k={k} FEM": partial(FEMSpace, order=k, element=element)
                for element, k in (("lagrange", 1), ("lagrange", 2), ("hermite", 3), ("hermite", 4))
            },
            **{
                f"{element} k={k} {'mapped' if mapped else 'physical'} VEM":
                    partial(VEMSpace, order=k, element=element, mapped=mapped)
                for element, k in (("lagrange", 1), ("lagrange", 2), ("hermite", 3), ("hermite", 4))
                for mapped in (False, True)
            },
        },
        refinements=3,
        plot=False,
        plot_eoc=False,
    )
    plt.show()
