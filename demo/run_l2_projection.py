import numpy
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
from dune.grid import cartesianDomain, gridFunction
from VEM.assembly import assemble_l2_projection
from VEM import compare_projectors, error, mesh_size, plot_eoc_curves
from VEM import (
    LinearLagrangeSpace,
    QuadraticLagrangeSpace,
    CubicHermiteSpace,
    CubicHermiteMappedVEMSpace,
    CubicHermitePhysicalVEMSpace,
    LinearLagrangeMappedVEMSpace,
    LinearLagrangePhysicalVEMSpace
)

# Use a triangular grid for demo
from dune.alugrid import aluConformGrid


def run_projection_demo(
    spaces=(CubicHermiteMappedVEMSpace,),
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

        _, view = build_demo_view()
        u = make_test_function(view)

        if plot:
            u.plot(level=3)

        old_pde_error = None
        history = []

        for _ in range(refinements):
            space = space_type(view)
            quad_order = 4 if space.localDofs == 3 else 6
            h = mesh_size(view)

            print(
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
            print("  pde problem:", err, eoc)
            old_pde_error = err

            view.hierarchicalGrid.globalRefine(2)

        histories[space_type.__name__] = history
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
        space_global = CubicHermitePhysicalVEMSpace(compare_view)
        space_mapped = CubicHermiteMappedVEMSpace(compare_view)
        return compare_projectors(
            space_global,
            space_mapped,
            quad_order=8,
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
            LinearLagrangePhysicalVEMSpace,
            LinearLagrangeMappedVEMSpace,
            CubicHermitePhysicalVEMSpace,
            CubicHermiteMappedVEMSpace,
        ),
        compare_mapped=False,
        refinements=3,
        plot=False,
        plot_eoc=False,
    )
    plt.show()
