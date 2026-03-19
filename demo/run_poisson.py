import numpy
import scipy.sparse.linalg
from dune.grid import cartesianDomain, gridFunction
from dune.alugrid import aluConformGrid

from VEM import (
    CubicHermiteMappedVEMSpace,
    CubicHermitePhysicalVEMSpace,
    CubicHermiteSpace,
    LinearLagrangeMappedVEMSpace,
    LinearLagrangePhysicalVEMSpace,
    LinearLagrangeSpace,
    QuadraticLagrangeSpace,
    apply_dirichlet,
    assemble_poisson,
    projected_error,
)


def run_poisson_demo(
    spaces=(LinearLagrangeMappedVEMSpace,),
    refinements=3,
    plot=False,
    stabilization="auto",
    stabilization_scale=1.0,
):
    def build_demo_view(nx=8, ny=8):
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
            indices = space.mapper(e)
            phi_vals = numpy.asarray(space.evaluateLocal(x), dtype=float)
            return float(dofs[indices].dot(phi_vals))

        return uh

    def boundary_dofs(space, tol=1e-12):
        ids = set()
        for e in space.view.elements:
            idx = numpy.asarray(space.mapper(e), dtype=int)
            for ldof, xhat in enumerate(space.points):
                xphys = e.geometry.toGlobal(xhat)
                if (
                    abs(xphys[0]) < tol
                    or abs(xphys[0] - 1.0) < tol
                    or abs(xphys[1]) < tol
                    or abs(xphys[1] - 1.0) < tol
                ):
                    ids.add(int(idx[ldof]))
        return numpy.array(sorted(ids), dtype=int)

    for space_type in spaces:
        print("Testing space:", space_type.__name__)

        _, view = build_demo_view()
        u = make_exact_solution(view)
        f = make_rhs(view)

        old_err = None
        for _ in range(refinements):
            space = space_type(view)
            quad_order = 8 if space.localDofs >= 10 else (6 if space.localDofs > 3 else 4)

            print(
                "number of elements:", view.size(0),
                "number of dofs:", len(space.mapper),
            )

            rhs, matrix = assemble_poisson(
                space,
                f,
                quad_order=quad_order,
                stabilization=stabilization,
                stabilization_scale=stabilization_scale,
            )

            exact_dofs = space.interpolate(u)
            bdy = boundary_dofs(space)
            rhs_bc, matrix_bc = apply_dirichlet(matrix, rhs, bdy, exact_dofs[bdy])

            dofs = scipy.sparse.linalg.spsolve(matrix_bc, rhs_bc)
            uh = make_projected_function(view, space, dofs)

            if plot:
                uh.plot(level=1)

            err = projected_error(space, dofs, u, quad_order=max(quad_order, 6))
            if old_err is not None:
                eoc = [numpy.log(old / new) / numpy.log(2.0) for old, new in zip(old_err, err)]
            else:
                eoc = None

            print("  projected [L2, H1-semi]:", err, eoc)
            old_err = err

            view.hierarchicalGrid.globalRefine(1)
        print()


if __name__ == "__main__":
    run_poisson_demo(
        spaces=(
            LinearLagrangeSpace,
            QuadraticLagrangeSpace,
            CubicHermiteSpace,
            LinearLagrangeMappedVEMSpace,
            LinearLagrangePhysicalVEMSpace,
            CubicHermiteMappedVEMSpace,
            CubicHermitePhysicalVEMSpace,
        ),
        refinements=3,
        plot=False,
    )