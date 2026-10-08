import numpy
import scipy.sparse.linalg


def solve_heat(mass, stiffness, u0, load, dirichlet, dirichlet_ids, dt, steps, theta=0.5):
    """
    Theta-scheme for M u' + A u = F(t) with u = g(t) on dirichlet_ids:

      (M + theta dt A) u^{n+1} = (M - (1 - theta) dt A) u^n + dt (theta F^{n+1} + (1 - theta) F^n).

    load(t) returns F(t) and dirichlet(t) the values g(t). theta=1 is backward
    Euler and theta=1/2 Crank-Nicolson. The free-dof block is factorised once
    and the Dirichlet values are lifted into the right-hand side each step.

    Returns the dof vector at t = steps * dt.
    """
    dirichlet_ids = numpy.asarray(dirichlet_ids, dtype=int)
    free = numpy.setdiff1d(numpy.arange(mass.shape[0]), dirichlet_ids)

    lhs = (mass + theta * dt * stiffness).tocsr()
    explicit = (mass - (1.0 - theta) * dt * stiffness).tocsr()
    solve = scipy.sparse.linalg.splu(lhs[free][:, free].tocsc()).solve
    lift = lhs[free][:, dirichlet_ids]

    u = numpy.array(u0, dtype=float)
    load_old = load(0.0)
    for n in range(1, steps + 1):
        t = n * dt
        load_new = load(t)
        b = explicit @ u + dt * (theta * load_new + (1.0 - theta) * load_old)
        u[dirichlet_ids] = dirichlet(t)
        u[free] = solve(b[free] - lift @ u[dirichlet_ids])
        load_old = load_new
    return u
