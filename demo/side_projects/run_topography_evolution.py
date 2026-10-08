import numpy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import TwoSlopeNorm
from matplotlib.path import Path
from matplotlib.tri import Triangulation
from dune.alugrid import aluConformGrid

from VEM import VEMSpace, assemble_l2_projection, assemble_poisson, solve_heat
from run_topography import MAP, boundary_elimination, build_mesh, crescent_path, inside_land, islets, peaks, sample_matrix


def smoothstep(t, a, b):
    s = numpy.clip((t - a) / (b - a), 0.0, 1.0)
    return s * s * (3 - 2 * s)


def run_topography_evolution(
    kappa=11.0,
    sea_level=0.3,
    final_time=0.09,
    frames=90,
    steps_per_frame=2,
    crescent_ramp=(0.0, 0.015),
    islet_ramp=(0.03, 0.045),
    filename=".tmp/island_evolution.gif",
):
    """
    Grow the run_topography landscape from a flat seabed by solving

      u_t - Laplace u + kappa^2 u = f(t),

    whose steady state is the screened Poisson map of run_topography. The uplift
    is split so the crescent rises first and the islets switch on later, each
    ramped smoothly over its (start, end) window.
    """
    crescent, bay = crescent_path()
    cluster, islet_peaks = islets(bay)
    land = Path.make_compound_path(crescent, *cluster)
    V, T = build_mesh(land)
    print("elements:", len(T))

    view = aluConformGrid({"vertices": V, "simplices": T})
    space = VEMSpace(view, 3, element="hermite", mapped=True)

    centres, heights, widths = (numpy.concatenate(p) for p in zip(peaks(crescent), islet_peaks))
    bump = lambda x: 1.0 + sum(h * numpy.exp(-numpy.sum((x - c) ** 2) / w ** 2)
                               for c, h, w in zip(centres, heights, widths))
    on_crescent, on_islets = {}, {}
    for e in view.elements:
        c = numpy.asarray(e.geometry.center)
        k = view.indexSet.index(e)
        crescent_land = crescent.contains_point(c)
        on_crescent[k] = bump(c) if crescent_land else 0.0
        on_islets[k] = bump(c) if not crescent_land and inside_land(land, c[None])[0] else 0.0

    load_c, stiffness = assemble_poisson(space, lambda e, x: on_crescent[view.indexSet.index(e)], quad_order=8)
    load_i, mass = assemble_l2_projection(
        space, lambda e, x: on_islets[view.indexSet.index(e)], 8, stabilisation="auto",
    )
    Q = boundary_elimination(space)
    mass_r = (Q.T @ mass @ Q).tocsr()
    operator_r = (Q.T @ (stiffness + kappa ** 2 * mass) @ Q).tocsr()
    load_c, load_i = Q.T @ load_c, Q.T @ load_i
    load = lambda t: smoothstep(t, *crescent_ramp) * load_c + smoothstep(t, *islet_ramp) * load_i

    pts, S, tris = sample_matrix(space, level=2)
    S = (S @ Q).tocsr()

    dt = final_time / (frames * steps_per_frame)
    z = numpy.zeros(mass_r.shape[0])
    samples = [S @ z]
    for n in range(frames):
        t0 = n * steps_per_frame * dt
        z = solve_heat(
            mass_r, operator_r, z,
            load=lambda t: load(t0 + t),
            dirichlet=lambda t: numpy.zeros(0),
            dirichlet_ids=[],
            dt=dt,
            steps=steps_per_frame,
        )
        samples.append(S @ z)

    # Normalise by the final landscape, as run_topography does with its steady state.
    scale = samples[-1].max()
    tri = Triangulation(pts[:, 0], pts[:, 1], tris)
    sea = numpy.linspace(-1.0, 0.0, 9)
    land_levels = numpy.linspace(0.0, 1.0, 19)
    levels = numpy.concatenate((sea, land_levels[1:]))
    norm = TwoSlopeNorm(0.0, -1.0, 1.0)

    fig = plt.figure(figsize=(5, 5.3), facecolor="#f3ecd9")
    ax = fig.add_axes([0.04, 0.03, 0.92, 0.87])

    def draw(k):
        ax.clear()
        d = samples[k] / scale - sea_level
        h = numpy.where(d > 0, (d / (1 - sea_level)).clip(0) ** 1.4, -(-d / sea_level).clip(0) ** 0.5).clip(-1, 1)
        ax.tricontourf(tri, h, levels=levels, cmap=MAP, norm=norm, extend="both")
        ax.tricontour(tri, h, levels=sea[1:-1], colors="#1f4f7a", linewidths=0.3, alpha=0.45)
        if h.max() > 0:
            ax.tricontour(tri, h, levels=land_levels[1:-1], colors="#5a3e22",
                          linewidths=[0.7 if j % 5 == 4 else 0.3 for j in range(len(land_levels) - 2)], alpha=0.7)
            ax.tricontour(tri, h, levels=[0.0], colors="#2b2118", linewidths=0.9)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#2b2118")
            spine.set_linewidth(1.5)
        fig.suptitle(f"Island uplift, Hermite k=3 mapped VEM   t = {k * steps_per_frame * dt:.3f}",
                     fontsize=10, color="#2b2118", y=0.96)

    # Hold on the finished map before looping.
    animation = FuncAnimation(fig, draw, frames=list(range(frames + 1)) + [frames] * 20)
    animation.save(filename, writer=PillowWriter(fps=15), savefig_kwargs={"facecolor": fig.get_facecolor()})
    return fig


if __name__ == "__main__":
    run_topography_evolution()
