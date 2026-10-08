import io

import numpy
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import triangle
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.path import Path
from matplotlib.tri import Triangulation
from PIL import Image
from dune.alugrid import aluConformGrid

from VEM import VEMSpace, assemble_poisson
from VEM.assembly import assemble_l2_projection

MAP = LinearSegmentedColormap.from_list("map", [
    (0.00, "#1b3f6b"), (0.30, "#3f7fb5"), (0.49, "#a9d6e5"),
    (0.50, "#e8dcae"), (0.56, "#a7c98a"), (0.68, "#6f9e5c"),
    (0.82, "#b49a6c"), (0.93, "#8a6a48"), (1.00, "#f6f3ec"),
])


def inside_land(path, points):
    # Even-odd over the path's polygons.
    count = sum(Path(p).contains_points(points).astype(int) for p in path.to_polygons())
    return count % 2 == 1


def resample(poly, h):
    poly = poly[:-1] if numpy.allclose(poly[0], poly[-1]) else poly
    closed = numpy.vstack((poly, poly[:1]))
    s = numpy.concatenate(([0.0], numpy.cumsum(numpy.linalg.norm(numpy.diff(closed, axis=0), axis=1))))
    t = numpy.linspace(0.0, s[-1], max(int(s[-1] / h), 8), endpoint=False)
    return numpy.column_stack((numpy.interp(t, s, closed[:, 0]), numpy.interp(t, s, closed[:, 1])))


def build_mesh(path, R=1.0, h_land=0.02, h_outer=0.035):
    outer = resample(R * numpy.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float), h_outer)
    loops = [outer] + [resample(p, h_land) for p in path.to_polygons()]
    vertices, segments, offset = [], [], 0
    for loop in loops:
        k = len(loop)
        vertices.append(loop)
        segments.append(offset + numpy.column_stack((numpy.arange(k), (numpy.arange(k) + 1) % k)))
        offset += k
    pslg = {"vertices": numpy.vstack(vertices), "segments": numpy.vstack(segments)}

    mesh = triangle.triangulate(pslg, f"pq30a{0.5 * h_outer ** 2}")
    centroids = mesh["vertices"][mesh["triangles"]].mean(axis=1)
    inside = inside_land(path, centroids)
    mesh["triangle_max_area"] = numpy.where(inside, 0.5 * h_land ** 2, 0.5 * h_outer ** 2)[:, None]
    mesh = triangle.triangulate(mesh, "rpq30a")

    # Jitter interior vertices so the polygonal mesh shows through the contours.
    rng = numpy.random.default_rng(0)
    V = mesh["vertices"].copy()
    free = mesh["vertex_markers"].ravel() == 0
    V[free] += 0.12 * h_land * rng.standard_normal((free.sum(), 2))
    # Undo the jitter around any element it inverted; ALUGrid crashes on those.
    T = mesh["triangles"]
    while True:
        d1, d2 = V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]
        flipped = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0] <= 0
        if not flipped.any():
            return V, T
        V[T[flipped]] = mesh["vertices"][T[flipped]]


def boundary_elimination(space):
    """
    Homogeneous Dirichlet data for Hermite dofs: clamp boundary values and the
    tangential derivative, keep the normal derivative free. Returns Q with
    u = Q z so that the reduced system is Q^T A Q z = Q^T b.
    """
    view = space.view
    tangents = {}
    for e in view.elements:
        idx = numpy.asarray(space.mapper(e), dtype=int)
        corners = [numpy.asarray(e.geometry.corner(k), dtype=float) for k in range(3)]
        for i in view.intersections(e):
            if not i.boundary:
                continue
            a, b = (numpy.asarray(i.geometry.corner(k), dtype=float) for k in range(2))
            t = (b - a) / numpy.linalg.norm(b - a)
            for p in (a, b):
                k = next(k for k in range(3) if numpy.allclose(corners[k], p))
                tangents.setdefault(int(idx[3 * k]), []).append(t)

    n = len(space.mapper)
    rows, cols, vals = [], [], []
    removed = set()
    col = 0

    for vdof, ts in tangents.items():
        removed.update((vdof, vdof + 1, vdof + 2))
        t0, t1 = ts[0], ts[-1]
        if abs(t0[0] * t1[1] - t0[1] * t1[0]) > 0.5:
            continue  # corner: both derivatives clamped
        t = t0 if numpy.dot(t0, t1) >= 0 else -t0
        t = t + (t1 if numpy.dot(t1, t) >= 0 else -t1)
        normal = numpy.array([-t[1], t[0]]) / numpy.linalg.norm(t)
        rows += [vdof + 1, vdof + 2]
        cols += [col, col]
        vals += [normal[0], normal[1]]
        col += 1

    for dof in range(n):
        if dof not in removed:
            rows.append(dof)
            cols.append(col)
            vals.append(1.0)
            col += 1

    return scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, col))


def sample_matrix(space, level=3):
    """
    Return plot points, the sparse matrix S mapping dofs to the field sampled at
    them, and the plot triangles, so that a dof vector plots as S @ dofs.
    """
    m = 2 ** level
    ref = numpy.array([[i / m, j / m] for j in range(m + 1) for i in range(m + 1 - j)])
    index = {(i, j): k for k, (i, j) in enumerate((i, j) for j in range(m + 1) for i in range(m + 1 - j))}
    sub = []
    for j in range(m):
        for i in range(m - j):
            sub.append((index[i, j], index[i + 1, j], index[i, j + 1]))
            if i + j < m - 1:
                sub.append((index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]))
    sub = numpy.array(sub)

    pts, rows, cols, vals, tris = [], [], [], [], []
    for e in space.view.elements:
        space.bind(e)
        idx = numpy.asarray(space.mapper(e), dtype=int)
        base = len(pts) * len(ref)
        phi = numpy.array([numpy.asarray(space.evaluateLocal(x), dtype=float).reshape(-1) for x in ref])
        rows.append(numpy.repeat(numpy.arange(base, base + len(ref)), len(idx)))
        cols.append(numpy.tile(idx, len(ref)))
        vals.append(phi.ravel())
        tris.append(sub + base)
        pts.append([e.geometry.toGlobal(x) for x in ref])
    # Average the (slightly nonconforming) Pi_0 values at shared sample points
    # so the plotted field is connected and contours don't break at edges.
    pts, tris = numpy.vstack(pts), numpy.vstack(tris)
    _, first, inverse = numpy.unique(numpy.round(pts, 9), axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    values = scipy.sparse.csr_matrix(
        (numpy.concatenate(vals), (numpy.concatenate(rows), numpy.concatenate(cols))),
        shape=(len(pts), len(space.mapper)),
    )
    average = scipy.sparse.csr_matrix((1.0 / numpy.bincount(inverse)[inverse], (inverse, numpy.arange(len(pts)))))
    return pts[first], (average @ values).tocsr(), inverse[tris]


def sample_solution(space, dofs, level=3):
    pts, S, tris = sample_matrix(space, level)
    return pts, S @ dofs, tris


def crescent_path(r1=0.75, r2=0.64, d=0.24, angle=0.6):
    # Disc of radius r1 minus a disc of radius r2 offset by d along x.
    x = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d)
    y = numpy.sqrt(r1 ** 2 - x ** 2)
    phi, psi = numpy.arctan2(y, x), numpy.arctan2(y, x - d)
    a = numpy.linspace(phi, 2 * numpy.pi - phi, 160)
    b = numpy.linspace(2 * numpy.pi - psi, psi, 120)[1:-1]
    verts = numpy.vstack((
        numpy.column_stack((r1 * numpy.cos(a), r1 * numpy.sin(a))),
        numpy.column_stack((d + r2 * numpy.cos(b), r2 * numpy.sin(b))),
    ))
    c, s = numpy.cos(angle), numpy.sin(angle)
    rot = numpy.array([[c, s], [-s, c]])
    verts = verts @ rot
    shift = 0.5 * (verts.min(axis=0) + verts.max(axis=0))
    # Also return the centre of the removed disc, i.e. the middle of the bay.
    return Path(verts - shift), numpy.array([d, 0.0]) @ rot - shift


def peaks(path, n=9, seed=3):
    # Gaussian peaks scattered along the crescent's spine.
    rng = numpy.random.default_rng(seed)
    verts = path.vertices
    outer, inner = verts[:160], verts[160:][::-1]
    t = numpy.sort(rng.uniform(0.1, 0.9, n))
    i, j = (t * 160).astype(int), (t * len(inner)).astype(int)
    centres = 0.5 * (outer[i] + inner[j]) + 0.03 * rng.standard_normal((n, 2))
    return centres, rng.uniform(1.5, 4.0, n), rng.uniform(0.05, 0.1, n)


def islets(bay, seed=5):
    # Lumpy discs clustered up and to the right of the bay's centre, each with its own peak.
    rng = numpy.random.default_rng(seed)
    spec = numpy.array([[0.15, 0.1, 0.08], [0.47, -0.02, 0.07], [0.05, 0.45, 0.07], [0.42, 0.34, 0.06]])
    centres = bay + spec[:, :2]
    theta = numpy.linspace(0.0, 2 * numpy.pi, 48, endpoint=False)
    paths = []
    for c, r in zip(centres, spec[:, 2]):
        rr = r * (1 + 0.15 * numpy.sin(3 * theta + rng.uniform(0, 2 * numpy.pi)))
        paths.append(Path(c + numpy.column_stack((rr * numpy.cos(theta), rr * numpy.sin(theta)))))
    return paths, (centres, rng.uniform(4.5, 6.5, len(spec)), spec[:, 2])


def render(V, T, pts, tris, h, filename, R=1.0, n_sea=8, n_land=18):
    fig = plt.figure(figsize=(8, 8), facecolor="#f3ecd9")
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])

    tri = Triangulation(pts[:, 0], pts[:, 1], tris)
    sea = numpy.linspace(h.min(), 0.0, n_sea + 1)
    land = numpy.linspace(0.0, h.max(), n_land + 1)
    ax.tricontourf(tri, h, levels=numpy.concatenate((sea, land[1:])), cmap=MAP, norm=TwoSlopeNorm(0.0, h.min(), h.max()))
    ax.tricontour(tri, h, levels=sea[1:-1], colors="#1f4f7a", linewidths=0.4, alpha=0.45)
    ax.tricontour(tri, h, levels=land[1:], colors="#5a3e22", linewidths=[0.9 if k % 5 == 4 else 0.35 for k in range(n_land)], alpha=0.7)
    ax.tricontour(tri, h, levels=[0.0], colors="#2b2118", linewidths=1.1)
    ax.triplot(Triangulation(V[:, 0], V[:, 1], T), color="#2b2118", linewidth=0.2, alpha=0.06)

    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#2b2118")
        spine.set_linewidth(2.0)
    fig.savefig(filename, dpi=256, facecolor=fig.get_facecolor())
    return fig


def render_icon(pts, tris, h, filename, size=192, R=1.0, n_sea=4, n_land=8):
    # Full-bleed, no mesh or frame, few thick non-black lines: survives downscaling
    # and avatar cropping. Supersample 8x, then Lanczos down to an opaque RGB PNG.
    fig = plt.figure(figsize=(8, 8), dpi=size)
    ax = fig.add_axes([0, 0, 1, 1])
    tri = Triangulation(pts[:, 0], pts[:, 1], tris)
    sea = numpy.linspace(h.min(), 0.0, n_sea + 1)
    land = numpy.linspace(0.0, h.max(), n_land + 1)
    ax.tricontourf(tri, h, levels=numpy.concatenate((sea, land[1:])), cmap=MAP, norm=TwoSlopeNorm(0.0, h.min(), h.max()))
    ax.tricontour(tri, h, levels=sea[1:-1], colors="#2f6496", linewidths=1.5, alpha=0.6)
    ax.tricontour(tri, h, levels=land[1:-1], colors="#6b4f2e", linewidths=1.5, alpha=0.7)
    ax.tricontour(tri, h, levels=[0.0], colors="#3b2a1a", linewidths=3.0)
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, dpi=size, facecolor=MAP(0.0))
    plt.close(fig)
    Image.open(buf).convert("RGB").resize((size, size), Image.LANCZOS).save(filename)


def run_topography(kappa=11.0, sea_level=0.3, h_land=0.02, h_outer=0.035, filename=".tmp/topography.png"):
    crescent, bay = crescent_path()
    cluster, islet_peaks = islets(bay)
    land = Path.make_compound_path(crescent, *cluster)
    V, T = build_mesh(land, h_land=h_land, h_outer=h_outer)
    print("elements:", len(T))

    view = aluConformGrid({"vertices": V, "simplices": T})
    space = VEMSpace(view, 3, element="hermite", mapped=True)

    # Uplift f = 1_land * (base + peaks); screened Poisson spreads it into
    # slopes that decay into the sea over a length of roughly 1/κ.
    centres, heights, widths = (numpy.concatenate(p) for p in zip(peaks(crescent), islet_peaks))
    bump = lambda x: 1.0 + sum(h * numpy.exp(-numpy.sum((x - c) ** 2) / w ** 2)
                               for c, h, w in zip(centres, heights, widths))
    source = {}
    for e in view.elements:
        c = numpy.asarray(e.geometry.center)
        source[view.indexSet.index(e)] = bump(c) if inside_land(land,c[None])[0] else 0.0
    f = lambda e, x: source[view.indexSet.index(e)]

    rhs, stiffness = assemble_poisson(space, f, quad_order=8)
    _, mass = assemble_l2_projection(space, f, quad_order=8)
    Q = boundary_elimination(space)
    z = scipy.sparse.linalg.spsolve((Q.T @ (stiffness + kappa ** 2 * mass) @ Q).tocsc(), Q.T @ rhs)
    dofs = Q @ z

    pts, vals, tris = sample_solution(space, dofs)
    u = vals / vals.max()
    # Raise the land to a power so peaks sharpen while coasts stay gentle.
    d = u - sea_level
    h = numpy.where(d > 0, (d / (1 - sea_level)).clip(0) ** 1.4, -(-d / sea_level).clip(0) ** 0.5)
    render_icon(pts, tris, h, filename.replace(".png", "_192.png"))
    return render(V, T, pts, tris, h, filename)


if __name__ == "__main__":
    run_topography()
    plt.show()
