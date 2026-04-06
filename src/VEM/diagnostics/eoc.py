import numpy
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

from ..spaces.common.triangle_geometry import coerce_triangle_vertices


def _set_power_of_two_ticks(ax):
    """
    Label the x-axis only at negative powers of two: 1, 2^{-1}, 2^{-2}, ...
    within the visible range.
    """

    xmin, xmax = ax.get_xlim()
    lo = float(min(xmin, xmax))
    hi = float(max(xmin, xmax))

    k_start = max(0, int(numpy.ceil(-numpy.log2(hi))))
    k_end = max(k_start, int(numpy.floor(-numpy.log2(lo))))

    ticks = [2.0 ** (-k) for k in range(k_start, k_end + 1)]
    ticks = [tick for tick in ticks if lo <= tick <= hi]
    if not ticks:
        return

    labels = [r"$1$" if k == 0 else rf"$2^{{-{k}}}$" for k in range(k_start, k_end + 1)]
    labels = [label for tick, label in zip([2.0 ** (-k) for k in range(k_start, k_end + 1)], labels) if lo <= tick <= hi]

    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.xaxis.set_minor_locator(NullLocator())


def mesh_size(view):
    """
    Return the mesh size h := max_E diam(E) on the current grid view.
    """
    h = 0.0
    for e in view.elements:
        verts = coerce_triangle_vertices(e)
        e01 = numpy.linalg.norm(verts[1] - verts[0])
        e12 = numpy.linalg.norm(verts[2] - verts[1])
        e20 = numpy.linalg.norm(verts[0] - verts[2])
        h = max(h, float(max(e01, e12, e20)))
    return h


def estimate_eoc(h_values, error_values):
    """
    Least-squares slope of log(error) against log(h).
    """
    h = numpy.asarray(h_values, dtype=float)
    e = numpy.asarray(error_values, dtype=float)

    mask = (h > 0.0) & (e > 0.0) & numpy.isfinite(h) & numpy.isfinite(e)
    h = h[mask]
    e = e[mask]

    if h.size < 2:
        return numpy.nan

    A = numpy.column_stack((numpy.log(h), numpy.ones_like(h)))
    slope, _ = numpy.linalg.lstsq(A, numpy.log(e), rcond=None)[0]
    return float(slope)


def _reference_curve(h_values, error_values, order):
    h = numpy.asarray(h_values, dtype=float)
    e = numpy.asarray(error_values, dtype=float)
    mask = (h > 0.0) & (e > 0.0) & numpy.isfinite(h) & numpy.isfinite(e)
    h = h[mask]
    e = e[mask]
    if h.size == 0:
        return h, e

    href = float(h[-1])
    eref = float(e[-1])
    return h, eref * (h / href) ** order


def plot_eoc_curves(histories, component_names, title_prefix="", show_reference=True):
    """
    Plot standard FEM-style convergence curves: error vs mesh size on log-log axes.

    Parameters
    ----------
    histories : dict[str, list[dict]]
        Mapping from label to refinement-history entries with keys
        {"h": float, "errors": sequence[float]}.
    component_names : sequence[str]
        Names of the error components, e.g. ("L2",) or ("L2", "H1-semi").
    title_prefix : str
        Prefix for plot titles.
    show_reference : bool
        If True, overlay a dashed reference line using the fitted EOC of each
        curve whenever at least two refinement levels are available.

    Returns
    -------
    figures : list[matplotlib.figure.Figure]
        Created figures.
    """

    figures = []

    for comp_id, comp_name in enumerate(component_names):
        fig, ax = plt.subplots()
        for label, entries in histories.items():
            hs = numpy.array([row["h"] for row in entries], dtype=float)
            errs = numpy.array([row["errors"][comp_id] for row in entries], dtype=float)

            order = estimate_eoc(hs, errs)
            if numpy.isfinite(order):
                curve_label = f"{label} (EOC≈{order:.2f})"
            else:
                curve_label = label

            ax.loglog(hs[::-1], errs, marker="o", linewidth=1.5, label=curve_label)

            if show_reference and numpy.isfinite(order) and hs.size >= 2:
                href, eref = _reference_curve(hs, errs, order)
                ax.loglog(
                    href[::-1],
                    eref,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.7,
                    label=f"{label} ref. h^{order:.2f}",
                )

        full_title = f"{title_prefix} {comp_name}".strip()
        ax.set_title(full_title)
        ax.set_xlabel("mesh size h")
        ax.set_ylabel(f"{comp_name} error")
        ax.grid(True, which="both", linestyle=":")
        ax.invert_xaxis()
        _set_power_of_two_ticks(ax)
        ax.legend()
        figures.append(fig)

    return figures
