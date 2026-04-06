import argparse
import math
import numpy as np
from dune.grid import cartesianDomain
from dune.alugrid import aluConformGrid
from dune.geometry import quadratureRule

from VEM import (
    CubicHermitePhysicalVEMSpace,
    CubicHermiteMappedVEMSpace,
    QuarticHermitePhysicalVEMSpace,
    QuarticHermiteMappedVEMSpace,
    mesh_size,
)


SPACE_MAP = {
    "cubic-physical": CubicHermitePhysicalVEMSpace,
    "cubic-mapped": CubicHermiteMappedVEMSpace,
    "quartic-physical": QuarticHermitePhysicalVEMSpace,
    "quartic-mapped": QuarticHermiteMappedVEMSpace,
}

PAIR_MAP = {
    "cubic": (CubicHermitePhysicalVEMSpace, CubicHermiteMappedVEMSpace),
    "quartic": (QuarticHermitePhysicalVEMSpace, QuarticHermiteMappedVEMSpace),
}


def build_demo_view(level, nx0=8, ny0=8):
    nx = nx0 * (2 ** level)
    ny = ny0 * (2 ** level)
    domain = cartesianDomain([0, 0], [1, 1], [nx, ny])
    return aluConformGrid(domain)


def local_consistency_matrix(space, element, quad_order):
    geo = element.geometry
    space.bind(element)
    nloc = space.localDofs
    K = np.zeros((nloc, nloc), dtype=float)
    for p in quadratureRule(element.type, quad_order):
        xhat = p.position
        w = float(p.weight * geo.integrationElement(xhat))
        grad_vals = np.asarray(space.evaluateLocalGradient(xhat), dtype=float)
        K += w * grad_vals.dot(grad_vals.T)
    return K


def orthonormal_nullspace(A, rtol=1e-11):
    _, s, vh = np.linalg.svd(A, full_matrices=True)
    if s.size == 0:
        return np.eye(A.shape[1], dtype=float)
    tol = rtol * max(A.shape) * (s[0] if s[0] > 0.0 else 1.0)
    rank = int(np.sum(s > tol))
    return vh[rank:, :].T.copy()


def restricted_spectrum(A, Z, atol=1e-13):
    if Z.shape[1] == 0:
        return {
            "dim": 0,
            "min": np.nan,
            "max": np.nan,
            "cond": np.nan,
            "min_abs": np.nan,
            "max_abs": np.nan,
        }

    R = Z.T.dot(0.5 * (A + A.T)).dot(Z)
    eigs = np.linalg.eigvalsh(R)
    abs_eigs = np.abs(eigs)
    pos = eigs[eigs > atol]
    return {
        "dim": int(Z.shape[1]),
        "min": float(eigs[0]),
        "max": float(eigs[-1]),
        "cond": float(pos[-1] / pos[0]) if pos.size else np.inf,
        "min_abs": float(abs_eigs[0]),
        "max_abs": float(abs_eigs[-1]),
    }


def summarize_stats(name, rows):
    if not rows:
        print(f"  {name}: no data")
        return

    mins = np.array([r["min"] for r in rows], dtype=float)
    maxs = np.array([r["max"] for r in rows], dtype=float)
    conds = np.array([r["cond"] for r in rows], dtype=float)
    dims = {r["dim"] for r in rows}

    print(
        f"  {name}: ker_dim={sorted(dims)} "
        f"min_eig in [{mins.min():.3e}, {mins.max():.3e}] "
        f"max_eig in [{maxs.min():.3e}, {maxs.max():.3e}] "
        f"cond in [{np.nanmin(conds):.3e}, {np.nanmax(conds):.3e}]"
    )


def summarize_scalar(name, values):
    if not values:
        print(f"  {name}: no data")
        return
    arr = np.asarray(values, dtype=float)
    print(f"  {name}: [{arr.min():.3e}, {arr.max():.3e}]")


def summarize_int(name, values):
    if not values:
        print(f"  {name}: no data")
        return
    print(f"  {name}: [{min(values)}, {max(values)}]")


def projector_metrics(P):
    defect = P.dot(P) - P
    return {
        "rank": int(np.linalg.matrix_rank(P)),
        "idemp_fro": float(np.linalg.norm(defect, ord="fro")),
        "idemp_inf": float(np.linalg.norm(defect, ord=np.inf)),
        "norm_fro": float(np.linalg.norm(P, ord="fro")),
        "trace": float(np.trace(P)),
    }


class PhysicalMonomialGF:
    def __init__(self, ax, ay):
        self.ax = int(ax)
        self.ay = int(ay)

    def __call__(self, e, xhat):
        x_phys = np.asarray(e.geometry.toGlobal(xhat), dtype=float)
        x = float(x_phys[0])
        y = float(x_phys[1])
        return (x ** self.ax) * (y ** self.ay)

    def jacobian(self, e, xhat):
        x_phys = np.asarray(e.geometry.toGlobal(xhat), dtype=float)
        x = float(x_phys[0])
        y = float(x_phys[1])

        if self.ax == 0:
            dx = 0.0
        else:
            dx = self.ax * (x ** (self.ax - 1)) * (y ** self.ay)

        if self.ay == 0:
            dy = 0.0
        else:
            dy = self.ay * (x ** self.ax) * (y ** (self.ay - 1))

        return np.array([dx, dy], dtype=float)

    def label(self):
        return f"x^{self.ax} y^{self.ay}"


def monomial_exponents(max_degree):
    exps = []
    for total in range(max_degree + 1):
        for ax in range(total + 1):
            ay = total - ax
            exps.append((ax, ay))
    return exps


def default_exactness_degree(space_cls):
    name = space_cls.__name__.lower()
    if "quartic" in name:
        return 4
    if "cubic" in name:
        return 3
    raise ValueError(f"Do not know a default exactness degree for {space_cls.__name__}.")


def projected_value(space, local_dofs, xhat):
    phi = np.asarray(space.evaluateLocal(xhat), dtype=float).reshape(-1)
    return float(local_dofs.dot(phi))


def projected_gradient(space, local_dofs, xhat):
    grad_phi = np.asarray(space.evaluateLocalGradient(xhat), dtype=float)
    return np.asarray(local_dofs.dot(grad_phi), dtype=float).reshape(2)


def analyze_space(space_cls, refinements, quad_order, max_elements=None):
    print(f"\n=== {space_cls.__name__} ===")
    for level in range(refinements):
        view = build_demo_view(level)
        space = space_cls(view)
        h = mesh_size(view)
        print(f"level={level}  h={h:.12g}  elements={view.size(0)}  localDofs={space.localDofs}")

        S_id_rows = []
        A_id_rows = []
        S_alpha_rows = []
        A_alpha_rows = []
        K_rows = []
        alpha_vals = []
        P_ranks = []
        P_idemp_fro = []
        P_idemp_inf = []
        P_norm_fro = []
        P_trace = []

        for elem_id, e in enumerate(view.elements):
            if max_elements is not None and elem_id >= max_elements:
                break

            K = local_consistency_matrix(space, e, quad_order=quad_order)
            P = np.asarray(space.localProjectorDofs(), dtype=float)
            I = np.eye(space.localDofs, dtype=float)
            S_id = (I - P).T.dot(I - P)
            alpha = float(np.trace(K)) / max(space.localDofs, 1)
            if abs(alpha) < 1e-14:
                alpha = 1.0
            S_alpha = alpha * S_id

            Z = orthonormal_nullspace(P)
            pm = projector_metrics(P)
            P_ranks.append(pm["rank"])
            P_idemp_fro.append(pm["idemp_fro"])
            P_idemp_inf.append(pm["idemp_inf"])
            P_norm_fro.append(pm["norm_fro"])
            P_trace.append(pm["trace"])
            alpha_vals.append(alpha)

            K_rows.append(restricted_spectrum(K, Z))
            S_id_rows.append(restricted_spectrum(S_id, Z))
            A_id_rows.append(restricted_spectrum(K + S_id, Z))
            S_alpha_rows.append(restricted_spectrum(S_alpha, Z))
            A_alpha_rows.append(restricted_spectrum(K + S_alpha, Z))

        print(f"  sampled_elements={len(alpha_vals)}")
        summarize_int("rank(P)", P_ranks)
        summarize_scalar("trace(P)", P_trace)
        summarize_scalar("||P||_F", P_norm_fro)
        summarize_scalar("||P^2-P||_F", P_idemp_fro)
        summarize_scalar("||P^2-P||_inf", P_idemp_inf)
        summarize_scalar("alpha", alpha_vals)
        summarize_stats("K | ker(P)", K_rows)
        summarize_stats("S_id | ker(P)", S_id_rows)
        summarize_stats("K+S_id | ker(P)", A_id_rows)
        summarize_stats("S_alpha | ker(P)", S_alpha_rows)
        summarize_stats("K+S_alpha | ker(P)", A_alpha_rows)


def compare_pair(physical_cls, mapped_cls, refinements, quad_order, max_elements=None):
    pair_name = f"{physical_cls.__name__} vs {mapped_cls.__name__}"
    print(f"\n=== Pair comparison: {pair_name} ===")

    for level in range(refinements):
        view = build_demo_view(level)
        space_phys = physical_cls(view)
        space_mapped = mapped_cls(view)
        h = mesh_size(view)

        if space_phys.localDofs != space_mapped.localDofs:
            raise ValueError(
                f"Local dof mismatch: {space_phys.localDofs} vs {space_mapped.localDofs}"
            )

        P_diff_fro = []
        P_diff_inf = []
        P_phys_idemp = []
        P_map_idemp = []
        value_diff_fro = []
        value_diff_inf = []
        grad_diff_fro = []
        grad_diff_inf = []

        sampled = 0
        for elem_id, e in enumerate(view.elements):
            if max_elements is not None and elem_id >= max_elements:
                break

            sampled += 1
            space_phys.bind(e)
            space_mapped.bind(e)

            P_phys = np.asarray(space_phys.localProjectorDofs(), dtype=float)
            P_map = np.asarray(space_mapped.localProjectorDofs(), dtype=float)
            P_delta = P_map - P_phys

            P_diff_fro.append(float(np.linalg.norm(P_delta, ord="fro")))
            P_diff_inf.append(float(np.linalg.norm(P_delta, ord=np.inf)))
            P_phys_idemp.append(float(np.linalg.norm(P_phys.dot(P_phys) - P_phys, ord="fro")))
            P_map_idemp.append(float(np.linalg.norm(P_map.dot(P_map) - P_map, ord="fro")))

            for p in quadratureRule(e.type, quad_order):
                xhat = p.position
                val_phys = np.asarray(space_phys.evaluateLocal(xhat), dtype=float)
                val_map = np.asarray(space_mapped.evaluateLocal(xhat), dtype=float)
                val_delta = val_map - val_phys
                value_diff_fro.append(float(np.linalg.norm(val_delta, ord=2)))
                value_diff_inf.append(float(np.linalg.norm(val_delta, ord=np.inf)))

                grad_phys = np.asarray(space_phys.evaluateLocalGradient(xhat), dtype=float)
                grad_map = np.asarray(space_mapped.evaluateLocalGradient(xhat), dtype=float)
                grad_delta = grad_map - grad_phys
                grad_diff_fro.append(float(np.linalg.norm(grad_delta, ord="fro")))
                grad_diff_inf.append(float(np.linalg.norm(grad_delta, ord=np.inf)))

        print(
            f"level={level}  h={h:.12g}  elements={view.size(0)}  sampled_elements={sampled}"
        )
        summarize_scalar("||P_map - P_phys||_F", P_diff_fro)
        summarize_scalar("||P_map - P_phys||_inf", P_diff_inf)
        summarize_scalar("||P_phys^2-P_phys||_F", P_phys_idemp)
        summarize_scalar("||P_map^2-P_map||_F", P_map_idemp)
        summarize_scalar("||Pi0_map - Pi0_phys||_2 @ quad", value_diff_fro)
        summarize_scalar("||Pi0_map - Pi0_phys||_inf @ quad", value_diff_inf)
        summarize_scalar("||Pi1_map - Pi1_phys||_F @ quad", grad_diff_fro)
        summarize_scalar("||Pi1_map - Pi1_phys||_inf @ quad", grad_diff_inf)


def analyze_exactness(space_cls, refinements, quad_order, max_elements=None, max_degree=None):
    if max_degree is None:
        max_degree = default_exactness_degree(space_cls)
    exponents = monomial_exponents(max_degree)

    print(f"\n=== Gradient exactness: {space_cls.__name__} on P_{max_degree} ===")
    print(f"  monomials tested: {len(exponents)}")

    for level in range(refinements):
        view = build_demo_view(level)
        space = space_cls(view)
        h = mesh_size(view)
        print(f"level={level}  h={h:.12g}  elements={view.size(0)}  localDofs={space.localDofs}")

        global_value_l2 = []
        global_value_rel = []
        global_value_max = []
        global_grad_l2 = []
        global_grad_rel = []
        global_grad_max = []

        worst_value = (-1.0, None)
        worst_grad = (-1.0, None)

        for ax, ay in exponents:
            gf = PhysicalMonomialGF(ax, ay)
            all_dofs = np.asarray(space.interpolate(gf), dtype=float)

            value_err_sq = 0.0
            value_true_sq = 0.0
            grad_err_sq = 0.0
            grad_true_sq = 0.0
            value_max = 0.0
            grad_max = 0.0
            sampled = 0

            for elem_id, e in enumerate(view.elements):
                if max_elements is not None and elem_id >= max_elements:
                    break

                sampled += 1
                space.bind(e)
                idx = np.asarray(space.mapper(e), dtype=int)
                local_dofs = all_dofs[idx]
                geo = e.geometry

                for p in quadratureRule(e.type, quad_order):
                    xhat = p.position
                    w = float(p.weight * geo.integrationElement(xhat))
                    x_phys = np.asarray(geo.toGlobal(xhat), dtype=float)

                    uh = projected_value(space, local_dofs, xhat)
                    gh = projected_gradient(space, local_dofs, xhat)

                    u = (float(x_phys[0]) ** ax) * (float(x_phys[1]) ** ay)
                    if ax == 0:
                        gux = 0.0
                    else:
                        gux = ax * (float(x_phys[0]) ** (ax - 1)) * (float(x_phys[1]) ** ay)
                    if ay == 0:
                        guy = 0.0
                    else:
                        guy = ay * (float(x_phys[0]) ** ax) * (float(x_phys[1]) ** (ay - 1))
                    g = np.array([gux, guy], dtype=float)

                    du = uh - u
                    dg = gh - g
                    value_err_sq += w * du * du
                    value_true_sq += w * u * u
                    grad_err_sq += w * float(np.dot(dg, dg))
                    grad_true_sq += w * float(np.dot(g, g))
                    value_max = max(value_max, abs(du))
                    grad_max = max(grad_max, float(np.max(np.abs(dg))))

            value_l2 = math.sqrt(value_err_sq)
            grad_l2 = math.sqrt(grad_err_sq)
            value_rel = value_l2 / max(math.sqrt(value_true_sq), 1e-16)
            grad_rel = grad_l2 / max(math.sqrt(grad_true_sq), 1e-16)

            global_value_l2.append(value_l2)
            global_value_rel.append(value_rel)
            global_value_max.append(value_max)
            global_grad_l2.append(grad_l2)
            global_grad_rel.append(grad_rel)
            global_grad_max.append(grad_max)

            if value_l2 > worst_value[0]:
                worst_value = (value_l2, (ax, ay, value_l2, value_rel, value_max, sampled))
            if grad_l2 > worst_grad[0] and ax + ay > 0:
                worst_grad = (grad_l2, (ax, ay, grad_l2, grad_rel, grad_max, sampled))

        summarize_scalar("value L2 error over monomials", global_value_l2)
        summarize_scalar("value relative L2 error over monomials", global_value_rel)
        summarize_scalar("value max abs error over monomials", global_value_max)
        summarize_scalar("grad L2 error over monomials", global_grad_l2)
        summarize_scalar("grad relative L2 error over monomials", global_grad_rel)
        summarize_scalar("grad max abs error over monomials", global_grad_max)

        if worst_value[1] is not None:
            ax, ay, l2err, relerr, maxerr, sampled = worst_value[1]
            print(
                "  worst value monomial: "
                f"x^{ax} y^{ay}  sampled_elements={sampled}  "
                f"L2={l2err:.3e}  rel={relerr:.3e}  maxabs={maxerr:.3e}"
            )
        if worst_grad[1] is not None:
            ax, ay, l2err, relerr, maxerr, sampled = worst_grad[1]
            print(
                "  worst grad monomial:  "
                f"x^{ax} y^{ay}  sampled_elements={sampled}  "
                f"L2={l2err:.3e}  rel={relerr:.3e}  maxabs={maxerr:.3e}"
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Print kernel-restricted local spectra, mapped-vs-physical diagnostics, and polynomial exactness checks for Hermite VEM spaces."
        )
    )
    parser.add_argument(
        "--spaces",
        nargs="+",
        default=["cubic-physical", "cubic-mapped", "quartic-physical", "quartic-mapped"],
        choices=sorted(SPACE_MAP.keys()),
        help="Spaces to test individually.",
    )
    parser.add_argument(
        "--analyze-spaces",
        default=False,
        action="store_true",
        help="Test spaces individually.",
    )
    parser.add_argument(
        "--compare-pairs",
        action="store_true",
        help="Also compare mapped spaces against their physical counterparts on the same mesh.",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["cubic", "quartic"],
        choices=sorted(PAIR_MAP.keys()),
        help="Mapped/physical pairs to compare when --compare-pairs is enabled.",
    )
    parser.add_argument(
        "--test-exactness",
        action="store_true",
        help="Also test exactness of projected values/gradients on physical monomials.",
    )
    parser.add_argument(
        "--exactness-degree",
        type=int,
        default=None,
        help="Override the polynomial degree used in the exactness test. Defaults to 3 for cubic spaces and 4 for quartic spaces.",
    )
    parser.add_argument("--refinements", type=int, default=5, help="Number of refinement levels.")
    parser.add_argument("--quad-order", type=int, default=10, help="Quadrature order for all diagnostics.")
    parser.add_argument(
        "--max-elements",
        type=int,
        default=None,
        help="Optional cap on elements sampled per level.",
    )
    args = parser.parse_args()

    if args.analyze_spaces:
        for key in args.spaces:
            analyze_space(
                SPACE_MAP[key],
                refinements=args.refinements,
                quad_order=args.quad_order,
                max_elements=args.max_elements,
            )

    if args.compare_pairs:
        for key in args.pairs:
            physical_cls, mapped_cls = PAIR_MAP[key]
            compare_pair(
                physical_cls,
                mapped_cls,
                refinements=args.refinements,
                quad_order=args.quad_order,
                max_elements=args.max_elements,
            )

    if args.test_exactness:
        for key in args.spaces:
            analyze_exactness(
                SPACE_MAP[key],
                refinements=args.refinements,
                quad_order=args.quad_order,
                max_elements=args.max_elements,
                max_degree=args.exactness_degree,
            )


if __name__ == "__main__":
    main()
