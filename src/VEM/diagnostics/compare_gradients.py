import numpy
from dune.geometry import quadratureRule


def compare_gradient_projectors(spaceA, spaceB, quad_order=8, print_per_element=False):
    """
    Compare evaluateLocalGradient(xhat) between two spaces on the same grid.

    Metrics are quadrature-based over the physical mesh:
      - global L2/Frobenius norm of projector-gradient difference
      - relative difference (against spaceA)
      - max pointwise absolute discrepancy
    """
    if spaceA.localDofs != spaceB.localDofs:
        raise ValueError(f"localDofs mismatch: {spaceA.localDofs} vs {spaceB.localDofs}")
    if len(spaceA.mapper) != len(spaceB.mapper):
        raise ValueError(
            f"global dof count mismatch: {len(spaceA.mapper)} vs {len(spaceB.mapper)}"
        )

    nloc = spaceA.localDofs
    total_diff_sq = 0.0
    total_A_sq = 0.0
    total_area = 0.0
    global_max_abs = 0.0
    element_stats = []

    for elem_id, e in enumerate(spaceA.view.elements):
        geo = e.geometry
        spaceA.bind(e)
        spaceB.bind(e)

        ediff_sq = 0.0
        eA_sq = 0.0
        emax_abs = 0.0
        earea = 0.0

        for p in quadratureRule(e.type, quad_order):
            xhat = p.position
            w = float(p.weight * geo.integrationElement(xhat))
            earea += w

            gA = numpy.asarray(spaceA.evaluateLocalGradient(xhat), dtype=float)
            gB = numpy.asarray(spaceB.evaluateLocalGradient(xhat), dtype=float)

            if gA.shape != (nloc, 2) or gB.shape != (nloc, 2):
                raise ValueError(
                    f"evaluateLocalGradient shape mismatch on element {elem_id}: "
                    f"{gA.shape} vs {gB.shape}, expected ({nloc}, 2)"
                )

            d = gA - gB
            ediff_sq += w * float(numpy.sum(d * d))
            eA_sq += w * float(numpy.sum(gA * gA))
            emax_abs = max(emax_abs, float(numpy.max(numpy.abs(d))))

        total_diff_sq += ediff_sq
        total_A_sq += eA_sq
        total_area += earea
        global_max_abs = max(global_max_abs, emax_abs)

        estats = {
            "element_id": elem_id,
            "area": earea,
            "L2_fro_diff": numpy.sqrt(ediff_sq),
            "L2_fro_A": numpy.sqrt(eA_sq),
            "rel_to_A": numpy.sqrt(ediff_sq) / max(numpy.sqrt(eA_sq), 1e-16),
            "max_abs": emax_abs,
        }
        element_stats.append(estats)

        if print_per_element:
            print(
                f"[elem {elem_id:4d}] "
                f"grad diff={estats['L2_fro_diff']:.3e} "
                f"(rel {estats['rel_to_A']:.3e}), "
                f"maxabs={estats['max_abs']:.3e}"
            )

    stats = {
        "num_elements": len(element_stats),
        "local_dofs": nloc,
        "total_area": total_area,
        "global_L2_fro_diff": numpy.sqrt(total_diff_sq),
        "global_L2_fro_A": numpy.sqrt(total_A_sq),
        "global_rel_to_A": numpy.sqrt(total_diff_sq) / max(numpy.sqrt(total_A_sq), 1e-16),
        "global_max_abs": global_max_abs,
        "element_stats": element_stats,
    }

    print("\n=== Gradient projector comparison summary ===")
    print(f"elements               : {stats['num_elements']}")
    print(f"local dofs             : {stats['local_dofs']}")
    print(f"mesh area (quadrature) : {stats['total_area']:.12g}")
    print(f"global ||GA-GB||       : {stats['global_L2_fro_diff']:.6e}")
    print(f"global ||GA||          : {stats['global_L2_fro_A']:.6e}")
    print(f"global relative diff   : {stats['global_rel_to_A']:.6e}")
    print(f"global max abs diff    : {stats['global_max_abs']:.6e}")

    return stats