import numpy
from dune.geometry import quadratureRule

def compare_projectors(spaceA, spaceB, quad_order=8, print_per_element=False,
                       compare_local_mass=True):
    """
    Compare two projector-based spaces on the same grid by evaluating their
    projected local basis vectors phi(xhat) = evaluateLocal(xhat) on each
    element.

    Metrics are quadrature-based over the physical mesh:
      - global L2/Frobenius norm of projector difference
      - relative difference (against spaceA)
      - per-local-basis-column L2 differences
      - max pointwise absolute basis-value discrepancy at quadrature points
      - (optional) local mass-matrix discrepancy induced by projected basis
        values

    Parameters
    ----------
    spaceA, spaceB : spaces with methods
        - bind(e)
        - evaluateLocal(xhat) -> shape (localDofs,)
        - localDofs
        - view
    quad_order : int
        Triangle quadrature order for the comparison integrals.
    print_per_element : bool
        If True, print a short line of metrics per element.
    compare_local_mass : bool
        If True, also compare local matrices int phi_i phi_j on each element.

    Returns
    -------
    stats : dict
        Summary dictionary with global metrics and (optionally) per-element
        data.
    """
    if spaceA.localDofs != spaceB.localDofs:
        raise ValueError(f"localDofs mismatch: {spaceA.localDofs} vs {spaceB.localDofs}")
    if len(spaceA.mapper) != len(spaceB.mapper):
        raise ValueError(f"global dof count mismatch: {len(spaceA.mapper)} vs"\
                         f"{len(spaceB.mapper)}")

    nloc = spaceA.localDofs

    total_diff_sq = 0.0
    total_A_sq = 0.0
    total_B_sq = 0.0
    total_area = 0.0
    global_max_abs = 0.0

    col_diff_sq = numpy.zeros(nloc, dtype=float)
    col_A_sq = numpy.zeros(nloc, dtype=float)
    col_B_sq = numpy.zeros(nloc, dtype=float)

    mass_diff_sq_sum = 0.0   # sum over elements of ||M_A^E - M_B^E||_F^2
    mass_A_sq_sum = 0.0      # sum over elements of ||M_A^E||_F^2

    element_stats = []

    # Iterate over elements (same grid assumed)
    for elem_id, e in enumerate(spaceA.view.elements):
        geo = e.geometry

        spaceA.bind(e)
        spaceB.bind(e)

        ediff_sq = 0.0
        eA_sq = 0.0
        eB_sq = 0.0
        emax_abs = 0.0
        earea = 0.0

        ecol_diff_sq = numpy.zeros(nloc, dtype=float)
        ecol_A_sq = numpy.zeros(nloc, dtype=float)
        ecol_B_sq = numpy.zeros(nloc, dtype=float)

        if compare_local_mass:
            MA = numpy.zeros((nloc, nloc), dtype=float)
            MB = numpy.zeros((nloc, nloc), dtype=float)

        for p in quadratureRule(e.type, quad_order):
            xhat = p.position
            w = float(p.weight * geo.integrationElement(xhat))
            earea += w

            phiA = numpy.asarray(spaceA.evaluateLocal(xhat),
                                 dtype=float).reshape(-1)
            phiB = numpy.asarray(spaceB.evaluateLocal(xhat),
                                 dtype=float).reshape(-1)

            if phiA.shape[0] != nloc or phiB.shape[0] != nloc:
                raise ValueError(
                    f"evaluateLocal shape mismatch on element {elem_id}: "
                    f"{phiA.shape} vs {phiB.shape}, expected ({nloc},)"
                )

            d = phiA - phiB

            ediff_sq += w * float(numpy.dot(d, d))
            eA_sq += w * float(numpy.dot(phiA, phiA))
            eB_sq += w * float(numpy.dot(phiB, phiB))

            ecol_diff_sq += w * (d * d)
            ecol_A_sq += w * (phiA * phiA)
            ecol_B_sq += w * (phiB * phiB)

            pt_max = float(numpy.max(numpy.abs(d)))
            if pt_max > emax_abs:
                emax_abs = pt_max

            if compare_local_mass:
                MA += w * numpy.outer(phiA, phiA)
                MB += w * numpy.outer(phiB, phiB)

        total_diff_sq += ediff_sq
        total_A_sq += eA_sq
        total_B_sq += eB_sq
        total_area += earea
        global_max_abs = max(global_max_abs, emax_abs)

        col_diff_sq += ecol_diff_sq
        col_A_sq += ecol_A_sq
        col_B_sq += ecol_B_sq

        estats = {
            "element_id": elem_id,
            "area": earea,
            "L2_fro_diff": numpy.sqrt(ediff_sq),
            "L2_fro_A": numpy.sqrt(eA_sq),
            "L2_fro_B": numpy.sqrt(eB_sq),
            "rel_to_A": (numpy.sqrt(ediff_sq) / max(numpy.sqrt(eA_sq), 1e-16)),
            "max_abs": emax_abs,
        }

        if compare_local_mass:
            m_diff = numpy.linalg.norm(MA - MB, ord="fro")
            m_A = numpy.linalg.norm(MA, ord="fro")
            estats["local_mass_fro_diff"] = m_diff
            estats["local_mass_rel_to_A"] = m_diff / max(m_A, 1e-16)

            mass_diff_sq_sum += m_diff * m_diff
            mass_A_sq_sum += m_A * m_A

        element_stats.append(estats)

        if print_per_element:
            msg = (f"[elem {elem_id:4d}] "
                   f"L2proj diff={estats['L2_fro_diff']:.3e} "
                   f"(rel {estats['rel_to_A']:.3e}), "
                   f"maxabs={estats['max_abs']:.3e}")
            if compare_local_mass:
                msg += (f",  Mdiff={estats['local_mass_fro_diff']:.3e} "
                        f"(rel {estats['local_mass_rel_to_A']:.3e})")
            print(msg)

    # Global summaries
    global_L2_diff = numpy.sqrt(total_diff_sq)
    global_L2_A = numpy.sqrt(total_A_sq)
    global_L2_B = numpy.sqrt(total_B_sq)

    per_col_L2_diff = numpy.sqrt(col_diff_sq)
    per_col_L2_A = numpy.sqrt(col_A_sq)
    per_col_rel = per_col_L2_diff / numpy.maximum(per_col_L2_A, 1e-16)

    stats = {
        "num_elements": len(element_stats),
        "num_global_dofs_A": len(spaceA.mapper),
        "num_global_dofs_B": len(spaceB.mapper),
        "local_dofs": nloc,
        "total_area": total_area,
        "global_L2_fro_diff": global_L2_diff,
        "global_L2_fro_A": global_L2_A,
        "global_L2_fro_B": global_L2_B,
        "global_rel_to_A": global_L2_diff / max(global_L2_A, 1e-16),
        "global_max_abs": global_max_abs,
        "per_local_basis_L2_diff": per_col_L2_diff,
        "per_local_basis_L2_A": per_col_L2_A,
        "per_local_basis_rel_to_A": per_col_rel,
        "element_stats": element_stats,
    }

    if compare_local_mass:
        stats["global_local_mass_fro_diff_l2sum"] = numpy.sqrt(mass_diff_sq_sum)
        stats["global_local_mass_fro_A_l2sum"] = numpy.sqrt(mass_A_sq_sum)
        stats["global_local_mass_rel_to_A_l2sum"] = (
            stats["global_local_mass_fro_diff_l2sum"] /
            max(stats["global_local_mass_fro_A_l2sum"], 1e-16)
        )

    # Pretty print summary
    print("\n=== Projector comparison summary ===")
    print(f"elements                : {stats['num_elements']}")
    print(f"local dofs              : {stats['local_dofs']}")
    print(f"global dofs (A, B)      : {stats['num_global_dofs_A']}, {stats['num_global_dofs_B']}")
    print(f"mesh area (quadrature)  : {stats['total_area']:.12g}")
    print(f"global ||PiA-PiB||      : {stats['global_L2_fro_diff']:.6e}")
    print(f"global ||PiA||          : {stats['global_L2_fro_A']:.6e}")
    print(f"global relative diff    : {stats['global_rel_to_A']:.6e}")
    print(f"global max abs diff     : {stats['global_max_abs']:.6e}")

    if compare_local_mass:
        print(f"mass diff (l2-sum elems): {stats['global_local_mass_fro_diff_l2sum']:.6e}")
        print(f"mass rel  (l2-sum elems): {stats['global_local_mass_rel_to_A_l2sum']:.6e}")

    print("per-local-basis relative L2 diff:")
    for i, r in enumerate(stats["per_local_basis_rel_to_A"]):
        print(f"  basis {i:2d}: {r:.6e}")

    return stats