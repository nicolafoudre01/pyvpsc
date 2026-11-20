"""
Driver: reads parameters.py, runs the strain-rate driven VPSC, prints the results.
    python run.py             run
    python run.py --systems   list every slip system of the crystal (index, label) and exit
"""
import sys

import numpy as np

import parameters as P
import pyvpsc as pv


def main():
    if "--systems" in sys.argv:
        rec = pv.systems_records(P.CRYSTAL, P.SLIP, P.C_OVER_A, P.SYSTEM_OVERRIDES, P.DEFAULT_GAMMA0,
                                 include_unlisted=True)
        print(pv.systems_table(rec))
        return

    # ---- texture, slip systems, local law -------------------------------------------
    tex = pv.read_texture(P.TEXTURE_FILE)
    slip = pv.build_slip_systems(P.CRYSTAL, P.SLIP, P.C_OVER_A, P.SYSTEM_OVERRIDES, P.DEFAULT_GAMMA0)
    law = pv.CrystalPlasticity(slip, n=P.N_EXP, k_pen=P.K_PEN)
    D = pv.strain_rate_to_mandel(P.D)
    agg = pv.build_aggregate(tex, law)

    print(f"texture : {P.TEXTURE_FILE}  ({len(tex)} grains, shape a = {tex.shape})")
    n_all = len(slip.records)
    print(f"crystal : {P.CRYSTAL}, {len(slip)} active systems out of {n_all} in the listed families, "
          f"{slip.n_independent()} independent modes (5 needed)")
    print(pv.systems_table(slip.records))
    print(f"law     : n = {law.n}, k_pen = {law.kappa:.3g}")
    print(f"scheme  : {P.LINEARIZATION}, NQ = {P.NQ}\n")
    if slip.n_independent() < 5:
        raise SystemExit("the active systems cannot accommodate an arbitrary strain rate: activate more systems")

    # ---- solve -------------------------------------------------------------------
    res = pv.vpsc(D, agg, law, P.LINEARIZATION, a=tex.shape, nq=P.NQ, tol=P.TOL,
                  max_iter=P.MAX_ITER, relax=P.RELAX, verbose=P.VERBOSE)

    # ---- results -------------------------------------------------------------------
    np.set_printoptions(precision=5, suppress=True, linewidth=120)
    print(f"\nconverged in {res['n_iter']} iterations (relaxation {res['relax']})")
    print("\nimposed strain rate D (sample axes):\n", P.D)
    print("\nmacroscopic deviatoric stress Sigma' (sample axes):\n", pv.deviator3(res["Sigma"]))
    print(f"\nSigma_eq = {pv.sigma_eq(res['Sigma']):.5f}      (Taylor model: {pv.sigma_eq(res['Sigma_taylor']):.5f})")

    print("\nequation residuals (should be ~0):")
    for name, val in pv.check_equations(res, D, agg, law).items():
        print(f"  {name:20s} {val:.2e}")

    # ---- slip activity ---------------------------------------------------------------
    w = agg.fractions
    act = np.zeros(len(slip))
    for g, ph in enumerate(agg.phases):
        act += w[g] * np.abs(pv.slip_rates(res["sigma"][g], ph.internal["S"], law))
    tot = act.sum()
    print("\nslip activity (share of the total shear rate):")
    for fam in dict.fromkeys(slip.family):
        share = sum(a for a, f in zip(act, slip.family) if f == fam)
        print(f"  {fam:10s} {100 * share / tot:6.2f} %")
    if P.PRINT_ACTIVITY_PER_SYSTEM:
        print("\n  per system:")
        for s in range(len(slip)):
            print(f"  {slip.family[s]:10s} {slip.index[s]:3d}  {slip.label[s]:26s} {100 * act[s] / tot:6.2f} %")

    # ---- per-grain output ------------------------------------------------------------
    if P.OUTPUT_GRAINS_CSV:
        rows = []
        for g, ph in enumerate(agg.phases):
            gam = np.abs(pv.slip_rates(res["sigma"][g], ph.internal["S"], law)).sum()
            rows.append([g, *tex.euler_deg[g], w[g], pv.sigma_eq(res["sigma"][g]), gam, *res["sigma"][g], *res["d"][g]])
        hdr = "grain phi1 Phi phi2 weight sigma_eq total_slip " + " ".join(
            [f"s{c}" for c in ("xx", "yy", "zz", "xy", "xz", "yz")] + [f"d{c}" for c in ("xx", "yy", "zz", "xy", "xz", "yz")])
        np.savetxt(P.OUTPUT_GRAINS_CSV, np.array(rows), header=hdr, fmt="%.8g")
        print(f"\nper-grain results written to {P.OUTPUT_GRAINS_CSV}  (stress / strain rate in Mandel components)")


if __name__ == "__main__":
    main()
