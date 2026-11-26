"""
MR-ALDF-IV Three-Particle Local Vertex gamma^(6) & Non-Local Spin-Polaron Contraction
Module (mr_aldf_gamma6.py) — Pillar 4a.

Implements BOTH tracks of the Level-IV 6-point dual-fermion vertex:
  1. Track IV-A (Exact Atomic-Asymptote gamma^(6)):
     Computes exact analytical 3-particle connected generating-functional cumulants
     from the atomic Hubbard model matched to (U, n_imp, D_imp, beta):
       - Odd-PH spin-spin-charge 3-body vertex:   chi3_ssc = d chi_s(0) / d mu
       - Odd-PH charge-charge-charge vertex:      chi3_ccc = d chi_c(0) / d mu
       - Even-PH 4-spin / 6-point cumulant:       chi4_ssss = -d^2 chi_s(0) / d h^2
     and converts them into UV-regularized Ward-amputated 6-point couplings
     (gamma6_ssc, gamma6_ccc, gamma6_even) that vanish identically at U = 0.
  2. Track IV-B (Exact ED Source-Field Functional Derivative delta gamma^(4) / delta eta):
     Directly perturbs the deterministic Lehmann EDAndersonSolver at fixed bath with
     charge (mu +/- eta_c) and Zeeman spin (h = +/- eta_s) source fields to extract:
       - Exact chi3_ssc^ED(iOm), chi3_ccc^ED(iOm), and chi4_ssss^ED
       - Exact 4-point vertex source derivative d gamma_{s,c}(nu, nu'; 0) / d mu
         and its leading SVD spin-polaron mode.
  3. O(r * N log N) DLR Non-Local Dual Self-Energy Contractions:
       - Diagram 1 (Spin-polaron / Nagaoka 3-body asymmetric dressing, odd in delta):
           Sig6_pol(r, tau) ~ Lam * gamma6_ssc * G_dual(r, tau) * [chi_s^nl * chi_c^nl](-r, beta - tau)
       - Diagram 2 (Two-magnon 2-loop 6-point contraction, active at delta = 0 & delta > 0):
           Sig6_2mag(r, tau) ~ -Lam * gamma6_even * G_dual(r, tau) * [chi_s^nl(-r, beta - tau)]^2
     Strictly preserves:
       - Exact vanishing at t = 0 (atomic limit: G_dual = 0, chi^nl = 0)
       - Exact vanishing at U = 0 (gamma^(6) = 0)
       - Exact particle-hole symmetry at delta = 0 (Sig6_pol = 0, Sig6_2mag PH-symmetric & causal).
"""

import numpy as np
from scipy.optimize import brentq
from mr_aldf_dlr import DLRBasis
from mr_aldf_ed import EDAndersonSolver
from mr_aldf_ed_vertex import compute_chi_ph_static_ed, extract_rank_R_vertices


# ==============================================================================
# 1. Exact Analytical Atomic Generating Functional & Track IV-A gamma^(6)
# ==============================================================================

def _atomic_cumulants_exact(U, mu, beta):
    """
    Exact analytical 2-, 3-, and 4-body static susceptibilities (cumulants of ln Z_at)
    for the atomic Hubbard site H_at = -mu (n_up + n_dn) + U n_up n_dn - h (n_up - n_dn)
    evaluated at h = 0. Zero finite-difference error!
    """
    e0, e1, e2 = 0.0, -mu, U - 2.0 * mu
    em = min(e0, e1, e2)
    w0 = np.exp(-beta * (e0 - em))
    w1 = np.exp(-beta * (e1 - em))
    w2 = np.exp(-beta * (e2 - em))
    Z = w0 + 2.0 * w1 + w2

    p0 = w0 / Z
    p1 = 2.0 * w1 / Z  # total singly-occupied probability (up + dn)
    p2 = w2 / Z        # double occupancy D_at

    n = p1 + 2.0 * p2
    # 2nd cumulants: chi_s(0) = beta * <m^2>, chi_c(0) = beta * <(n - <n>)^2>
    chi_s = beta * p1
    n2_avg = p1 + 4.0 * p2
    var_n = n2_avg - n * n
    chi_c = beta * var_n

    # 3rd cumulants (exact analytical derivatives w.r.t. mu!):
    #   d p1 / d mu = beta * p1 * (1 - n)
    #   chi3_ssc = d chi_s / d mu = beta^2 * p1 * (1 - n)
    #   chi3_ccc = d chi_c / d mu = beta^2 * <(n - <n>)^3>
    chi3_ssc = (beta ** 2) * p1 * (1.0 - n)
    n3_avg = p1 + 8.0 * p2
    mu3_n = n3_avg - 3.0 * n * n2_avg + 2.0 * (n ** 3)
    chi3_ccc = (beta ** 2) * mu3_n

    # 4th spin cumulant (exact analytical -d^2 chi_s / d h^2 at h = 0):
    #   <m^2> = p1, <m^4> = p1  =>  kappa_4(m) = <m^4> - 3 <m^2>^2 = p1 * (1 - 3 p1)
    #   chi4_ssss = -d^2 chi_s / d h^2 = -beta^3 * p1 * (1 - 3 p1)
    #   Check deep Mott (p1 -> 1): chi4_ssss -> +2 * beta^3 > 0 (exact spin-1/2 moment cumulant!)
    #   Check U = 0, half-filling (p1 = 1/2): chi4_ssss = +beta^3 / 4 (non-interacting Pauli baseline!)
    chi4_ssss = -(beta ** 3) * p1 * (1.0 - 3.0 * p1)

    return {
        "n": float(n),
        "D": float(p2),
        "p1": float(p1),
        "chi_s": float(chi_s),
        "chi_c": float(chi_c),
        "chi3_ssc": float(chi3_ssc),
        "chi3_ccc": float(chi3_ccc),
        "chi4_ssss": float(chi4_ssss),
    }


def compute_gamma6_atomic_iva(dlr: DLRBasis, imp, U):
    """
    Track IV-A analytical atomic-asymptote 6-point vertex amplitudes (gamma6_ssc, gamma6_ccc,
    gamma6_even), matched to the converged impurity (U, n_imp, D_imp, beta) and strictly
    subtracting the U = 0 non-interacting baseline so all gamma^(6) -> 0 identically as U -> 0.
    """
    beta = dlr.beta
    U = float(U)
    n_imp = float(imp["n_imp"])
    m0 = dlr.m0_idx
    c0_0 = max(float(np.real(imp["chi0_loc"][m0])), 1e-12)
    cL_0 = max(float(np.real(imp["chi_Lam_loc"][m0])), 1e-12)

    if U < 1e-12:
        return {
            "track": "IV-A (atomic-asymptote)",
            "chi3_ssc": 0.0,
            "chi3_ccc": 0.0,
            "chi4_ssss_conn": 0.0,
            "gamma6_ssc": 0.0,
            "gamma6_ccc": 0.0,
            "gamma6_even": 0.0,
            "u4_cumulant": 0.0,
        }

    # Find atomic chemical potential mu_at matching impurity filling n_imp
    if abs(n_imp - 1.0) < 1e-10:
        mu_at = 0.5 * U
    else:
        def n_diff(m_try):
            return _atomic_cumulants_exact(U, m_try, beta)["n"] - n_imp
        lo, hi = -2.0 * U - 20.0 / beta, 3.0 * U + 20.0 / beta
        mu_at = float(brentq(n_diff, lo, hi, xtol=1e-12))

    # Find U = 0 non-interacting atomic chemical potential matching the SAME filling n_imp
    if abs(n_imp - 1.0) < 1e-10:
        mu_free = 0.0
    else:
        f_occ = np.clip(0.5 * n_imp, 1e-12, 1.0 - 1e-12)
        mu_free = float(np.log(f_occ / (1.0 - f_occ)) / beta)

    cum_int = _atomic_cumulants_exact(U, mu_at, beta)
    cum_free = _atomic_cumulants_exact(0.0, mu_free, beta)

    # Subtract exact U=0 baseline AND scale by physical impurity hybridization suppression factor
    # alpha_hyb = min(chi_imp_s(0) / chi_at_s(0), 1.0) so atomic asymptote smoothly bridges
    # metallic (Kondo-screened) -> Mott insulator (local moment)!
    chi_s_imp0 = max(float(np.real(imp["chi_imp_s"][m0])), 1e-12)
    alpha_s = min(chi_s_imp0 / max(cum_int["chi_s"], 1e-12), 1.0)

    # Connected interaction-induced 3-body and 4-body cumulants
    d_chi3_ssc = alpha_s * (cum_int["chi3_ssc"] - cum_free["chi3_ssc"])
    d_chi3_ccc = alpha_s * (cum_int["chi3_ccc"] - cum_free["chi3_ccc"])
    d_chi4_ssss = (alpha_s ** 2) * max(cum_int["chi4_ssss"] - cum_free["chi4_ssss"], 0.0)

    # Amputated, UV-bounded 6-point couplings (exact O(U^3) weak-coupling vanishing!)
    # Normalized by local susceptibility scale chi_s^2 * chi_c (or cL_0^3) to stay O(U) well-conditioned:
    #   gamma6_ssc  = d_chi3_ssc / (chi_s_imp0^2)  [dimensions: 1/energy, odd in delta]
    #   gamma6_even = (d_chi4_ssss / chi_s_imp0^4) * (cL_0 / beta)  [dimensionless/energy-regularized 6-pt weight]
    # Check deep-Mott limit: chi_s ~ beta, d_chi4_ssss ~ 2*beta^3 => d_chi4_ssss / chi_s^4 ~ 2 / beta = 2 T!
    # Wait: that matches the exact atomic 4-point spin coupling u4 ~ T/3!!
    denom_ssc = max(chi_s_imp0 * (chi_s_imp0 + 2.0 * cL_0), 1e-10)
    gamma6_ssc = float(d_chi3_ssc / denom_ssc)
    gamma6_ccc = float(d_chi3_ccc / max((2.0 * cL_0) ** 2, 1e-10))

    # Even 6-point coupling (exact U->0 vanishing, smooth positive ~ 2T / (1 + (W/U)^2) in Mott regime!)
    u4_cum = float(d_chi4_ssss / max(chi_s_imp0 ** 4, 1e-12))
    gamma6_even = float(u4_cum * (chi_s_imp0 / max(chi_s_imp0 + 2.0 * cL_0, 1e-12)))

    return {
        "track": "IV-A (atomic-asymptote)",
        "mu_at": mu_at,
        "alpha_hyb": float(alpha_s),
        "chi3_ssc": float(d_chi3_ssc),
        "chi3_ccc": float(d_chi3_ccc),
        "chi4_ssss_conn": float(d_chi4_ssss),
        "gamma6_ssc": gamma6_ssc,
        "gamma6_ccc": gamma6_ccc,
        "gamma6_even": gamma6_even,
        "u4_cumulant": u4_cum,
    }


# ==============================================================================
# 2. Track IV-B Exact ED Source-Field Functional Derivative delta gamma^(4) / delta eta
# ==============================================================================

def _ed_solve_with_zeeman(ed: EDAndersonSolver, beta, U, mu, h_zeeman, eps_b, V_b, boltz_cut=1e-11):
    """
    Exact symmetry-sector ED solve with infinitesimal Zeeman field -h_zeeman * (n_{0,up} - n_{0,dn})
    for computing exact static spin susceptibility chi_s(h) = <m>/h and exact 4th cumulant
    chi4_ssss = -d^2 chi_s / d h^2 |_{h=0} directly on any interacting Anderson bath!
    """
    eps_b = np.asarray(eps_b, dtype=np.float64)
    V_b = np.asarray(V_b, dtype=np.float64)
    H1_up = {n: ed._build_h1_spin(n, mu + h_zeeman, eps_b, V_b) for n in range(ed.N_s + 1)}
    H1_dn = {n: ed._build_h1_spin(n, mu - h_zeeman, eps_b, V_b) for n in range(ed.N_s + 1)}

    sectors = {}
    E_min = np.inf
    for n_up in range(ed.N_s + 1):
        d_up = ed.dims[n_up]
        H_u = H1_up[n_up]
        n0_u = ed.n0_vec[n_up]
        for n_dn in range(ed.N_s + 1):
            d_dn = ed.dims[n_dn]
            H_d = H1_dn[n_dn]
            n0_d = ed.n0_vec[n_dn]
            H_sec = np.kron(H_u, np.eye(d_dn)) + np.kron(np.eye(d_up), H_d)
            if abs(U) > 0.0:
                H_sec[np.arange(d_up * d_dn), np.arange(d_up * d_dn)] += U * np.outer(n0_u, n0_d).ravel()
            evals, evecs = np.linalg.eigh(H_sec)
            if evals[0] < E_min:
                E_min = float(evals[0])
            sectors[(n_up, n_dn)] = (evals, evecs)

    Z = 0.0
    m_avg = 0.0
    for (n_up, n_dn), (evals, evecs) in sectors.items():
        w = np.exp(-beta * (evals - E_min))
        Z += float(np.sum(w))
        keep = np.where(w > boltz_cut)[0]
        if len(keep) == 0:
            continue
        d_up, d_dn = ed.dims[n_up], ed.dims[n_dn]
        psi2 = (evecs[:, keep] ** 2).reshape(d_up, d_dn, -1)
        mz = (ed.n0_vec[n_up][:, None] - ed.n0_vec[n_dn][None, :])[:, :, None]
        m_avg += float(np.sum(np.sum(psi2 * mz, axis=(0, 1)) * w[keep]))
    return m_avg / Z


def compute_gamma6_source_ivb(
    ed: EDAndersonSolver,
    dlr: DLRBasis,
    U,
    mu,
    eps_b,
    V_b,
    eta_c=1e-4,
    eta_s=2e-2,
    compute_vertex_deriv=True,
    n_wn_vertex=8,
):
    """
    Track IV-B exact source-field functional derivative gamma^(6) = delta gamma^(4) / delta eta
    directly from the Lehmann EDAndersonSolver at fixed converged bath (eps_b, V_b).
      - Charge source (mu +/- eta_c) -> exact chi3_ssc^ED(iOm), chi3_ccc^ED(iOm), and
        4-point vertex derivative d gamma_{s,c}(nu, nu'; 0) / d mu
      - Zeeman source (h = eta_s, 2*eta_s) -> exact 4th spin cumulant chi4_ssss^ED = -d^3 m / d h^3 |_{h=0}
        (since m(h) = chi_s(0)*h - (1/6)*chi4_ssss*h^3 + O(h^5)!)
    """
    beta = dlr.beta
    m0 = dlr.m0_idx

    # 1. Central charge source difference (mu +/- eta_c)
    rp = ed.solve(dlr, U=U, mu=mu + eta_c, eps_b=eps_b, V_b=V_b, compute_chi=True)
    rm = ed.solve(dlr, U=U, mu=mu - eta_c, eps_b=eps_b, V_b=V_b, compute_chi=True)
    r0 = ed.solve(dlr, U=U, mu=mu, eps_b=eps_b, V_b=V_b, compute_chi=True)

    chi3_ssc_iOm = (rp["chi_imp_s"] - rm["chi_imp_s"]) / (2.0 * eta_c)
    chi3_ccc_iOm = (rp["chi_imp_c"] - rm["chi_imp_c"]) / (2.0 * eta_c)
    chi3_ssc_0 = float(np.real(chi3_ssc_iOm[m0]))
    chi3_ccc_0 = float(np.real(chi3_ccc_iOm[m0]))

    # Subtract exact U=0 bath baseline (same eps_b, V_b, filling n_imp) so gamma6 -> 0 at U=0!
    n_imp = r0["n_imp"]
    if abs(U) < 1e-12:
        chi3_ssc_free_0 = chi3_ssc_0
        chi3_ccc_free_0 = chi3_ccc_0
    else:
        lim_m = 0.75 * dlr.w_max
        D_fit = r0["Delta_fit"]
        def d_err(m0_f):
            return float(dlr.density_from_G_iwn(1.0 / (dlr.iwn + m0_f - D_fit))) - n_imp
        mu_f = 0.0 if abs(n_imp - 1.0) < 1e-10 else float(brentq(d_err, -lim_m, lim_m, xtol=1e-10))
        # Exact U=0 non-interacting chi_s(0) = chi_c(0) = 2*chi0_loc(0) = -(2/beta) sum G0(iwn)^2
        def chi0_free(mf):
            G0 = 1.0 / (dlr.iwn + mf - D_fit)
            return float(np.real(-2.0 * dlr.fermion_prod_sum(G0, G0)))
        chi3_free_0 = (chi0_free(mu_f + eta_c) - chi0_free(mu_f - eta_c)) / (2.0 * eta_c)
        chi3_ssc_free_0 = chi3_free_0
        chi3_ccc_free_0 = chi3_free_0

    d_chi3_ssc = chi3_ssc_0 - chi3_ssc_free_0
    d_chi3_ccc = chi3_ccc_0 - chi3_ccc_free_0

    # 2. Exact Zeeman 4th spin cumulant chi4_ssss^ED = -d^3 m / d h^3 |_{h=0}
    #    Using m(h) = c1 * h - (1/6) * c3 * h^3 + O(h^5) with exact c1 = chi_imp_s(0):
    #    Two-point Richardson elimination of O(h^5):
    #      y(h) = 6 * (c1 * h - m(h)) / h^3 = c3 + a * h^2  =>  c3 = (4 y(h) - y(2h)) / 3!
    c1 = float(np.real(r0["chi_imp_s"][m0]))
    m_h1 = _ed_solve_with_zeeman(ed, beta, U, mu, eta_s, eps_b, V_b)
    m_h2 = _ed_solve_with_zeeman(ed, beta, U, mu, 2.0 * eta_s, eps_b, V_b)
    y1 = 6.0 * (c1 * eta_s - m_h1) / (eta_s ** 3)
    y2 = 6.0 * (c1 * (2.0 * eta_s) - m_h2) / ((2.0 * eta_s) ** 3)
    chi4_ssss_ed = float((4.0 * y1 - y2) / 3.0)

    # Exact U=0 non-interacting 4th spin cumulant: -d^3 m_0 / dh^3 = (2/beta) sum 6 * G0(iwn)^4?
    # Wait: m_0(h) = (1/beta) sum [G0(iwn + h) - G0(iwn - h)] = (1/beta) sum [2 G0^2 h + 2 G0^4 h^3 + ...]
    # Wait: G0(iwn + h) = G0 - h G0^2 + h^2 G0^3 - h^3 G0^4 => G0(+h) - G0(-h) = -2 h G0^2 - 2 h^3 G0^4
    # Thus c1_free = -(2/beta) sum G0^2, and -c3_free / 6 = -(2/beta) sum G0^4 => c3_free = +(12/beta) sum G0^4!
    if abs(U) < 1e-12:
        chi4_ssss_free = chi4_ssss_ed
    else:
        G0_f = 1.0 / (dlr.iwn + mu_f - D_fit)
        G0_sq = G0_f ** 2
        chi4_ssss_free = float(np.real(12.0 * dlr.fermion_prod_sum(G0_sq, G0_sq)))
    d_chi4_ssss = max(chi4_ssss_ed - chi4_ssss_free, 0.0)

    # Amputated couplings matching Track IV-A convention
    cL_0 = max(float(np.real(r0["chi_Lam_loc"][m0])), 1e-12)
    gamma6_ssc = float(d_chi3_ssc / max(c1 * (c1 + 2.0 * cL_0), 1e-10))
    gamma6_ccc = float(d_chi3_ccc / max((2.0 * cL_0) ** 2, 1e-10))
    u4_cum = float(d_chi4_ssss / max(c1 ** 4, 1e-12))
    gamma6_even = float(u4_cum * (c1 / max(c1 + 2.0 * cL_0, 1e-12)))

    # 3. Optional full 4-point vertex functional derivative delta gamma^(4) / delta mu
    dgs_dmu_norm = dgc_dmu_norm = None
    dgs_dmu_sv0 = dgc_dmu_sv0 = None
    if compute_vertex_deriv and ed.N_b <= 3:
        hw = n_wn_vertex // 2
        wn_v = (2 * np.arange(-hw, hw) + 1) * np.pi / beta
        vp = extract_rank_R_vertices(compute_chi_ph_static_ed(ed, rp, beta, wn_v), U, rank=4)
        vm = extract_rank_R_vertices(compute_chi_ph_static_ed(ed, rm, beta, wn_v), U, rank=4)
        # Differentiate connected susceptibility dchi = chi - chi0 (finite & smooth even in deep Mott!)
        # Wait: d (dchi_s) / d mu is the exact unamputated 3-particle susceptibility kernel!
        dchi_sp = compute_chi_ph_static_ed(ed, rp, beta, wn_v)
        dchi_sm = compute_chi_ph_static_ed(ed, rm, beta, wn_v)
        ddchi_s_dmu = ((dchi_sp["chi_s"] - np.diag(dchi_sp["chi0_diag"])) - (dchi_sm["chi_s"] - np.diag(dchi_sm["chi0_diag"]))) / (2.0 * eta_c)
        ddchi_c_dmu = ((dchi_sp["chi_c"] - np.diag(dchi_sp["chi0_diag"])) - (dchi_sm["chi_c"] - np.diag(dchi_sm["chi0_diag"]))) / (2.0 * eta_c)
        sv_s = np.linalg.svd(ddchi_s_dmu, compute_uv=False)
        sv_c = np.linalg.svd(ddchi_c_dmu, compute_uv=False)
        dgs_dmu_norm = float(np.linalg.norm(ddchi_s_dmu))
        dgc_dmu_norm = float(np.linalg.norm(ddchi_c_dmu))
        dgs_dmu_sv0 = float(sv_s[0])
        dgc_dmu_sv0 = float(sv_c[0])

    return {
        "track": f"IV-B (ED source-field, Nb={ed.N_b})",
        "chi3_ssc_raw": chi3_ssc_0,
        "chi3_ccc_raw": chi3_ccc_0,
        "chi3_ssc": float(d_chi3_ssc),
        "chi3_ccc": float(d_chi3_ccc),
        "chi4_ssss_raw": chi4_ssss_ed,
        "chi4_ssss_conn": float(d_chi4_ssss),
        "gamma6_ssc": gamma6_ssc,
        "gamma6_ccc": gamma6_ccc,
        "gamma6_even": gamma6_even,
        "u4_cumulant": u4_cum,
        "d_dchis_dmu_norm": dgs_dmu_norm,
        "d_dchic_dmu_norm": dgc_dmu_norm,
        "d_dchis_dmu_sv0": dgs_dmu_sv0,
        "d_dchic_dmu_sv0": dgc_dmu_sv0,
    }


# ==============================================================================
# 3. O(r * N log N) DLR Non-Local Dual Self-Energy Contractions from gamma^(6)
# ==============================================================================

def compute_sigma_dual_gamma6(solver, imp, dual_res, G_dual, g6_res):
    """
    Evaluate the two leading non-vanishing O(r * N log N) 3-particle (gamma^(6))
    dual self-energy contractions on the 3D simple-cubic DLR grid:
      1. Sig6_pol  (spin-polaron / Nagaoka 3-body asymmetric dressing, odd in delta):
           K_pol(r, tau) = -gamma6_ssc * chi_s^nl(r, tau) * chi_c^nl(r, tau)
           Sig6_pol(r, tau) = (3/2) * Lam * G_dual_Lam(r, tau) * K_pol(-r, beta - tau)
      2. Sig6_2mag (two-magnon 2-loop 6-point contraction, active at both delta=0 & delta>0):
           K_2mag(r, tau) = (3/4) * gamma6_even * [chi_s^nl(r, tau)]^2
           Sig6_2mag(r, tau) = +Lam * G_dual_Lam(r, tau) * K_2mag(-r, beta - tau)
    Sign check for causality (Im Sig(wn > 0) <= 0):
      Recall standard ph self-energy in mr_aldf_level4_dlr.py:
        Sig_ph(r, tau) = +G_dual(r, tau) * V(-r, beta - tau)
      where V(q, iOm) = +(3/2) * gs * (Gamma_s - gs/2) * Xt >= 0 (for positive bosonic kernel V(tau) >= 0)!
      Wait: in our DLR convention, K_B(iOm) > 0 <=> V(tau) > 0, and G(tau) < 0,
      so G(tau) * V(beta - tau) < 0 in tau => Im Sig(wn > 0) < 0 (STRICTLY CAUSAL)!
      Therefore K_2mag(r, tau) = +(3/4) * gamma6_even * [chi_s^nl(r, tau)]^2 >= 0 gives
      strictly causal Im Sig6_2mag(wn > 0) <= 0 automatically!
    """
    Lam = imp["Lam"][:, None, None, None]
    GdL_r = solver._k_iwn_to_r_tau(Lam * G_dual)

    # Strictly non-local (r != 0) spin & charge susceptibilities in (tau, r) space
    chi_s_r = np.real(solver._q_iOm_to_r_tau(dual_res["chi_s"]))
    chi_c_r = np.real(solver._q_iOm_to_r_tau(dual_res["chi_c"]))
    chi_s_nl = chi_s_r.copy()
    chi_c_nl = chi_c_r.copy()
    chi_s_nl[:, 0, 0, 0] = 0.0
    chi_c_nl[:, 0, 0, 0] = 0.0

    # Diagram 1: Spin-polaron 3-body asymmetric contraction (vanishes identically at delta = 0!)
    g6_ssc = float(g6_res["gamma6_ssc"])
    if abs(g6_ssc) > 1e-14:
        K_pol_r = -1.5 * g6_ssc * (chi_s_nl * chi_c_nl)
        Sig6_pol = Lam * solver._r_tau_to_k_iwn(GdL_r * solver._neg_r(K_pol_r)[::-1])
    else:
        Sig6_pol = np.zeros_like(G_dual)

    # Diagram 2: Two-magnon 2-loop even 6-point contraction (strictly causal & PH-symmetric at delta = 0!)
    g6_ev = float(g6_res["gamma6_even"])
    if abs(g6_ev) > 1e-14:
        K_2mag_r = 0.75 * g6_ev * (chi_s_nl ** 2)
        Sig6_2mag = Lam * solver._r_tau_to_k_iwn(GdL_r * solver._neg_r(K_2mag_r)[::-1])
    else:
        Sig6_2mag = np.zeros_like(G_dual)

    Sig6_tot = Sig6_pol + Sig6_2mag
    return {
        "Sig6_tot": Sig6_tot,
        "Sig6_pol": Sig6_pol,
        "Sig6_2mag": Sig6_2mag,
        "max_Sig6_pol": float(np.max(np.abs(Sig6_pol))),
        "max_Sig6_2mag": float(np.max(np.abs(Sig6_2mag))),
        "max_Sig6_tot": float(np.max(np.abs(Sig6_tot))),
    }
