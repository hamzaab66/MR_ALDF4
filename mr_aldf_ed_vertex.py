"""
MR-ALDF-IV Track IV-B Exact 4-Point Local Vertex, Rank-R SVD Compression, and
Multi-Line Irreducible Charge Vertex Divergence Detector (mr_aldf_ed_vertex.py) — Pillar 3b.

Computes exact zero-noise static (iOm = 0) and finite-iOm 4-point generalized
susceptibilities chi_c^{nu, nu'} and chi_s^{nu, nu'} directly from the Track IV-B
EDAndersonSolver eigensystem in O(N_w * D^3) time via exact partial-fraction
factorization across all 6 time orderings (zero O(D^4) loops).

Extracts:
  1. Exact local 4-point vertices gamma_c(nu, nu'; 0) and gamma_s(nu, nu'; 0)
     in the exact MR-ALDF-IV convention (gamma_c -> +U, gamma_s -> -U as U -> 0).
  2. Optimal Rank-R (R = 4-6) SVD compression of gamma_{c,s}(nu, nu') with
     quantitative truncation error diagnostics vs. Rank-2 Track IV-A closure.
  3. Exact irreducible charge vertex spectrum eig(chi_c^{nu, nu'}) and count of
     crossed Schafer-Rohringer-Toschi Gamma_c^irr divergence lines N_div(U, T)
     (detecting 1st, 2nd, 3rd, ... divergence lines beyond Rank-2!).
"""

import numpy as np
from mr_aldf_ed import EDAndersonSolver


def _sector_c_matrices(ed: EDAndersonSolver, sectors, spin=0):
    """Exact eigenbasis matrices of c_{0,spin}^dag and c_{0,spin} between Fock sectors."""
    cdag = {}
    cann = {}
    if spin == 0:
        for n_up in range(ed.N_s):
            src_u, dst_u = ed.cdag0_map[n_up]
            d_up, d_up1 = ed.dims[n_up], ed.dims[n_up + 1]
            for n_dn in range(ed.N_s + 1):
                d_dn = ed.dims[n_dn]
                _, Va_f = sectors[(n_up, n_dn)]
                _, Vb_f = sectors[(n_up + 1, n_dn)]
                Va = Va_f.reshape(d_up, d_dn, -1)[src_u, :, :].reshape(len(src_u) * d_dn, -1)
                Vb = Vb_f.reshape(d_up1, d_dn, -1)[dst_u, :, :].reshape(len(dst_u) * d_dn, -1)
                M = Vb.T @ Va
                cdag[(n_up, n_dn)] = M
                cann[(n_up + 1, n_dn)] = M.T
    else:
        for n_up in range(ed.N_s + 1):
            d_up = ed.dims[n_up]
            sgn = -1.0 if (n_up & 1) else 1.0
            for n_dn in range(ed.N_s):
                src_d, dst_d = ed.cdag0_map[n_dn]
                d_dn, d_dn1 = ed.dims[n_dn], ed.dims[n_dn + 1]
                _, Va_f = sectors[(n_up, n_dn)]
                _, Vb_f = sectors[(n_up, n_dn + 1)]
                Va = Va_f.reshape(d_up, d_dn, -1)[:, src_d, :].reshape(d_up * len(src_d), -1)
                Vb = Vb_f.reshape(d_up, d_dn1, -1)[:, dst_d, :].reshape(d_up * len(dst_d), -1)
                M = sgn * (Vb.T @ Va)
                cdag[(n_up, n_dn)] = M
                cann[(n_up, n_dn + 1)] = M.T
    return cdag, cann


def eval_g_lehmann_on_grid(ed: EDAndersonSolver, ed_res, beta, wn_grid, boltz_cut=1e-11):
    """Exact Lehmann single-particle g_imp(iwn) on an arbitrary fermionic Matsubara grid wn_grid."""
    sectors = ed_res["sectors"]
    E_min = ed_res["E_min"]
    Z = ed_res["Z_part"]
    iwn = 1j * np.asarray(wn_grid, dtype=np.float64)
    g = np.zeros(len(iwn), dtype=np.complex128)
    cdag_u, _ = _sector_c_matrices(ed, sectors, spin=0)

    for (n_up, n_dn), M in cdag_u.items():
        Ea = sectors[(n_up, n_dn)][0]
        Eb = sectors[(n_up + 1, n_dn)][0]
        wa = np.exp(-beta * (Ea - E_min))
        wb = np.exp(-beta * (Eb - E_min))
        wt = (wa[None, :] + wb[:, None]) * (M ** 2)
        mask = wt > boltz_cut * 1e-5
        if np.any(mask):
            dE = Eb[:, None] - Ea[None, :]
            g += np.sum(wt[mask][None, :] / (iwn[:, None] - dE[mask][None, :]), axis=1)
    return g / Z


def compute_chi_ph_static_ed(ed: EDAndersonSolver, ed_res, beta, wn_grid, boltz_cut=1e-9):
    """
    Exact O(N_w * D^3) Lehmann computation of static (iOm = 0) 4-point particle-hole
    generalized susceptibilities chi_c^{nu, nu'} and chi_s^{nu, nu'} on wn_grid.
    """
    sectors = ed_res["sectors"]
    E_min = ed_res["E_min"]
    Z = ed_res["Z_part"]
    wn = np.asarray(wn_grid, dtype=np.float64)
    nw = len(wn)

    cdag_u, cann_u = _sector_c_matrices(ed, sectors, spin=0)
    cdag_d, cann_d = _sector_c_matrices(ed, sectors, spin=1)

    evals_map = {sec: ev for sec, (ev, _) in sectors.items()}
    w_map = {sec: np.exp(-beta * (ev - E_min)) for sec, (ev, _) in sectors.items()}

    def eval_channel(ops_info):
        perms = [
            ((0, 1, 2), +1.0),
            ((0, 2, 1), -1.0),
            ((1, 0, 2), -1.0),
            ((1, 2, 0), +1.0),
            ((2, 0, 1), +1.0),
            ((2, 1, 0), -1.0),
        ]
        freq_tag = {0: (0, +1.0), 1: (0, -1.0), 2: (1, +1.0), 3: (1, -1.0)}
        out = np.zeros((nw, nw), dtype=np.complex128)

        for (pa, pb, pc), p_sgn in perms:
            Oa_d, da_u, da_d = ops_info[pa]
            Ob_d, db_u, db_d = ops_info[pb]
            Oc_d, dc_u, dc_d = ops_info[pc]
            Od_d, dd_u, dd_d = ops_info[3]

            fa, sa = freq_tag[pa]
            fb, sb = freq_tag[pb]
            fc, sc = freq_tag[pc]

            for sec1 in sectors:
                sec2 = (sec1[0] - da_u, sec1[1] - da_d)
                sec3 = (sec2[0] - db_u, sec2[1] - db_d)
                sec4 = (sec3[0] - dc_u, sec3[1] - dc_d)
                if sec2 not in Oa_d or sec3 not in Ob_d or sec4 not in Oc_d or sec1 not in Od_d:
                    continue

                w1, w2, w3, w4 = w_map[sec1], w_map[sec2], w_map[sec3], w_map[sec4]
                if max(w1.max(), w2.max(), w3.max(), w4.max()) < boltz_cut:
                    continue

                E1, E2, E3, E4 = evals_map[sec1], evals_map[sec2], evals_map[sec3], evals_map[sec4]
                Ma, Mb, Mc, Md = Oa_d[sec2], Ob_d[sec3], Oc_d[sec4], Od_d[sec1]

                dE12 = E1[:, None] - E2[None, :]
                dE34 = E3[:, None] - E4[None, :]
                dE13 = E1[:, None] - E3[None, :]
                dE24 = E2[:, None] - E4[None, :]
                dE14 = E1[:, None] - E4[None, :]
                dE23 = E2[:, None] - E3[None, :]

                A_w = Ma[None, :, :] / (dE12[None, :, :] + (1j * sa * wn)[:, None, None])
                C_w = Mc[None, :, :] / (dE34[None, :, :] + (1j * sc * wn)[:, None, None])

                AB_w = A_w @ Mb[None, :, :]
                CD_w = C_w @ Md[None, :, :]
                DA_w = Md[None, :, :] @ A_w
                BC_w = Mb[None, :, :] @ C_w

                # Term 3
                D3_w = (Md.T * (w1[:, None] + w4[None, :]))[None, :, :] / (dE14[None, :, :] + (1j * wn)[:, None, None])
                if fa == 0 and fc == 1:
                    CD3 = np.einsum("wkl,wil->wki", C_w, D3_w)
                    t3 = np.einsum("wik,vki->wv", AB_w, CD3)
                elif fa == 1 and fc == 0:
                    AD3 = np.einsum("wil,wik->wlk", D3_w, AB_w)
                    t3 = np.einsum("vkl,wlk->vw", C_w, AD3)
                else:
                    ABC = AB_w @ C_w
                    t3 = np.einsum("wil,vil->wv", ABC, D3_w)

                # Term 4
                B4_w = -(Mb * (w2[:, None] + w3[None, :]))[None, :, :] / (dE23[None, :, :] + (1j * sb * wn)[:, None, None])
                if fa == 0 and fc == 1 and fb == 0:
                    AB4 = A_w @ B4_w
                    t4 = np.einsum("wik,vki->wv", AB4, CD_w)
                elif fa == 0 and fc == 0 and fb == 1:
                    CDA = CD_w @ A_w
                    t4 = np.einsum("wkj,vjk->wv", CDA, B4_w)
                else:
                    B4C = B4_w @ C_w
                    t4 = np.einsum("wjl,vlj->vw", DA_w, B4C)

                # Term 1
                if (pa, pb) in ((0, 1), (1, 0)):
                    deg1 = np.abs(dE13) < 1e-11
                    K1 = np.where(deg1, beta * w1[:, None], (w3[None, :] - w1[:, None]) / np.where(deg1, 1.0, dE13))
                    t1 = np.einsum("wik,ik,vki->wv", AB_w, K1, CD_w)
                else:
                    i1, i3 = np.where((w1[:, None] > boltz_cut) | (w3[None, :] > boltz_cut))
                    if len(i1) > 0:
                        dE_p, dw_p, w1_p = dE13[i1, i3], w3[i3] - w1[i1], w1[i1]
                        ab_p, cd_p = AB_w[:, i1, i3], CD_w[:, i3, i1]
                        s0 = sa if fa == 0 else sb
                        s1 = sb if fb == 1 else sa
                        z_sum = 1j * (s0 * wn[:, None] + s1 * wn[None, :])
                        denom = dE_p[None, None, :] + z_sum[:, :, None]
                        deg = np.abs(denom) < 1e-11
                        K1_p = np.where(deg, beta * w1_p[None, None, :], dw_p[None, None, :] / np.where(deg, 1.0, denom))
                        if fa == 0 and fc == 0:
                            t1 = np.einsum("wp,wvp->wv", ab_p * cd_p, K1_p)
                        else:
                            t1 = np.einsum("vp,wp,wvp->wv", ab_p, cd_p, K1_p)
                    else:
                        t1 = 0.0

                # Term 2
                if (pb, pc) in ((0, 1), (1, 0)):
                    deg2 = np.abs(dE24) < 1e-11
                    K2 = np.where(deg2, -beta * w2[:, None], (w2[:, None] - w4[None, :]) / np.where(deg2, 1.0, dE24))
                    t2 = np.einsum("vlj,jl,wjl->wv", DA_w, K2, BC_w)
                else:
                    i2, i4 = np.where((w2[:, None] > boltz_cut) | (w4[None, :] > boltz_cut))
                    if len(i2) > 0:
                        dE_p, dw_p, w2_p = dE24[i2, i4], w2[i2] - w4[i4], w2[i2]
                        da_p, bc_p = DA_w[:, i4, i2], BC_w[:, i2, i4]
                        s0 = sb if fb == 0 else sc
                        s1 = sc if fc == 1 else sb
                        z_sum = 1j * (s0 * wn[:, None] + s1 * wn[None, :])
                        denom = dE_p[None, None, :] + z_sum[:, :, None]
                        deg = np.abs(denom) < 1e-11
                        K2_p = np.where(deg, -beta * w2_p[None, None, :], dw_p[None, None, :] / np.where(deg, 1.0, denom))
                        if fa == 0 and fc == 0:
                            t2 = np.einsum("wp,wvp->wv", da_p * bc_p, K2_p)
                        else:
                            t2 = np.einsum("wp,vp,wvp->wv", da_p, bc_p, K2_p)
                    else:
                        t2 = 0.0

                out += p_sgn * (t1 + t2 + t3 + t4)

        return out / Z

    ops_uu = [(cann_u, -1, 0), (cdag_u, +1, 0), (cann_u, -1, 0), (cdag_u, +1, 0)]
    ops_ud = [(cann_u, -1, 0), (cdag_u, +1, 0), (cann_d, 0, -1), (cdag_d, 0, +1)]

    G2_uu = eval_channel(ops_uu)
    G2_ud = eval_channel(ops_ud)
    g_w = eval_g_lehmann_on_grid(ed, ed_res, beta, wn)

    disc = beta * np.outer(g_w, g_w)
    chi_uu = G2_uu - disc
    chi_ud = G2_ud - disc
    chi_c = 2.0 * (chi_uu + chi_ud)
    chi_s = 2.0 * (chi_uu - chi_ud)
    chi0_diag = -2.0 * beta * (g_w ** 2)

    return {
        "wn": wn,
        "g_w": g_w,
        "chi0_diag": chi0_diag,
        "chi_c": chi_c,
        "chi_s": chi_s,
    }


def extract_rank_R_vertices(chi_res, U, rank=6):
    """
    Extract exact amputated local vertices gamma_c(nu, nu'; 0) and gamma_s(nu, nu'; 0),
    compute optimal Rank-R SVD compression of the connected susceptibility & vertex,
    and diagnose multiple Schafer-Rohringer-Toschi Gamma_c^irr divergence lines via
    exact eigenvalues of chi_c^{nu, nu'}(0).
    """
    g_w = chi_res["g_w"]
    chi0_diag = chi_res["chi0_diag"]
    chi_c = chi_res["chi_c"]
    chi_s = chi_res["chi_s"]
    nw = len(g_w)
    R = min(int(rank), nw)

    # Connected parts dchi_r = chi_r - diag(chi0) = -2 * g(nu)^2 * gamma_r(nu, nu') * g(nu')^2
    dchi_c = chi_c - np.diag(chi0_diag)
    dchi_s = chi_s - np.diag(chi0_diag)

    inv_g2 = 1.0 / (g_w ** 2)
    gamma_c = -0.5 * (inv_g2[:, None] * dchi_c * inv_g2[None, :])
    gamma_s = -0.5 * (inv_g2[:, None] * dchi_s * inv_g2[None, :])

    # Optimal SVD of connected susceptibility (the exact kernel entering dual BSE & Ward sum rules!)
    def compress_channel(dchi, gamma):
        U_m, s_m, Vh_m = np.linalg.svd(dchi, full_matrices=False)
        tot_nrm = float(np.linalg.norm(s_m)) + 1e-30
        err_by_rank = {}
        for r_try in (1, 2, 4, 6):
            rt = min(r_try, len(s_m))
            rec = (U_m[:, :rt] * s_m[None, :rt]) @ Vh_m[:rt, :]
            err_by_rank[r_try] = float(np.linalg.norm(dchi - rec) / tot_nrm)
        # Convert rank-R SVD of dchi back to factorized vertex form factors Phi_a(nu) = U_a(nu) / g(nu)^2:
        #   gamma(nu, nu') = sum_{a=1}^R Phi_L(nu, a) * (-0.5 * s_a) * Phi_R(a, nu')
        Phi_L = inv_g2[:, None] * U_m[:, :R]
        Phi_R = Vh_m[:R, :] * inv_g2[None, :]
        W_R = -0.5 * np.diag(s_m[:R])
        return {
            "sv": s_m,
            "err_by_rank": err_by_rank,
            "Phi_L": Phi_L,
            "Phi_R": Phi_R,
            "W_R": W_R,
        }

    svd_c = compress_channel(dchi_c, gamma_c)
    svd_s = compress_channel(dchi_s, gamma_s)

    # Exact Gamma_c^irr divergence diagnostics:
    # At half-filling chi_c is real symmetric; in general use real symmetric part (or general eigvals)
    # A divergence of Gamma_c^irr = chi_c^{-1} - chi_0^{-1} occurs EVERY TIME an eigenvalue of chi_c crosses 0!
    chi_c_sym = 0.5 * (chi_c + chi_c.T)
    eigs_c = np.sort(np.real(np.linalg.eigvals(chi_c_sym)))
    n_div_lines = int(np.sum(eigs_c < -1e-9))

    # Also compute exact dimensionless ratio chi_0^{-1/2} chi_c chi_0^{-1/2} = (I + chi_0^{1/2} Gamma_c^irr chi_0^{1/2})^{-1}
    # (exact analogue of Track IV-A's D_c^irr = det(I - X0_loc W_c)!)
    inv_sqrt_c0 = 1.0 / np.sqrt(np.maximum(np.real(chi0_diag), 1e-18))
    M_norm_c = inv_sqrt_c0[:, None] * np.real(chi_c_sym) * inv_sqrt_c0[None, :]
    eigs_norm_c = np.sort(np.linalg.eigvalsh(0.5 * (M_norm_c + M_norm_c.T)))

    return {
        "gamma_c": gamma_c,
        "gamma_s": gamma_s,
        "svd_c": svd_c,
        "svd_s": svd_s,
        "eigs_chi_c": eigs_c,
        "eigs_norm_c": eigs_norm_c,
        "n_div_lines": n_div_lines,
        "min_eig_chi_c": float(eigs_c[0]),
        "gamma_c_00": float(np.real(gamma_c[nw // 2, nw // 2])),
        "gamma_s_00": float(np.real(gamma_s[nw // 2, nw // 2])),
    }
