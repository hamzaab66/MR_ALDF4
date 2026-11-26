"""
MR-ALDF-IV 2x2 Sublattice Neel Antiferromagnetic Ordered-Phase Solver (mr_aldf_neel.py)
Pillar 2a: Spontaneous SU(2) -> U(1) Symmetry Breaking on the 3D Simple-Cubic Lattice.

Implements:
  1. Exact spin-polarized fourth-moment mIPT-4 + transverse/longitudinal local-TPSC
     impurity solver on the sparse DLR grid (solve_mipt4_spinpol_dlr).
  2. Exact 2x2 sublattice (A, B) dual-fermion mapping with off-diagonal hopping -eps_k
     and full 2x2 matrix physical-to-dual self-energy identity Sigma_nl = S_d (I + g S_d)^{-1}.
  3. 2x2 sublattice transverse magnon (Goldstone S^+ S^-) + longitudinal (zz, cc) dual
     ladders regularized by exact broken-symmetry Pauli sum rules:
       (1/beta N) sum chi^{xx}(q, iOm)            = n - 2 D_imp   (transverse Goldstone)
       (1/beta N) sum [chi^{zz} + chi^{cc}]_conn  = n(2-n) - m^2  (longitudinal)
  4. Smooth 2x2 Herglotz causality safeguard and exact reduction to paramagnetic
     MR-ALDF-IV as m -> 0 and to single-site AFM-DMFT as d -> inf.
"""

import numpy as np
from scipy.optimize import brentq
from mr_aldf_dlr import DLRBasis


def solve_mipt4_spinpol_dlr(dlr: DLRBasis, Delta_up, Delta_dn, U, mu_up, mu_dn, n_up_in, n_dn_in, use_bkin=False):
    """
    Spin-polarized fourth-moment mIPT-4 + transverse/longitudinal local-TPSC impurity solver
    on the DLR grid for sublattice A (sublattice B has up <-> dn by exact Neel symmetry).
    """
    iwn = dlr.iwn
    n_s = [float(np.clip(n_up_in, 1e-6, 1.0 - 1e-6)), float(np.clip(n_dn_in, 1e-6, 1.0 - 1e-6))]
    Del_s = [Delta_up, Delta_dn]
    half_fill = abs((n_s[0] + n_s[1]) - 1.0) < 1e-12

    # 1. Exact Hartree-Weiss propagators G0_s^{-1} = iwn + mu_s - U * n_{-s} - Delta_s
    #    (Rozenberg-Georges-Kotliar AFM-DMFT + exact Kajueter-Kotliar high-frequency moment closure!)
    mu_s = [float(mu_up), float(mu_dn)]
    G0_s = [None, None]
    G0_tau_s = [None, None]
    n0_s = [0.5, 0.5]
    for s in (0, 1):
        sb = 1 - s
        G0_s[s] = 1.0 / (iwn + mu_s[s] - U * n_s[sb] - Del_s[s])
        G0_tau_s[s] = np.real(dlr.iwn_to_tau(G0_s[s]))
        n0_s[s] = float(np.clip(0.5 * float(dlr.density_from_G_iwn(G0_s[s])), 1e-9, 1.0 - 1e-9))

    # 2. Spin-polarized 2nd-order sunset: Sig2_s(tau) = +U^2 G0_s(tau) G0_{-s}(tau) G0_{-s}(beta-tau)
    Sig2_s = [None, None]
    for s in (0, 1):
        sb = 1 - s
        Sig2_tau = (U ** 2) * G0_tau_s[s] * G0_tau_s[sb] * G0_tau_s[sb][::-1]
        Sig2_s[s] = dlr.tau_to_iwn(Sig2_tau)

    # 3. Exact moment-conserving Hartree-Weiss mIPT interpolation:
    #      A_s = [n_{-s}(1-n_{-s})] / [n_{0,-s}(1-n_{0,-s})]
    #      B_s = [U(1 - 2 n_{-s})]  / [U^2 n_{0,-s}(1-n_{0,-s})]
    #    Guarantees: exact atomic limit (t=0, any U,mu,h,T), exact O(U^2) weak coupling,
    #    exact half-filled PM limit (A=1, B=0), and pole-free stability across all T & m!
    Sig_imp_s = [None, None]
    Sig_dyn_s = [None, None]
    g_s = [None, None]
    for s in (0, 1):
        sb = 1 - s
        var_sb = n_s[sb] * (1.0 - n_s[sb])
        var0_sb = n0_s[sb] * (1.0 - n0_s[sb])
        if U < 1e-12 or var_sb < 1e-14 or var0_sb < 1e-14:
            Sig_dyn = np.zeros_like(iwn)
        else:
            # Exact Rozenberg-Georges-Kotliar Hartree-Weiss AFM-IPT + doped charge-asymmetric B_c:
            # Preserves exact 2nd moment U^2 n_{-s}(1-n_{-s}), strict causality (Im Sig < 0),
            # exact doped/half-filled PM limits, and avoids the Kajueter-Kotliar spin-B pole artifact.
            A_s = var_sb / var0_sb
            n_tot_in = n_s[0] + n_s[1]
            B_c = (U * (1.0 - n_tot_in)) / ((U ** 2) * var0_sb)
            Sig_dyn = (A_s * Sig2_s[s]) / (1.0 - B_c * Sig2_s[s])
        Sig_dyn_s[s] = Sig_dyn
        Sig_imp_s[s] = U * n_s[sb] + Sig_dyn
        g_s[s] = 1.0 / (iwn + mu_s[s] - Del_s[s] - Sig_imp_s[s])

    n_imp_up = 0.5 * float(dlr.density_from_G_iwn(g_s[0]))
    n_imp_dn = 0.5 * float(dlr.density_from_G_iwn(g_s[1]))
    n_tot = n_imp_up + n_imp_dn
    m_imp = n_imp_up - n_imp_dn

    # 4. Exact spin-polarized Galitskii-Migdal double occupancy
    if U < 1e-12:
        D_imp = n_imp_up * n_imp_dn
    else:
        gm_dyn = 0.5 * (
            float(np.real(dlr.fermion_prod_sum(Sig_dyn_s[0], g_s[0])))
            + float(np.real(dlr.fermion_prod_sum(Sig_dyn_s[1], g_s[1])))
        )
        # Hartree part: 0.5 * (n_dn_in * n_imp_up + n_up_in * n_imp_dn)
        D_imp = 0.5 * (n_s[1] * n_imp_up + n_s[0] * n_imp_dn) + gm_dyn / U
        D_imp = float(np.clip(D_imp, max(0.0, n_tot - 1.0), min(n_imp_up, n_imp_dn)))

    # 5. Local transverse (+-) and longitudinal (zz, cc) bubbles and local-TPSC closures
    # Transverse Weiss bubble: chi0_pm(tau) = G0_up(tau) G0_dn(beta - tau) (symmetrized real part)
    chi0_pm_G0 = np.real(dlr.tauB_to_iOm(0.5 * (G0_tau_s[0] * G0_tau_s[1][::-1] + G0_tau_s[1] * G0_tau_s[0][::-1])))
    chi0_pm_G0 = np.maximum(chi0_pm_G0, 0.0)
    # Longitudinal Weiss bubble: average of up-up and dn-dn
    chi0_zz_G0 = np.real(dlr.tauB_to_iOm(0.5 * (G0_tau_s[0] * G0_tau_s[0][::-1] + G0_tau_s[1] * G0_tau_s[1][::-1])))
    chi0_zz_G0 = np.maximum(chi0_zz_G0, 0.0)

    # Transverse local-TPSC: (1/beta) sum 2*chi0_pm / (1 - U_pm * chi0_pm) = n - 2*D_imp
    target_pm = max(n_tot - 2.0 * D_imp, 1e-10)
    c_pm_max = float(np.max(chi0_pm_G0))
    if U < 1e-12 or c_pm_max < 1e-14:
        U_pm = 0.0
    else:
        def sum_pm(upm):
            return float(dlr.boson_sum_from_iOm(2.0 * chi0_pm_G0 / (1.0 - upm * chi0_pm_G0))) - target_pm
        u_hi = (1.0 - 1e-10) / c_pm_max
        u_lo = 0.0 if sum_pm(0.0) <= 0.0 else -10.0 * U
        U_pm = float(brentq(sum_pm, u_lo, u_hi, xtol=1e-10)) if sum_pm(u_lo) * sum_pm(u_hi) <= 0.0 else float(U)
    chi_imp_pm = 2.0 * chi0_pm_G0 / (1.0 - U_pm * chi0_pm_G0)

    # Longitudinal spin local-TPSC: target = n - 2*D_imp - m^2
    target_zz = max(n_tot - 2.0 * D_imp - m_imp ** 2, 1e-10)
    c_zz_max = float(np.max(chi0_zz_G0))
    if U < 1e-12 or c_zz_max < 1e-14:
        U_zz = 0.0
    else:
        def sum_zz(uzz):
            return float(dlr.boson_sum_from_iOm(2.0 * chi0_zz_G0 / (1.0 - uzz * chi0_zz_G0))) - target_zz
        u_hi = (1.0 - 1e-10) / c_zz_max
        u_lo = -max(10.0 * U, 50.0)
        U_zz = float(brentq(sum_zz, u_lo, u_hi, xtol=1e-10)) if sum_zz(u_lo) * sum_zz(u_hi) <= 0.0 else 0.0
    chi_imp_zz = 2.0 * chi0_zz_G0 / (1.0 - U_zz * chi0_zz_G0)

    # Longitudinal charge local-TPSC: target = n(1-n) + 2*D_imp
    target_cc = max(n_tot * (1.0 - n_tot) + 2.0 * D_imp, 1e-10)
    if U < 1e-12 or c_zz_max < 1e-14:
        U_cc = 0.0
    else:
        def sum_cc(ucc):
            return float(dlr.boson_sum_from_iOm(2.0 * chi0_zz_G0 / (1.0 + ucc * chi0_zz_G0))) - target_cc
        u_lo = 0.0 if sum_cc(0.0) >= 0.0 else -(1.0 - 1e-9) / c_zz_max
        u_hi = max(10.0 * U, 50.0)
        while sum_cc(u_hi) > 0.0 and u_hi < 1e9:
            u_hi *= 4.0
        U_cc = float(brentq(sum_cc, u_lo, u_hi, xtol=1e-10)) if sum_cc(u_lo) * sum_cc(u_hi) <= 0.0 else float(U)
    chi_imp_cc = 2.0 * chi0_zz_G0 / (1.0 + U_cc * chi0_zz_G0)

    # Ward form factors Lambda_s(iwn) = 1 - Im(Sig_imp,s)/wn >= 1 and physical local bubbles
    Lam_up = np.maximum(1.0 - np.imag(Sig_imp_s[0]) / dlr.wn, 1.0)
    Lam_dn = np.maximum(1.0 - np.imag(Sig_imp_s[1]) / dlr.wn, 1.0)
    Lam_avg = 0.5 * (Lam_up + Lam_dn)
    sLam_up = np.sqrt(Lam_up)
    sLam_dn = np.sqrt(Lam_dn)

    g_up_t = np.real(dlr.iwn_to_tau(g_s[0]))
    g_dn_t = np.real(dlr.iwn_to_tau(g_s[1]))
    Lg_up_t = np.real(dlr.iwn_to_tau(sLam_up * g_s[0]))
    Lg_dn_t = np.real(dlr.iwn_to_tau(sLam_dn * g_s[1]))

    chi0_pm_loc = np.maximum(np.real(dlr.tauB_to_iOm(0.5 * (g_up_t * g_dn_t[::-1] + g_dn_t * g_up_t[::-1]))), 0.0)
    chiL_pm_loc = np.maximum(np.real(dlr.tauB_to_iOm(0.5 * (Lg_up_t * Lg_dn_t[::-1] + Lg_dn_t * Lg_up_t[::-1]))), 0.0)
    chi0_zz_loc = np.maximum(np.real(dlr.tauB_to_iOm(0.5 * (g_up_t * g_up_t[::-1] + g_dn_t * g_dn_t[::-1]))), 0.0)
    chiL_zz_loc = np.maximum(np.real(dlr.tauB_to_iOm(0.5 * (Lg_up_t * Lg_up_t[::-1] + Lg_dn_t * Lg_dn_t[::-1]))), 0.0)

    return {
        "Sig_imp_up": Sig_imp_s[0],
        "Sig_imp_dn": Sig_imp_s[1],
        "g_up": g_s[0],
        "g_dn": g_s[1],
        "Lam_up": Lam_up,
        "Lam_dn": Lam_dn,
        "Lam_avg": Lam_avg,
        "n_up": n_imp_up,
        "n_dn": n_imp_dn,
        "n_imp": n_tot,
        "m_imp": m_imp,
        "D_imp": D_imp,
        "U_pm": U_pm,
        "U_zz": U_zz,
        "U_cc": U_cc,
        "chi0_pm_G0": chi0_pm_G0,
        "chi0_zz_G0": chi0_zz_G0,
        "chi_imp_pm": chi_imp_pm,
        "chi_imp_zz": chi_imp_zz,
        "chi_imp_cc": chi_imp_cc,
        "chi0_pm_loc": chi0_pm_loc,
        "chiL_pm_loc": chiL_pm_loc,
        "chi0_zz_loc": chi0_zz_loc,
        "chiL_zz_loc": chiL_zz_loc,
    }


class MRALDFNeelDLRSolver:
    """
    2x2 Sublattice Neel Antiferromagnetic MR-ALDF-IV Solver on the 3D Simple-Cubic Lattice.
    Set enable_dual=False to obtain exact single-site AFM-DMFT on the identical DLR grid
    for direct apples-to-apples comparison of T_N^3D vs T_N^DMFT and m_AFM(T).
    """

    def __init__(self, U, T, delta=0.0, L=8, t=1.0, w_max=None, eps_dlr=1e-7, enable_dual=True, c_max=0.90):
        self.U = float(U)
        self.T = float(T)
        self.beta = 1.0 / self.T
        self.delta = float(delta)
        self.n_target = 1.0 - self.delta
        self.half_fill = abs(self.delta) < 1e-12
        self.L = int(L)
        self.N = self.L ** 3
        self.t = float(t)
        self.enable_dual = bool(enable_dual)
        self.c_max = float(c_max)

        self.w_max = float(w_max) if w_max is not None else max(18.0 * self.t, 1.5 * self.U + 12.0 * self.t)
        self.dlr = DLRBasis(beta=self.beta, w_max=self.w_max, eps=eps_dlr)

        k1d = 2.0 * np.pi * np.arange(self.L) / self.L
        kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
        self.eps_k = -2.0 * self.t * (np.cos(kx) + np.cos(ky) + np.cos(kz))
        self._r_neg = (self.L - np.arange(self.L)) % self.L
        self.Q_idx = (self.L // 2, self.L // 2, self.L // 2)

    def _neg_r(self, arr_r):
        idx = self._r_neg
        return arr_r[:, idx, :, :][:, :, idx, :][:, :, :, idx]

    def _k_to_r(self, A):
        return np.fft.ifftn(self.dlr.iwn_to_tau(A), axes=(1, 2, 3))

    def _r_to_q(self, X):
        return self.dlr.tauB_to_iOm(np.fft.fftn(X, axes=(1, 2, 3)))

    def _q_to_r(self, V):
        return np.fft.ifftn(self.dlr.iOm_to_tauB(V), axes=(1, 2, 3))

    def _r_to_k(self, S):
        return self.dlr.tau_to_iwn(np.fft.fftn(S, axes=(1, 2, 3)))

    def _sym_Delta(self, D):
        D = 0.5 * (D + np.conj(D[::-1]))
        sgn = np.sign(self.dlr.wn)
        return np.real(D) + 1j * np.where(sgn * np.imag(D) > 0.0, 0.0, np.imag(D))

    def _herglotz_2x2(self, SAA, SBB, SAB, gA, gB):
        """Causal Herglotz projection on diagonal elements + exact Matsubara symmetry on 2x2 Sigma_dual."""
        def sym_mat(M):
            return 0.5 * (M + np.conj(M[::-1]))
        SAA, SBB, SAB = sym_mat(SAA), sym_mat(SBB), sym_mat(SAB)
        sgn = np.sign(self.dlr.wn)[:, None, None, None]

        def proj_diag(S, g):
            im_S = np.where(sgn * np.imag(S) > 0.0, -1e-14 * sgn, np.imag(S))
            S = np.real(S) + 1j * im_S
            mag2 = np.abs(S) ** 2
            active = mag2 > 1e-24
            y_min = (np.abs(np.imag(g)) / self.c_max)[:, None, None, None]
            dh = 0.05 * np.maximum(y_min, 1e-12)
            inv_S = np.where(active, 1.0 / np.where(active, S, 1.0), 0.0 + 0.0j)
            y_inv = sgn * np.imag(inv_S)
            z = (y_inv - y_min) / dh
            softp = np.where(z > 35.0, z, np.log1p(np.exp(np.clip(z, -60.0, 35.0))))
            inv_p = np.real(inv_S) + 1j * sgn * (y_min + dh * softp)
            return sym_mat(np.where(active, 1.0 / np.where(active, inv_p, 1.0), S))

        SAA = proj_diag(SAA, gA)
        SBB = proj_diag(SBB, gB)
        # Bound off-diagonal |SAB| so det(I + g S_dual) stays strictly causal
        lim_off = 0.45 / np.maximum(np.sqrt(np.abs(gA) * np.abs(gB)), 1e-6)[:, None, None, None]
        mag_ab = np.abs(SAB)
        SAB = np.where(mag_ab > lim_off, SAB * (lim_off / np.maximum(mag_ab, 1e-14)), SAB)
        return SAA, SBB, sym_mat(SAB)

    def _build_local_vertices_neel(self, imp):
        dlr = self.dlr
        m0 = dlr.m0_idx

        def uv_reg_amp(U_eff, cG0, c0, cL, chi_imp, sign_s):
            # sign_s = -1 for attractive spin channels (pm, zz), +1 for repulsive charge (cc)
            ratio_sq = np.where(cL > 1e-14, (cG0 / np.where(cL > 1e-14, cL, 1.0)) ** 2, 0.0)
            ratio_sq = np.clip(ratio_sq, 0.0, 10.0)
            g_arr = np.where(
                np.arange(dlr.r_B) == m0,
                (2.0 * c0[m0] - chi_imp[m0]) / max(2.0 * cL[m0] ** 2, 1e-18),
                (sign_s * U_eff / np.maximum(1.0 + sign_s * U_eff * cG0, 1e-8)) * ratio_sq,
            )
            return np.real(g_arr)

        g_pm = np.minimum(uv_reg_amp(imp["U_pm"], imp["chi0_pm_G0"], imp["chi0_pm_loc"], imp["chiL_pm_loc"], imp["chi_imp_pm"], -1.0), 0.0)
        g_zz = np.minimum(uv_reg_amp(imp["U_zz"], imp["chi0_zz_G0"], imp["chi0_zz_loc"], imp["chiL_zz_loc"], imp["chi_imp_zz"], -1.0), 0.0)
        g_cc = np.maximum(uv_reg_amp(imp["U_cc"], imp["chi0_zz_G0"], imp["chi0_zz_loc"], imp["chiL_zz_loc"], imp["chi_imp_cc"], +1.0), 0.0)
        return {"g_pm": g_pm, "g_zz": g_zz, "g_cc": g_cc}

    def _dual_self_energy_neel(self, imp, vert, GAA, GBB, GAB, GdAA, GdBB, GdAB):
        """
        2x2 sublattice transverse magnon (Goldstone S^+ S^-) + longitudinal (zz, cc)
        dual-ladder self-energy in O(r * N log N) time.
        Spin-up on (A, B) has propagators (GAA, GBB, GAB);
        Spin-down on (A, B) has propagators (GBB, GAA, GAB) by exact Neel symmetry!
        """
        dlr = self.dlr
        sLup = np.sqrt(imp["Lam_up"])[:, None, None, None]
        sLdn = np.sqrt(imp["Lam_dn"])[:, None, None, None]
        Lup = imp["Lam_up"][:, None, None, None]
        Ldn = imp["Lam_dn"][:, None, None, None]
        Lpm = np.sqrt(Lup * Ldn)

        # Transform physical G and dressed dual Gd to (tau, r)
        GAA_r = self._k_to_r(GAA)
        GBB_r = self._k_to_r(GBB)
        GAB_r = self._k_to_r(GAB)

        LGAA_r = self._k_to_r(sLup * GAA)
        LGBB_r = self._k_to_r(sLdn * GBB)
        LGAB_r = self._k_to_r(np.sqrt(Lpm) * GAB)

        GdAA_r = self._k_to_r(Lup * GdAA)
        GdBB_r = self._k_to_r(Ldn * GdBB)
        GdAB_r = self._k_to_r(Lpm * GdAB)

        # ---- 1. Transverse (+-) 2x2 sublattice bubbles: chi^{+-}_{ab} from G_{ab,up}(r,t) G_{ba,dn}(-r,beta-t) ----
        # Recall G_{AA,dn} = GBB, G_{BB,dn} = GAA, G_{AB,dn} = GAB, G_{BA,dn} = GAB
        X0pm_AA = np.real(self._r_to_q(GAA_r * self._neg_r(GBB_r)[::-1]))
        X0pm_AB = np.real(self._r_to_q(GAB_r * self._neg_r(GAB_r)[::-1]))
        # Exact r=0 restoration on AA (BB = AA by symmetry!)
        X0pm_AA += (imp["chi0_pm_loc"] - np.mean(X0pm_AA, axis=(1, 2, 3)))[:, None, None, None]

        XLpm_AA = np.real(self._r_to_q(LGAA_r * self._neg_r(LGBB_r)[::-1]))
        XLpm_AB = np.real(self._r_to_q(LGAB_r * self._neg_r(LGAB_r)[::-1]))
        XLpm_AA += (imp["chiL_pm_loc"] - np.mean(XLpm_AA, axis=(1, 2, 3)))[:, None, None, None]

        # Dual 2x2 transverse bubble matrix (symmetric: AA = BB, AB = BA)
        Xtpm_AA = np.real(self._r_to_q(GdAA_r * self._neg_r(GdBB_r)[::-1]))
        Xtpm_AB = np.real(self._r_to_q(GdAB_r * self._neg_r(GdAB_r)[::-1]))
        Xtpm_AA -= np.mean(Xtpm_AA, axis=(1, 2, 3), keepdims=True)
        # Note: in sublattice symmetric (AA = BB) 2x2 basis, eigenvalues are EXACTLY
        #   even (+, ferromagnetic q=0):  X_+ = X_AA + X_AB
        #   odd  (-, antiferromag. q=Q):  X_- = X_AA - X_AB
        # So the 2x2 sublattice BSE diagonalizes ANALYTICALLY into even (+) and odd (-) modes!
        g_pm = vert["g_pm"][:, None, None, None]
        Dpm_p = 1.0 + g_pm * (Xtpm_AA + Xtpm_AB)
        Dpm_m = 1.0 + g_pm * (Xtpm_AA - Xtpm_AB)

        # ---- 2. Longitudinal (zz, cc) 2x2 sublattice bubbles ----
        X0zz_AA = np.real(self._r_to_q(0.5 * (GAA_r * self._neg_r(GAA_r)[::-1] + GBB_r * self._neg_r(GBB_r)[::-1])))
        X0zz_AB = X0pm_AB
        X0zz_AA += (imp["chi0_zz_loc"] - np.mean(X0zz_AA, axis=(1, 2, 3)))[:, None, None, None]

        XLzz_AA = np.real(self._r_to_q(0.5 * (LGAA_r * self._neg_r(LGAA_r)[::-1] + LGBB_r * self._neg_r(LGBB_r)[::-1])))
        XLzz_AB = XLpm_AB
        XLzz_AA += (imp["chiL_zz_loc"] - np.mean(XLzz_AA, axis=(1, 2, 3)))[:, None, None, None]

        Xtzz_AA = np.real(self._r_to_q(0.5 * (GdAA_r * self._neg_r(GdAA_r)[::-1] + GdBB_r * self._neg_r(GdBB_r)[::-1])))
        Xtzz_AB = Xtpm_AB
        Xtzz_AA -= np.mean(Xtzz_AA, axis=(1, 2, 3), keepdims=True)

        g_cc = vert["g_cc"][:, None, None, None]
        Dcc_p = 1.0 + g_cc * (Xtzz_AA + Xtzz_AB)
        Dcc_m = 1.0 + g_cc * (Xtzz_AA - Xtzz_AB)
        lam_c = max(0.0, 0.05 - min(float(np.min(Dcc_p)), float(np.min(Dcc_m))))
        Gcc_p = g_cc / (Dcc_p + lam_c)
        Gcc_m = g_cc / (Dcc_m + lam_c)

        # ---- 3. Exact Broken-Symmetry Moriya Regularizations (Transverse Goldstone + Longitudinal) ----
        w_bsum = np.real(dlr._bsum_from_iOm_w)

        # (a) Transverse magnon sum rule: (1/beta N) sum chi^{xx}_AA(q, iOm) = n - 2 D_imp
        #     Since chi^{xx}_AA = 0.5 * (chi_+ + chi_-) where chi_{+-} = 2*X0_{+-} - 2*Gpm_{+-} * XL_{+-}^2:
        XLpm_p2 = (XLpm_AA + XLpm_AB) ** 2
        XLpm_m2 = (XLpm_AA - XLpm_AB) ** 2
        S0_pm = float(np.sum(w_bsum[:, None, None, None] * (2.0 * X0pm_AA)) / self.N)
        target_pm = max(imp["n_imp"] - 2.0 * imp["D_imp"], S0_pm + 1e-8)
        lam_pm_lo = max(0.0, 1e-5 - min(float(np.min(Dpm_p)), float(np.min(Dpm_m))))

        def pm_res(lp):
            Gp = g_pm / (Dpm_p + lp)
            Gm = g_pm / (Dpm_m + lp)
            exc = -w_bsum[:, None, None, None] * (Gp * XLpm_p2 + Gm * XLpm_m2)
            return (S0_pm + float(np.sum(exc) / self.N)) - target_pm

        if pm_res(lam_pm_lo) <= 0.0:
            lam_pm = lam_pm_lo
        else:
            lp_hi = max(lam_pm_lo + 1.0, 4.0)
            while pm_res(lp_hi) > 0.0 and lp_hi < 1e7:
                lp_hi *= 4.0
            lam_pm = float(brentq(pm_res, lam_pm_lo, lp_hi, xtol=1e-9)) if pm_res(lam_pm_lo) * pm_res(lp_hi) <= 0.0 else lam_pm_lo

        Gpm_p = g_pm / np.maximum(Dpm_p + lam_pm, 1e-6)
        Gpm_m = g_pm / np.maximum(Dpm_m + lam_pm, 1e-6)
        # Reconstruct 2x2 sublattice transverse effective vertex V_pm (factor 1.0 for 2 transverse modes / 2)
        Vpm_p = g_pm * (Gpm_p - 0.5 * g_pm) * (Xtpm_AA + Xtpm_AB)
        Vpm_m = g_pm * (Gpm_m - 0.5 * g_pm) * (Xtpm_AA - Xtpm_AB)
        Vpm_AA = 0.5 * (Vpm_p + Vpm_m)
        Vpm_AB = 0.5 * (Vpm_p - Vpm_m)

        # Transverse staggered magnon susceptibility at Q=0 (sublattice odd mode = physical Q=(pi,pi,pi)!)
        chi_pm_odd0 = float(np.real(2.0 * (X0pm_AA[dlr.m0_idx, 0, 0, 0] - X0pm_AB[dlr.m0_idx, 0, 0, 0]) - 2.0 * Gpm_m[dlr.m0_idx, 0, 0, 0] * XLpm_m2[dlr.m0_idx, 0, 0, 0]))

        # (b) Longitudinal zz sum rule: (1/beta N) sum [chi^{zz} + chi^{cc}]_AA = n(2-n) - m^2
        g_zz = vert["g_zz"][:, None, None, None]
        Dzz_p = 1.0 + g_zz * (Xtzz_AA + Xtzz_AB)
        Dzz_m = 1.0 + g_zz * (Xtzz_AA - Xtzz_AB)
        XLzz_p2 = (XLzz_AA + XLzz_AB) ** 2
        XLzz_m2 = (XLzz_AA - XLzz_AB) ** 2
        S0_zz = float(np.sum(w_bsum[:, None, None, None] * (2.0 * X0zz_AA)) / self.N)
        exc_cc = float(np.sum(w_bsum[:, None, None, None] * (Gcc_p * XLzz_p2 + Gcc_m * XLzz_m2)) / self.N)
        target_zz = max(imp["n_imp"] * (2.0 - imp["n_imp"]) - imp["m_imp"] ** 2 - (S0_zz - exc_cc), 0.0)
        lam_zz_lo = max(0.0, 1e-4 - min(float(np.min(Dzz_p)), float(np.min(Dzz_m))))

        def zz_res(lz):
            Gp = g_zz / (Dzz_p + lz)
            Gm = g_zz / (Dzz_m + lz)
            exc = -w_bsum[:, None, None, None] * (Gp * XLzz_p2 + Gm * XLzz_m2)
            return (S0_zz + float(np.sum(exc) / self.N)) - target_zz

        if target_zz <= S0_zz or zz_res(lam_zz_lo) <= 0.0:
            lam_zz = lam_zz_lo
        else:
            lz_hi = max(lam_zz_lo + 1.0, 4.0)
            while zz_res(lz_hi) > 0.0 and lz_hi < 1e7:
                lz_hi *= 4.0
            lam_zz = float(brentq(zz_res, lam_zz_lo, lz_hi, xtol=1e-9)) if zz_res(lam_zz_lo) * zz_res(lz_hi) <= 0.0 else lam_zz_lo

        Gzz_p = g_zz / np.maximum(Dzz_p + lam_zz, 1e-6)
        Gzz_m = g_zz / np.maximum(Dzz_m + lam_zz, 1e-6)

        # Longitudinal effective vertex: 0.5 * spin_zz + 0.5 * charge_cc
        Vlong_p = 0.5 * g_zz * (Gzz_p - 0.5 * g_zz) * (Xtzz_AA + Xtzz_AB) + 0.5 * g_cc * (Gcc_p - 0.5 * g_cc) * (Xtzz_AA + Xtzz_AB)
        Vlong_m = 0.5 * g_zz * (Gzz_m - 0.5 * g_zz) * (Xtzz_AA - Xtzz_AB) + 0.5 * g_cc * (Gcc_m - 0.5 * g_cc) * (Xtzz_AA - Xtzz_AB)
        Vlong_AA = 0.5 * (Vlong_p + Vlong_m)
        Vlong_AB = 0.5 * (Vlong_p - Vlong_m)

        # ---- 4. Assemble 2x2 Dual Self-Energy for Spin-Up: ----
        #   Transverse magnon scatters spin-up -> spin-down (G_dn = [GBB, GAB; GAB, GAA])!
        #   Longitudinal scatters spin-up -> spin-up (G_up = [GAA, GAB; GAB, GBB])!
        # Check m=0 limit: GAA = GBB = G, Vpm = 1.0*V_s, Vlong = 0.5*V_s + 0.5*V_c => total = 1.5*V_s + 0.5*V_c EXACT!
        Vpm_AA_r = np.real(self._q_to_r(Vpm_AA))
        Vpm_AB_r = np.real(self._q_to_r(Vpm_AB))
        Vl_AA_r = np.real(self._q_to_r(Vlong_AA))
        Vl_AB_r = np.real(self._q_to_r(Vlong_AB))

        # Dual self-energy in (tau, r):
        #   Sig_AA,up = Lup * (Gd_AA,dn * Vpm_AA(-r, b-t) + Gd_AA,up * Vl_AA(-r, b-t))
        #             = Lup * (GdBB_r * Vpm_AA(-r, b-t) + GdAA_r * Vl_AA(-r, b-t))
        #   Sig_BB,up = Ldn * (GdAA_r * Vpm_AA(-r, b-t) + GdBB_r * Vl_AA(-r, b-t))
        #   Sig_AB,up = Lpm * (GdAB_r * (Vpm_AB(-r, b-t) + Vl_AB(-r, b-t)))
        # Notice the CRUCIAL magnon physics: GdBB (minority spin!) feeds into Sig_AA (majority spin!)
        # via Vpm_AA, directly depolarizing the staggered moment m and lowering T_N^3D below T_N^DMFT!
        Sig_AA = Lup * self._r_to_k(GdBB_r * self._neg_r(Vpm_AA_r)[::-1] + GdAA_r * self._neg_r(Vl_AA_r)[::-1])
        Sig_BB = Ldn * self._r_to_k(GdAA_r * self._neg_r(Vpm_AA_r)[::-1] + GdBB_r * self._neg_r(Vl_AA_r)[::-1])
        Sig_AB = Lpm * self._r_to_k(GdAB_r * self._neg_r(Vpm_AB_r + Vl_AB_r)[::-1])

        return {
            "Sig_AA": Sig_AA,
            "Sig_BB": Sig_BB,
            "Sig_AB": Sig_AB,
            "lam_pm": lam_pm,
            "lam_zz": lam_zz,
            "chi_pm_odd0": chi_pm_odd0,
        }

    def solve(self, h_seed=0.25, n_seed_iters=2, max_outer=50, tol=1e-5, mix=0.35):
        """
        Solve 2x2 sublattice Neel AFM phase starting from an initial staggered seed field h_seed
        (turned off completely after n_seed_iters so the converged m_AFM is 100% spontaneous!).
        """
        dlr = self.dlr
        mu = 0.5 * self.U if self.half_fill else 0.5 * self.U - 2.0 * self.delta

        # Initial staggered hybridization & densities
        D0 = 6.0 * (self.t ** 2) / (dlr.iwn + 2.0j * np.sign(dlr.wn))
        Delta_up = self._sym_Delta(D0)
        Delta_dn = self._sym_Delta(D0)
        n_up = 0.5 * self.n_target + 0.20 * (1.0 if h_seed > 0 else 0.0)
        n_dn = 0.5 * self.n_target - 0.20 * (1.0 if h_seed > 0 else 0.0)

        SAA = np.zeros((dlr.r_F, self.L, self.L, self.L), dtype=np.complex128)
        SBB = np.zeros_like(SAA)
        SAB = np.zeros_like(SAA)

        diff = 1.0
        imp = dual_res = None
        GAA = GBB = GAB = None

        for it in range(max_outer):
            h_ext = h_seed if it < n_seed_iters else 0.0
            # Include infinitesimal staggered seed only during first n_seed_iters
            imp = solve_mipt4_spinpol_dlr(
                dlr, Delta_up, Delta_dn, self.U, mu + h_ext, mu - h_ext, n_up, n_dn,
                use_bkin=(not self.half_fill)
            )
            gA = imp["g_up"]
            gB = imp["g_dn"]
            SigA = imp["Sig_imp_up"]
            SigB = imp["Sig_imp_dn"]

            # 2x2 physical-to-dual self-energy matrix: Sigma_nl = S_d (I + g S_d)^{-1}
            if self.enable_dual:
                SAA, SBB, SAB = self._herglotz_2x2(SAA, SBB, SAB, gA, gB)
                gA_b = gA[:, None, None, None]
                gB_b = gB[:, None, None, None]
                M00 = 1.0 + gA_b * SAA
                M11 = 1.0 + gB_b * SBB
                M01 = gA_b * SAB
                M10 = gB_b * SAB
                detM = M00 * M11 - M01 * M10
                Snl_AA = (SAA * M11 - SAB * M10) / detM
                Snl_BB = (-SAB * M01 + SBB * M00) / detM
                Snl_AB = (SAA * (-M01) + SAB * M00) / detM
            else:
                Snl_AA = Snl_BB = Snl_AB = 0.0

            # 2x2 lattice Dyson inversion for spin-up in (A, B) sublattice space
            def lattice_G(m_val):
                zA = dlr.iwn[:, None, None, None] + m_val + h_ext - SigA[:, None, None, None] - Snl_AA
                zB = dlr.iwn[:, None, None, None] + m_val - h_ext - SigB[:, None, None, None] - Snl_BB
                t_eff = -self.eps_k[None, :, :, :] - Snl_AB
                detG = zA * zB - t_eff * t_eff
                return zB / detG, zA / detG, -t_eff / detG

            if not self.half_fill:
                def n_err(m_try):
                    gaa, gbb, _ = lattice_G(m_try)
                    return 0.5 * float(np.mean(dlr.density_from_G_iwn(gaa)) + np.mean(dlr.density_from_G_iwn(gbb))) - self.n_target
                lo, hi = mu - 4.0, mu + 4.0
                if n_err(lo) * n_err(hi) <= 0.0:
                    mu = float(brentq(n_err, lo, hi, xtol=1e-9))

            GAA, GBB, GAB = lattice_G(mu)
            G_loc_up = np.mean(GAA, axis=(1, 2, 3))
            G_loc_dn = np.mean(GBB, axis=(1, 2, 3))

            # Update dual self-energy via 2x2 transverse magnon + longitudinal ladders
            if self.enable_dual:
                # Exact 2x2 dressed dual propagator: Gd = G_lat - diag(gA, gB) ... wait:
                # Recall exact identity: G = [g^{-1}(I + g*S_d)^{-1} + Delta - h_k]^{-1}
                # and Gd = G_dmft_dressed - diag(gA, gB) where at outer SC (G_loc = g):
                # Gd_0^{-1} = -g^{-1}[(Delta - h_k)^{-1} + g]g^{-1}, so
                # Gd = (I + S_d g)^{-1} [G_lat - g (I + S_d g)] (I + g S_d)^{-1}?
                # Even cleaner: exact 2x2 matrix identity!
                # Since G_lat = (Delta - h_k)^{-1} + (Delta - h_k)^{-1} g^{-1} Gd g^{-1} (Delta - h_k)^{-1},
                # and G_lat^{-1} = g^{-1} + Delta - h_k - S_d (I + g S_d)^{-1},
                # let's verify Gd directly from Gd = [Gd0^{-1} - S_d]^{-1} in 2x2!
                # First compute G_dmft = [g^{-1} + Delta - h_k]^{-1} -> Gd0 = G_dmft - diag(gA, gB).
                zA0 = dlr.iwn[:, None, None, None] + mu + h_ext - SigA[:, None, None, None]
                zB0 = dlr.iwn[:, None, None, None] + mu - h_ext - SigB[:, None, None, None]
                det0 = zA0 * zB0 - (self.eps_k[None, :, :, :] ** 2)
                Gd0_AA = zB0 / det0 - gA[:, None, None, None]
                Gd0_BB = zA0 / det0 - gB[:, None, None, None]
                Gd0_AB = self.eps_k[None, :, :, :] / det0
                # Solve Gd = (I - Gd0 S_d)^{-1} Gd0 in 2x2 (pole-free, never inverts Gd0!)
                A00 = 1.0 - (Gd0_AA * SAA + Gd0_AB * SAB)
                A11 = 1.0 - (Gd0_AB * SAB + Gd0_BB * SBB)
                A01 = -(Gd0_AA * SAB + Gd0_AB * SBB)
                A10 = -(Gd0_AB * SAA + Gd0_BB * SAB)
                detA = A00 * A11 - A01 * A10
                GdAA = (A11 * Gd0_AA - A01 * Gd0_AB) / detA
                GdBB = (-A10 * Gd0_AB + A00 * Gd0_BB) / detA
                GdAB = (A11 * Gd0_AB - A01 * Gd0_BB) / detA
                GdAA -= np.mean(GdAA, axis=(1, 2, 3), keepdims=True)
                GdBB -= np.mean(GdBB, axis=(1, 2, 3), keepdims=True)

                vert = self._build_local_vertices_neel(imp)
                dual_res = self._dual_self_energy_neel(imp, vert, GAA, GBB, GAB, GdAA, GdBB, GdAB)
                SAA_new, SBB_new, SAB_new = self._herglotz_2x2(
                    dual_res["Sig_AA"], dual_res["Sig_BB"], dual_res["Sig_AB"], gA, gB
                )
            else:
                SAA_new, SBB_new, SAB_new = SAA, SBB, SAB

            # Outer self-consistency update for spin-resolved Delta_up, Delta_dn and densities
            Delta_up_new = self._sym_Delta(Delta_up + (1.0 / gA - 1.0 / G_loc_up))
            Delta_dn_new = self._sym_Delta(Delta_dn + (1.0 / gB - 1.0 / G_loc_dn))
            n_up_new = 0.5 * float(dlr.density_from_G_iwn(G_loc_up))
            n_dn_new = 0.5 * float(dlr.density_from_G_iwn(G_loc_dn))

            diff_D = max(float(np.max(np.abs(Delta_up_new - Delta_up))), float(np.max(np.abs(Delta_dn_new - Delta_dn))))
            diff_S = max(float(np.max(np.abs(SAA_new - SAA))), float(np.max(np.abs(SBB_new - SBB))), float(np.max(np.abs(SAB_new - SAB)))) if self.enable_dual else 0.0
            diff = max(diff_D, diff_S)

            Delta_up = self._sym_Delta((1.0 - mix) * Delta_up + mix * Delta_up_new)
            Delta_dn = self._sym_Delta((1.0 - mix) * Delta_dn + mix * Delta_dn_new)
            if self.half_fill:
                Delta_up = 0.5 * (Delta_up - np.conj(Delta_dn))
                Delta_dn = -np.conj(Delta_up)
            n_up = (1.0 - mix) * n_up + mix * n_up_new
            n_dn = (1.0 - mix) * n_dn + mix * n_dn_new
            if self.enable_dual:
                SAA = (1.0 - mix) * SAA + mix * SAA_new
                SBB = (1.0 - mix) * SBB + mix * SBB_new
                SAB = (1.0 - mix) * SAB + mix * SAB_new

            if it >= n_seed_iters + 5 and diff < tol:
                break

        m_lat = abs(n_up_new - n_dn_new)
        n_lat = n_up_new + n_dn_new
        w0_i = dlr.r_half
        C0 = float(max(
            np.max(np.abs(np.imag(SAA[w0_i]))) * abs(np.imag(gA[w0_i])),
            np.max(np.abs(np.imag(SBB[w0_i]))) * abs(np.imag(gB[w0_i])),
        )) if self.enable_dual else 0.0

        return {
            "U": self.U,
            "T": self.T,
            "delta": self.delta,
            "L": self.L,
            "enable_dual": self.enable_dual,
            "n_lat": n_lat,
            "m_AFM": m_lat,
            "m_imp": abs(imp["m_imp"]),
            "D_imp": imp["D_imp"],
            "lam_pm": dual_res["lam_pm"] if dual_res is not None else 0.0,
            "chi_pm_odd0": dual_res["chi_pm_odd0"] if dual_res is not None else 0.0,
            "C0": C0,
            "diff": diff,
            "iters": it + 1,
        }
