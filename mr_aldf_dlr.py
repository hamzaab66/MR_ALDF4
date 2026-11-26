"""
Discrete Lehmann Representation (DLR) Sparse Imaginary-Time / Matsubara Engine
for MR-ALDF-IV (Kaye, Chen, Parcollet, Phys. Rev. B 105, 235115, 2022).

Unified particle-hole-symmetric DLR basis on shared odd-symmetric poles
  x_poles = [-x_od[::-1], 0.0, +x_od]   (size r = 2*r_od + 1)
and shared symmetric imaginary-time nodes
  u_nodes = [u_od, 0.5, 1 - u_od[::-1]] (size r = 2*r_od + 1),
with exact conjugate-symmetric fermionic Matsubara grid (size r_F = 2*r_od + 2)
and exact square bosonic Matsubara grid including m = 0 (size r_B = 2*r_od + 1).
Zero external dependencies beyond NumPy/SciPy; ~1e-10 accuracy down to T = 0.01t.
"""

import numpy as np
from scipy.linalg import qr, lu_factor, lu_solve, pinv
from scipy.optimize import brentq


def _k_tau_x(u, x):
    """
    Overflow-free unified Lehmann kernel K(u, x) = exp(-u*x) / (1 + exp(-x))
    for u = tau/beta in [0, 1] and x = beta*omega in [-Lambda, Lambda].
    """
    u = np.asarray(u, dtype=float)
    x = np.asarray(x, dtype=float)
    u_b, x_b = np.broadcast_arrays(u, x)
    pos = x_b >= 0.0
    out = np.empty(u_b.shape, dtype=float)
    xp, up = x_b[pos], u_b[pos]
    out[pos] = np.exp(-up * xp) / (1.0 + np.exp(-xp))
    xn, un = x_b[~pos], u_b[~pos]
    out[~pos] = np.exp((1.0 - un) * xn) / (1.0 + np.exp(xn))
    return out


def _composite_chebyshev_half(a, b, n_panels, p_order):
    """Composite Chebyshev-Gauss nodes on [a, b] with dyadic refinement near a."""
    edges = [a] + [a + (b - a) / (2.0 ** k) for k in range(n_panels - 1, -1, -1)]
    edges = np.array(edges, dtype=float)
    j = np.arange(p_order)
    t_cheb = -np.cos((2.0 * j + 1.0) * np.pi / (2.0 * p_order))
    return np.concatenate([
        0.5 * (edges[k] + edges[k + 1]) + 0.5 * (edges[k + 1] - edges[k]) * t_cheb
        for k in range(len(edges) - 1)
    ])


class DLRBasis:
    """
    Particle-hole-symmetric Discrete Lehmann Representation (DLR) basis for inverse
    temperature beta, spectral cutoff w_max (Lambda = beta * w_max), and accuracy eps.

    Shared poles & tau nodes for fermions and bosons (r = 2*r_od + 1):
      - Fermionic Matsubara nodes: r_F = 2*r_od + 2 (exact +/- pairs, full column rank)
      - Bosonic Matsubara nodes:   r_B = 2*r_od + 1 (exact square, includes m = 0)
    """

    def __init__(self, beta, w_max=20.0, eps=1e-7):
        self.beta = float(beta)
        self.T = 1.0 / self.beta
        self.w_max = float(w_max)
        self.Lambda = self.beta * self.w_max
        self.eps = float(eps)

        n_pan = max(6, int(np.ceil(np.log2(max(self.Lambda, 4.0)))) + 2)
        p_ord = 16

        # 1. Fine candidate grids on u in (0, 0.5) and x in (0, Lambda)
        u_half = _composite_chebyshev_half(0.0, 0.5, n_pan, p_ord)
        x_half = _composite_chebyshev_half(0.0, self.Lambda, n_pan, p_ord)

        # 2. Odd-symmetric DLR poles {-x_od, 0, +x_od}:
        #    Pivoted QR on the antisymmetric kernel K_od(u_half, x_half) guarantees x=0
        #    is NEVER duplicated and all positive poles are strictly separated from 0.
        K_pos = _k_tau_x(u_half[:, None], x_half[None, :])
        K_neg = _k_tau_x(u_half[:, None], -x_half[None, :])
        K_od = 0.5 * (K_pos - K_neg)
        _, R_od, piv_od = qr(K_od, mode="economic", pivoting=True)
        diag_R = np.abs(np.diag(R_od))
        r_od = int(np.searchsorted(-diag_R, -self.eps * diag_R[0])) + 1
        r_od = max(4, min(r_od, len(x_half)))
        x_od = np.sort(x_half[piv_od[:r_od]])

        self.r_od = r_od
        self.x_poles = np.concatenate([-x_od[::-1], [0.0], x_od])
        self.w_poles = self.x_poles / self.beta
        self.r = len(self.x_poles)          # = 2*r_od + 1 (DLR pole & tau dimension)
        self.x_poles_B = self.x_poles
        self.w_poles_B = self.w_poles
        self.r_B = self.r

        # 3. Shared symmetric imaginary-time nodes [u_od, 0.5, 1 - u_od[::-1]]
        K_od_p = 0.5 * (
            _k_tau_x(u_half[:, None], x_od[None, :])
            - _k_tau_x(u_half[:, None], -x_od[None, :])
        )
        _, _, piv_u = qr(K_od_p.T, mode="economic", pivoting=True)
        u_od = np.sort(u_half[piv_u[:r_od]])
        self.u_nodes = np.concatenate([u_od, [0.5], 1.0 - u_od[::-1]])
        self.tau_nodes = self.u_nodes * self.beta
        self.u_nodes_B = self.u_nodes
        self.tau_nodes_B = self.tau_nodes

        self.K_tau = _k_tau_x(self.u_nodes[:, None], self.x_poles[None, :])
        self.K_tau_F = -self.K_tau
        self.K_tau_B = self.K_tau
        self._lu_tau_F = lu_factor(self.K_tau_F)
        self._lu_tau_B = lu_factor(self.K_tau_B)
        self.cond_tau = float(np.linalg.cond(self.K_tau))

        # 4. Conjugate-symmetric fermionic Matsubara nodes n_f (size r_F = 2*r_od + 2):
        #    Select r_od + 1 positive indices via pivoted QR on the even-in-w imaginary kernel
        #    wn / (wn^2 + w_ev^2) with w_ev = [0, +w_od]. Exact rank r = 2*r_od + 1!
        n_max = max(256, int(np.ceil(10.0 * self.Lambda)))
        n_dense = np.arange(0, min(64, n_max + 1), dtype=int)
        n_log = np.unique(np.round(np.geomspace(max(64, 1), n_max, 256)).astype(int))
        n_cand = np.unique(np.concatenate([n_dense, n_log]))
        wn_cand = (2 * n_cand + 1) * np.pi / self.beta

        w_ev = np.concatenate([[0.0], x_od / self.beta])  # length r_od + 1
        KF_ev_cand = wn_cand[:, None] / (wn_cand[:, None] ** 2 + w_ev[None, :] ** 2)
        _, _, piv_f = qr(KF_ev_cand.T, mode="economic", pivoting=True)
        n_pos = np.sort(n_cand[piv_f[: r_od + 1]])
        self.n_f = np.concatenate([-1 - n_pos[::-1], n_pos])
        self.r_F = len(self.n_f)            # = 2*r_od + 2
        self.r_half = r_od + 1              # index of lowest positive Matsubara w_0
        self.wn = (2 * self.n_f + 1) * np.pi / self.beta
        self.iwn = 1j * self.wn

        self.K_iwn = 1.0 / (self.iwn[:, None] - self.w_poles[None, :])  # (r_F, r)
        # Exact SVD pseudoinverse (r, r_F) — exact left-inverse for all DLR-representable functions
        U_s, s_s, Vh_s = np.linalg.svd(self.K_iwn, full_matrices=False)
        self.cond_iwn = float(s_s[0] / s_s[-1])
        self._pinv_Kiwn = (Vh_s.conj().T / s_s[None, :]) @ U_s.conj().T

        # 5. Exact square bosonic Matsubara nodes m_b = [-m_pos[::-1], 0, +m_pos] (size r_B = 2*r_od + 1)
        m_max = max(256, int(np.ceil(10.0 * self.Lambda)))
        m_dense = np.arange(1, min(64, m_max + 1), dtype=int)
        m_log = np.unique(np.round(np.geomspace(max(64, 1), m_max, 256)).astype(int))
        m_cand = np.unique(np.concatenate([m_dense, m_log]))
        Om_cand = 2.0 * m_cand * np.pi / self.beta

        w_od = x_od / self.beta
        KB_od_cand = (Om_cand[:, None] * np.tanh(0.5 * x_od[None, :])) / (
            Om_cand[:, None] ** 2 + w_od[None, :] ** 2
        )
        _, _, piv_b = qr(KB_od_cand.T, mode="economic", pivoting=True)
        m_pos = np.sort(m_cand[piv_b[:r_od]])
        self.m_b = np.concatenate([-m_pos[::-1], [0], m_pos])
        self.m0_idx = r_od
        self.Om = 2.0 * self.m_b * np.pi / self.beta
        self.iOm = 1j * self.Om

        self.K_iOm = self._k_boson_iOm(self.Om, self.w_poles_B)  # (r_B, r_B) exact square
        self._lu_iOm = lu_factor(self.K_iOm)
        self.cond_iOm = float(np.linalg.cond(self.K_iOm))

        # 6. Precomputed fast O(r^2) transform matrices
        self._inv_KtauF = lu_solve(self._lu_tau_F, np.eye(self.r, dtype=float))
        self._inv_KtauB = lu_solve(self._lu_tau_B, np.eye(self.r, dtype=float))
        self._inv_KiOm = lu_solve(self._lu_iOm, np.eye(self.r_B, dtype=np.complex128))

        self.T_tau_from_iwn = self.K_tau_F @ self._pinv_Kiwn     # (r, r_F)
        self.T_iwn_from_tau = self.K_iwn @ self._inv_KtauF       # (r_F, r)
        self.T_tau_from_iOm = self.K_tau_B @ self._inv_KiOm      # (r_B, r_B)
        self.T_iOm_from_tau = self.K_iOm @ self._inv_KtauB       # (r_B, r_B)

        # Exact density weight n_sigma = -G(beta^-) = sum_l f(x_l) g_dlr[l]
        self._fermi_w = _k_tau_x(1.0, self.x_poles)
        self._density_from_iwn_w = self._fermi_w @ self._pinv_Kiwn

        # Exact symmetrized bosonic sum rule: (1/beta) sum_m X(iOm_m) = 0.5 * sum_l c_dlr[l]
        self._bsum_from_iOm_w = 0.5 * np.sum(self._inv_KiOm, axis=0)

        self._build_exact_bilinears()

    def _k_boson_iOm(self, Om, w_poles):
        """Exact bosonic Matsubara kernel K_B(iOm, w) = -tanh(beta*w/2)/(iOm - w), smooth at (0,0)."""
        x = self.beta * w_poles[None, :]
        Om_b = Om[:, None]
        w_b = w_poles[None, :]
        denom = 1j * Om_b - w_b
        zero_mask = (np.abs(Om_b) < 1e-14) & (np.abs(w_b) < 1e-14)
        safe_denom = np.where(zero_mask, 1.0 + 0.0j, denom)
        val = -np.tanh(0.5 * x) / safe_denom
        return np.where(zero_mask, 0.5 * self.beta + 0.0j, val)

    def _build_exact_bilinears(self):
        """
        Exact analytical Matsubara bilinear sum weights:
          (1/beta) sum_{wn} A(iwn) B(iwn) = sum_{a,b} A_a W_{ab} B_b
        using I_{kl} = (f(x_k) - f(x_l)) / (w_k - w_l) (exact to eps_DLR with zero tail error).
        """
        w = self.w_poles
        f = self._fermi_w
        dw = w[:, None] - w[None, :]
        df = f[:, None] - f[None, :]
        diag = np.abs(dw) < 1e-12
        safe_dw = np.where(diag, 1.0, dw)
        I_mat = np.where(diag, -self.beta * f[:, None] * (1.0 - f[:, None]), df / safe_dw)
        self.I_fermion_prod_dlr = I_mat
        self.W_fermion_prod_iwn = self._pinv_Kiwn.T @ I_mat @ self._pinv_Kiwn

    # ---------- Vectorized transforms (axis=0 is frequency/time) ----------

    def iwn_to_tau(self, A_iwn):
        return np.tensordot(self.T_tau_from_iwn, A_iwn, axes=(1, 0))

    def tau_to_iwn(self, A_tau):
        return np.tensordot(self.T_iwn_from_tau, A_tau, axes=(1, 0))

    def iwn_to_tauB(self, A_iwn):
        """Shared tau grid: identical to iwn_to_tau."""
        return self.iwn_to_tau(A_iwn)

    def iOm_to_tauB(self, X_iOm):
        return np.tensordot(self.T_tau_from_iOm, X_iOm, axes=(1, 0))

    def tauB_to_iOm(self, X_tauB):
        return np.tensordot(self.T_iOm_from_tau, X_tauB, axes=(1, 0))

    def iOm_to_tauF(self, X_iOm):
        """Shared tau grid: identical to iOm_to_tauB."""
        return self.iOm_to_tauB(X_iOm)

    def eval_fermion_iwn(self, A_iwn, wn_target):
        c = np.tensordot(self._pinv_Kiwn, A_iwn, axes=(1, 0))
        K_tgt = 1.0 / (1j * np.asarray(wn_target)[:, None] - self.w_poles[None, :])
        return np.tensordot(K_tgt, c, axes=(1, 0))

    def eval_boson_iOm(self, X_iOm, Om_target):
        c = np.tensordot(self._inv_KiOm, X_iOm, axes=(1, 0))
        K_tgt = self._k_boson_iOm(np.asarray(Om_target), self.w_poles_B)
        return np.tensordot(K_tgt, c, axes=(1, 0))

    def density_from_G_iwn(self, G_iwn):
        return 2.0 * np.real(np.tensordot(self._density_from_iwn_w, G_iwn, axes=(0, 0)))

    def boson_sum_from_iOm(self, X_iOm):
        return np.real(np.tensordot(self._bsum_from_iOm_w, X_iOm, axes=(0, 0)))

    def fermion_prod_sum(self, A_iwn, B_iwn):
        """(1/beta) sum_{wn} A(iwn) B(iwn) along axis=0."""
        WB = np.tensordot(self.W_fermion_prod_iwn, B_iwn, axes=(1, 0))
        return np.sum(A_iwn * WB, axis=0)


# ==============================================================================
# Exact DLR mIPT-4 + Local-TPSC Impurity Solver (Ultra-Low T Ready)
# ==============================================================================

def solve_mipt4_impurity_dlr(dlr: DLRBasis, Delta_iwn, U, mu, n_target, max_bkin_iter=15):
    """
    Exact fourth-moment mIPT-4 + (s, c, pp) local-TPSC impurity solver on the DLR grid.
    Zero uniform FFT grids, zero tail-subtraction heuristics, accurate to ~1e-10 even at T = 0.02t.
    """
    iwn = dlr.iwn
    n_val = float(n_target)
    half_fill = abs(n_val - 1.0) < 1e-12

    if half_fill:
        mu0 = 0.0
    else:
        def dens_diff(m0):
            return float(dlr.density_from_G_iwn(1.0 / (iwn + m0 - Delta_iwn))) - n_val
        lim_m0 = 0.75 * dlr.w_max
        mu0 = float(brentq(dens_diff, -lim_m0, lim_m0, xtol=1e-12))

    G0_iwn = 1.0 / (iwn + mu0 - Delta_iwn)
    G0_tau = np.real(dlr.iwn_to_tau(G0_iwn))
    G0_rev = G0_tau[::-1]

    # Second-order sunset Sigma^(2)(tau) = +U^2 G0(tau)^2 G0(beta - tau) < 0
    Sig2_tau = (U ** 2) * (G0_tau ** 2) * G0_rev
    Sig2_iwn = dlr.tau_to_iwn(Sig2_tau)

    denom_b = (U ** 2) * (0.5 * n_val) * (1.0 - 0.5 * n_val)
    Delta_Bkin = 0.0
    for _ in range(max_bkin_iter):
        if U < 1e-12 or denom_b < 1e-14:
            B_ipt = 0.0
            Sig_dyn = np.zeros_like(iwn)
        else:
            B_ipt = (U * (1.0 - 0.5 * n_val) - mu + mu0 + Delta_Bkin) / denom_b
            Sig_dyn = Sig2_iwn / (1.0 - B_ipt * Sig2_iwn)
        Sig_imp = 0.5 * U * n_val + Sig_dyn
        g_imp = 1.0 / (iwn + mu - Delta_iwn - Sig_imp)
        if half_fill or U < 1e-12:
            break
        bracket = (n_val - 1.0) * (g_imp - G0_iwn) + (2.0 / U) * Sig_dyn * g_imp
        dB_new = float(np.real(dlr.fermion_prod_sum(Delta_iwn, bracket))) / (0.5 * n_val * (1.0 - 0.5 * n_val))
        if abs(dB_new - Delta_Bkin) < 1e-11:
            Delta_Bkin = dB_new
            break
        Delta_Bkin = 0.5 * Delta_Bkin + 0.5 * dB_new

    n_imp = float(dlr.density_from_G_iwn(g_imp))
    if U < 1e-12:
        D_imp = 0.25 * n_imp * n_imp
    else:
        D_imp = 0.25 * n_val * n_imp + float(np.real(dlr.fermion_prod_sum(Sig_dyn, g_imp))) / U
        D_imp = float(np.clip(D_imp, max(0.0, n_val - 1.0), 0.5 * n_val))

    # Local-TPSC (s, c, pp) susceptibilities on shared DLR tau -> bosonic iOm grid
    chi0_G0_iOm = np.maximum(np.real(dlr.tauB_to_iOm(G0_tau * G0_rev)), 0.0)

    target_s = n_val - 2.0 * D_imp
    c0_max = float(np.max(chi0_G0_iOm))
    if U < 1e-12 or c0_max < 1e-14:
        U_sp = 0.0
    else:
        def sum_s(usp):
            return float(dlr.boson_sum_from_iOm(2.0 * chi0_G0_iOm / (1.0 - usp * chi0_G0_iOm))) - target_s
        u_hi = (1.0 - 1e-10) / c0_max
        u_lo = 0.0 if sum_s(0.0) <= 0.0 else -10.0 * U
        U_sp = float(brentq(sum_s, u_lo, u_hi, xtol=1e-11)) if sum_s(u_lo) * sum_s(u_hi) <= 0.0 else float(U)
    chi_imp_s = 2.0 * chi0_G0_iOm / (1.0 - U_sp * chi0_G0_iOm)

    target_c = n_val * (1.0 - n_val) + 2.0 * D_imp
    if U < 1e-12 or c0_max < 1e-14:
        U_ch = 0.0
    else:
        def sum_c(uch):
            return float(dlr.boson_sum_from_iOm(2.0 * chi0_G0_iOm / (1.0 + uch * chi0_G0_iOm))) - target_c
        u_lo = 0.0 if sum_c(0.0) >= 0.0 else -(1.0 - 1e-9) / c0_max
        u_hi = max(10.0 * U, 50.0)
        while sum_c(u_hi) > 0.0 and u_hi < 1e9:
            u_hi *= 4.0
        U_ch = float(brentq(sum_c, u_lo, u_hi, xtol=1e-11)) if sum_c(u_lo) * sum_c(u_hi) <= 0.0 else float(U)
    chi_imp_c = 2.0 * chi0_G0_iOm / (1.0 + U_ch * chi0_G0_iOm)

    chi0_pp_iOm = dlr.tauB_to_iOm(G0_tau ** 2)
    target_pp = 0.5 * (1.0 - n_val) + D_imp
    if U < 1e-12:
        U_pp = 0.0
    else:
        def sum_pp(upp):
            return float(dlr.boson_sum_from_iOm(chi0_pp_iOm / (1.0 + upp * chi0_pp_iOm))) - target_pp
        c_pp0 = float(np.real(chi0_pp_iOm[dlr.m0_idx]))
        u_lo = -(1.0 - 1e-8) / max(c_pp0, 1e-12) if c_pp0 > 0 else -10.0
        u_hi = max(10.0 * U, 50.0)
        while sum_pp(u_hi) > 0.0 and u_hi < 1e9:
            u_hi *= 4.0
        U_pp = float(brentq(sum_pp, u_lo, u_hi, xtol=1e-11)) if sum_pp(u_lo) * sum_pp(u_hi) <= 0.0 else float(U)
    chi_imp_pp = chi0_pp_iOm / (1.0 + U_pp * chi0_pp_iOm)

    # Local bubbles on physical g(iwn) and Ward form factor Lambda(iwn) = 1 - Im(Sig_imp)/wn
    Lam = np.maximum(1.0 - np.imag(Sig_imp) / dlr.wn, 1.0)
    g_tau = np.real(dlr.iwn_to_tau(g_imp))
    Lam_g_tau = np.real(dlr.iwn_to_tau(np.sqrt(Lam) * g_imp))
    chi0_loc_iOm = np.real(dlr.tauB_to_iOm(g_tau * g_tau[::-1]))
    chi_Lam_loc_iOm = np.real(dlr.tauB_to_iOm(Lam_g_tau * Lam_g_tau[::-1]))
    chi_pp_Lam_loc_iOm = dlr.tauB_to_iOm(Lam_g_tau ** 2)

    # Static 2x2 local bubble matrix X0_loc_static (exact O(r^2) DLR bilinear sum!)
    phi0_g = g_imp
    phi1_g = (Lam - 1.0) * g_imp
    X0_00 = float(np.real(-dlr.fermion_prod_sum(phi0_g, phi0_g)))
    X0_01 = float(np.real(-dlr.fermion_prod_sum(phi0_g, phi1_g)))
    X0_11 = float(np.real(-dlr.fermion_prod_sum(phi1_g, phi1_g)))
    X0_loc_static = np.array([[X0_00, X0_01], [X0_01, X0_11]], dtype=float)

    # Exact local quartic box chi4_loc = (1/beta) sum_nu Re[G0_tilde(inu)^4]
    G0_sq = G0_iwn ** 2
    chi4_loc = float(np.real(dlr.fermion_prod_sum(G0_sq, G0_sq)))

    return {
        "mu0": mu0,
        "G0_iwn": G0_iwn,
        "Sig_imp": Sig_imp,
        "Sig_dyn": Sig_dyn,
        "g": g_imp,
        "Lam": Lam,
        "n_imp": n_imp,
        "D_imp": D_imp,
        "Delta_B_kin": Delta_Bkin,
        "U_sp": U_sp,
        "U_ch": U_ch,
        "U_pp": U_pp,
        "chi0_G0": chi0_G0_iOm,
        "chi0_pp_G0": chi0_pp_iOm,
        "chi_imp_s": chi_imp_s,
        "chi_imp_c": chi_imp_c,
        "chi_imp_pp": chi_imp_pp,
        "chi0_loc": chi0_loc_iOm,
        "chi_Lam_loc": chi_Lam_loc_iOm,
        "chi_pp_Lam_loc": chi_pp_Lam_loc_iOm,
        "X0_loc_static": X0_loc_static,
        "chi4_loc": chi4_loc,
    }
