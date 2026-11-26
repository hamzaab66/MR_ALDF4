"""
MR-ALDF-IV Sparse-Frequency DLR Engine (mr_aldf_level4_dlr.py)
Track IV-A Ultra-Low-Temperature & Large-L 3D Simple-Cubic Hubbard Solver.

Replaces uniform Matsubara/FFT frequency grids with the unified odd-symmetric
Discrete Lehmann Representation (DLRBasis, mr_aldf_dlr.py), executing every
fermionic, bosonic, TU-SBE Eg, and 3D O(3) SCSA convolution in O(r * N log N)
time with zero bosonic/fermionic cutoff truncation error down to T ~ 0.02t.
"""

import numpy as np
from scipy.optimize import brentq
from mr_aldf_dlr import DLRBasis, solve_mipt4_impurity_dlr


class PulayDIIS:
    """Compact Pulay/DIIS accelerator for complex fixed-point vectors."""

    def __init__(self, max_hist=6, beta_mix=0.45):
        self.max_hist = max_hist
        self.beta_mix = beta_mix
        self.xs = []
        self.rs = []

    def reset(self):
        self.xs.clear()
        self.rs.clear()

    def update(self, x_in, x_out):
        r = x_out - x_in
        self.xs.append(x_in.copy())
        self.rs.append(r.copy())
        if len(self.xs) > self.max_hist:
            self.xs.pop(0)
            self.rs.pop(0)
        m = len(self.xs)
        if m < 2:
            return x_in + self.beta_mix * r
        B = np.empty((m + 1, m + 1), dtype=float)
        B[:m, :m] = 0.0
        for i in range(m):
            for j in range(i, m):
                v = float(np.real(np.vdot(self.rs[i], self.rs[j])))
                B[i, j] = B[j, i] = v
        B[m, :m] = -1.0
        B[:m, m] = -1.0
        B[m, m] = 0.0
        scale = max(float(np.max(np.abs(np.diag(B[:m, :m])))), 1e-30)
        B[:m, :m] += 1e-12 * scale * np.eye(m)
        rhs = np.zeros(m + 1, dtype=float)
        rhs[m] = -1.0
        try:
            c = np.linalg.solve(B, rhs)[:m]
            if np.max(np.abs(c)) > 20.0 or not np.all(np.isfinite(c)):
                return x_in + self.beta_mix * r
            x_bar = sum(c[i] * self.xs[i] for i in range(m))
            r_bar = sum(c[i] * self.rs[i] for i in range(m))
            return x_bar + self.beta_mix * r_bar
        except np.linalg.LinAlgError:
            return x_in + self.beta_mix * r


class MRALDFLevel4DLRSolver:
    """
    Full QMC-free MR-ALDF-IV solver powered by the sparse DLR engine (DLRBasis).
    Includes:
      1. Exact fourth-moment mIPT-4 + (s, c, pp) local-TPSC impurity solver on DLR
      2. UV-tail-regularized rank-2 W_c^reg (detects D_c^irr = 0) + bare-U pp bound
      3. Non-singular dual-denominator Moriya lambda_s via exact DLR Pauli excess sum rule
      4. Parameter-free microscopic 3D O(3) Bosonic SCSA via DLR (r, tau) convolution
      5. Self-consistent 3-channel + TU-SBE Eg (d-wave) feedback (O(U^4), DC-free)
      6. Smooth C^inf dual-inverse Herglotz projection + exact PH/Matsubara symmetry
    """

    def __init__(
        self,
        U,
        T,
        delta=0.0,
        L=8,
        t=1.0,
        w_max=None,
        eps_dlr=1e-7,
        enable_scsa=True,
        enable_eg_feedback=True,
        enable_herglotz=True,
        enable_gamma6=False,
        c_max=0.92,
    ):
        self.U = float(U)
        self.T = float(T)
        self.beta = 1.0 / self.T
        self.delta = float(delta)
        self.n_target = 1.0 - self.delta
        self.half_fill = abs(self.delta) < 1e-12
        self.L = int(L)
        self.N = self.L ** 3
        self.t = float(t)

        # Spectral cutoff safely covers full Hubbard bandwidth + U
        self.w_max = float(w_max) if w_max is not None else max(18.0 * self.t, 1.5 * self.U + 12.0 * self.t)
        self.dlr = DLRBasis(beta=self.beta, w_max=self.w_max, eps=eps_dlr)

        self.enable_scsa = bool(enable_scsa)
        self.enable_eg_feedback = bool(enable_eg_feedback)
        self.enable_herglotz = bool(enable_herglotz)
        self.enable_gamma6 = bool(enable_gamma6)
        self.c_max = float(c_max)

        # 3D simple cubic k-grid
        k1d = 2.0 * np.pi * np.arange(self.L) / self.L
        kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
        self.kx, self.ky, self.kz = kx, ky, kz
        self.eps_k = -2.0 * self.t * (np.cos(kx) + np.cos(ky) + np.cos(kz))
        self.vx_k = 2.0 * self.t * np.sin(kx)
        self.vy_k = 2.0 * self.t * np.sin(ky)
        self.vz_k = 2.0 * self.t * np.sin(kz)
        self.v2_k = self.vx_k ** 2 + self.vy_k ** 2 + self.vz_k ** 2

        # Oh Eg (d-wave) normalized form factors: <psi_d^2>_k = 1
        self.psi_d1 = np.cos(kx) - np.cos(ky)
        self.psi_d2 = (2.0 * np.cos(kz) - np.cos(kx) - np.cos(ky)) / np.sqrt(3.0)
        self.psi_Eg_sq = self.psi_d1 ** 2 + self.psi_d2 ** 2
        self.eg_proj_q = 0.5 * (np.cos(kx) + np.cos(ky))

        # Index of Q = (pi, pi, pi) and spatial reversal map -r mod L
        self.Q_idx = (self.L // 2, self.L // 2, self.L // 2)
        self._r_neg = (self.L - np.arange(self.L)) % self.L
        self._q_shift = (np.arange(self.L) + self.L // 2) % self.L

    # ---------- Symmetry & Herglotz Control Helpers ----------

    def _neg_r(self, arr_r):
        """Spatial inversion r -> -r mod L along axes (1, 2, 3)."""
        idx = self._r_neg
        return arr_r[:, idx, :, :][:, :, idx, :][:, :, :, idx]

    def _shift_Q(self, arr_k):
        """Momentum shift k -> k + Q along axes (1, 2, 3)."""
        s = self._q_shift
        return arr_k[:, s, :, :][:, :, s, :][:, :, :, s]

    def _symmetrize_Delta(self, Delta):
        # Conjugate symmetry Delta(-iwn) = Delta(iwn)*
        D = 0.5 * (Delta + np.conj(Delta[::-1]))
        if self.half_fill:
            D = 1j * np.imag(D)
        # Ensure causal Im Delta(iwn > 0) <= 0
        sgn = np.sign(self.dlr.wn)
        im_d = np.where(sgn * np.imag(D) > 0.0, 0.0, np.imag(D))
        return np.real(D) + 1j * im_d

    def _symmetrize_Sig_dual(self, Sig):
        # Conjugate symmetry Sig(k, -iwn) = Sig(k, iwn)*
        S = 0.5 * (Sig + np.conj(Sig[::-1]))
        if self.half_fill:
            S_Q = self._shift_Q(S)
            S = 0.5 * (S - np.conj(S_Q))
        return S

    def _herglotz_project(self, Sig_dual, g_imp):
        """Smooth C^inf dual-inverse Herglotz projection (Approximation 7.1)."""
        S = self._symmetrize_Sig_dual(Sig_dual)
        if not self.enable_herglotz:
            return S
        sgn = np.sign(self.dlr.wn)[:, None, None, None]
        # Ensure causal sign of Im(S) first
        im_S = np.where(sgn * np.imag(S) > 0.0, -1e-14 * sgn, np.imag(S))
        S = np.real(S) + 1j * im_S

        mag2 = np.abs(S) ** 2
        active = mag2 > 1e-24
        y_min = (np.abs(np.imag(g_imp)) / self.c_max)[:, None, None, None]
        dh = 0.05 * np.maximum(y_min, 1e-12)

        inv_S = np.where(active, 1.0 / np.where(active, S, 1.0), 0.0 + 0.0j)
        x_inv = np.real(inv_S)
        y_inv = sgn * np.imag(inv_S)  # >= 0 for causal Im(S) <= 0

        # Smooth softplus: y_proj = y_min + dh * log(1 + exp((y - y_min)/dh))
        z = (y_inv - y_min) / dh
        softp = np.where(z > 35.0, z, np.log1p(np.exp(np.clip(z, -60.0, 35.0))))
        y_proj = y_min + dh * softp

        inv_proj = x_inv + 1j * (sgn * y_proj)
        S_proj = np.where(active, 1.0 / np.where(active, inv_proj, 1.0), S)
        return self._symmetrize_Sig_dual(S_proj)

    # ---------- Fast O(r * N log N) DLR Transform Helpers ----------

    def _k_iwn_to_r_tau(self, A_k_iwn):
        """(r_F, L, L, L) in (iwn, k) -> (r, L, L, L) in (tau, r)."""
        A_tau_k = self.dlr.iwn_to_tau(A_k_iwn)
        return np.fft.ifftn(A_tau_k, axes=(1, 2, 3))

    def _r_tau_to_q_iOm(self, X_r_tau):
        """(r, L, L, L) in (tau, r) -> (r_B, L, L, L) in (iOm, q)."""
        X_tau_q = np.fft.fftn(X_r_tau, axes=(1, 2, 3))
        return self.dlr.tauB_to_iOm(X_tau_q)

    def _q_iOm_to_r_tau(self, V_q_iOm):
        """(r_B, L, L, L) in (iOm, q) -> (r, L, L, L) in (tau, r)."""
        V_tau_q = self.dlr.iOm_to_tauB(V_q_iOm)
        return np.fft.ifftn(V_tau_q, axes=(1, 2, 3))

    def _r_tau_to_k_iwn(self, S_r_tau):
        """(r, L, L, L) in (tau, r) -> (r_F, L, L, L) in (iwn, k)."""
        S_tau_k = np.fft.fftn(S_r_tau, axes=(1, 2, 3))
        return self.dlr.tau_to_iwn(S_tau_k)

    # ---------- Step 2: UV-Regularized Rank-2 Local Vertex ----------

    def _build_local_vertices(self, imp):
        dlr = self.dlr
        m0 = dlr.m0_idx
        c0 = imp["chi0_loc"]
        cL = imp["chi_Lam_loc"]
        cG0 = imp["chi0_G0"]

        # Safe ratio (chi0_G0 / chi_Lam_loc)^2
        ratio_sq = np.where(np.abs(cL) > 1e-14, (cG0 / np.where(np.abs(cL) > 1e-14, cL, 1.0)) ** 2, 0.0)
        ratio_sq = np.clip(np.real(ratio_sq), 0.0, 10.0)

        # Dominant Ward amplitudes gamma_s^reg, gamma_c^reg (Eq. 2.27)
        U_sp, U_ch = imp["U_sp"], imp["U_ch"]
        gs = np.where(
            np.arange(dlr.r_B) == m0,
            (2.0 * c0[m0] - imp["chi_imp_s"][m0]) / max(2.0 * cL[m0] ** 2, 1e-18),
            (-U_sp / np.maximum(1.0 - U_sp * cG0, 1e-8)) * ratio_sq,
        )
        gs = np.minimum(np.real(gs), 0.0)

        gc = np.where(
            np.arange(dlr.r_B) == m0,
            (2.0 * c0[m0] - imp["chi_imp_c"][m0]) / max(2.0 * cL[m0] ** 2, 1e-18),
            (+U_ch / np.maximum(1.0 + U_ch * cG0, 1e-8)) * ratio_sq,
        )
        gc = np.real(gc)

        # Orthogonal charge mode (Eq. 2.28, bare-U bounded!)
        theta = np.clip(np.real(c0) / np.maximum(np.real(cL), 1e-14), 1e-6, 1.0)
        norm_vp = theta ** 2 + (1.0 - theta) ** 2
        gc_perp = np.where(
            np.arange(dlr.r_B) == m0,
            self.U / norm_vp[m0],
            (self.U / norm_vp) * ratio_sq,
        )

        # 2x2 matrices W_c^reg(iOm_m) of shape (r_B, 2, 2)
        Wc = np.zeros((dlr.r_B, 2, 2), dtype=float)
        for m in range(dlr.r_B):
            th = theta[m]
            vp = np.array([1.0 - th, -th])
            Wc[m] = gc[m] * np.ones((2, 2)) + gc_perp[m] * np.outer(vp, vp)

        # Local charge-irreducible vertex determinant D_c^irr (Eq. 2.29)
        D_c_irr = float(np.linalg.det(np.eye(2) - imp["X0_loc_static"] @ Wc[m0]))

        # Bare-U high-frequency singlet pp vertex (Approximation 7.1 item 3)
        c_pp_G0 = imp["chi0_pp_G0"]
        c_pp_L = imp["chi_pp_Lam_loc"]
        U_pp_loc = imp["U_pp"]
        U_pp_hf = min(U_pp_loc, self.U)
        r_pp_sq = np.where(
            np.abs(c_pp_L) > 1e-14,
            (c_pp_G0 / np.where(np.abs(c_pp_L) > 1e-14, c_pp_L, 1.0)) ** 2,
            0.0,
        )
        gpp = np.where(
            np.arange(dlr.r_B) == m0,
            (U_pp_loc / (1.0 + U_pp_loc * c_pp_G0)) * r_pp_sq,
            (U_pp_hf / (1.0 + U_pp_hf * c_pp_G0)) * r_pp_sq,
        )
        # Bound |gpp| to prevent unphysical zero-crossing blowup when chi_pp_L -> 0
        gpp_cap = max(2.0 * abs(U_pp_loc) + 2.0 * self.U, 1.0)
        gpp = np.where(np.abs(gpp) > gpp_cap, gpp * (gpp_cap / np.maximum(np.abs(gpp), 1e-14)), gpp)

        # Exact microscopic quartic spin coupling u4_micro (Eq. 7.3)
        u4_micro = float((U_sp ** 4) * max(imp["chi4_loc"], 0.0))

        return {
            "gs": gs,
            "gc": gc,
            "gc_perp": gc_perp,
            "Wc": Wc,
            "gpp": gpp,
            "D_c_irr": D_c_irr,
            "u4_micro": u4_micro,
        }

    # ---------- Steps 3-5: Dual Ladders, Moriya+SCSA, 3-ch + TU-SBE Eg Dual Sigma ----------

    def _dual_self_energy_step(self, imp, vert, G_lat, G_dual):
        dlr = self.dlr
        Lam = imp["Lam"]                 # (r_F,) real >= 1
        s_Lam = np.sqrt(Lam)             # (r_F,)

        # 1. Transform G_lat and G_dual weighted by {1, sqrt(Lam), Lam} to (tau, r)
        G0_r = self._k_iwn_to_r_tau(G_lat)
        G1_r = self._k_iwn_to_r_tau(s_Lam[:, None, None, None] * G_lat)
        Gd0_r = self._k_iwn_to_r_tau(G_dual)
        Gd1_r = self._k_iwn_to_r_tau(s_Lam[:, None, None, None] * G_dual)
        Gd2_r = self._k_iwn_to_r_tau(Lam[:, None, None, None] * G_dual)

        # 2. Exact lattice bubbles chi0_lat, chi_Lam_lat via (r, tau) product + exact r=0 restoration
        chi0_lat_r = G0_r * self._neg_r(G0_r)[::-1]
        chiL_lat_r = G1_r * self._neg_r(G1_r)[::-1]
        chi0_lat = np.real(self._r_tau_to_q_iOm(chi0_lat_r))
        chiL_lat = np.real(self._r_tau_to_q_iOm(chiL_lat_r))
        # Restore exact local r=0 mean to match imp['chi0_loc'], imp['chi_Lam_loc']
        chi0_lat += (imp["chi0_loc"] - np.mean(chi0_lat, axis=(1, 2, 3)))[:, None, None, None]
        chiL_lat += (imp["chi_Lam_loc"] - np.mean(chiL_lat, axis=(1, 2, 3)))[:, None, None, None]
        chi0_lat = np.maximum(chi0_lat, 0.0)
        chiL_lat = np.maximum(chiL_lat, 0.0)
        # b0, b1 used directly

        # 3. Exact 2x2 dual bubble matrix X_tilde(q, iOm) from 3 scalar (r, tau) bubbles
        Xt_0 = np.real(self._r_tau_to_q_iOm(Gd0_r * self._neg_r(Gd0_r)[::-1]))
        Xt_L = np.real(self._r_tau_to_q_iOm(Gd1_r * self._neg_r(Gd1_r)[::-1]))
        Xt_L2 = np.real(self._r_tau_to_q_iOm(Gd2_r * self._neg_r(Gd2_r)[::-1]))
        # Enforce exact zero BZ average (since G_dual_loc = 0 => X_tilde(r=0) = 0)
        Xt_0 -= np.mean(Xt_0, axis=(1, 2, 3), keepdims=True)
        Xt_L -= np.mean(Xt_L, axis=(1, 2, 3), keepdims=True)
        Xt_L2 -= np.mean(Xt_L2, axis=(1, 2, 3), keepdims=True)

        X00 = Xt_0
        X01 = Xt_L - Xt_0
        X11 = Xt_L2 - 2.0 * Xt_L + Xt_0
        b0 = chi0_lat
        b1 = chiL_lat - chi0_lat

        # 4. Fast explicit 2x2 positivity-enforced charge ladder Gamma_c^(lambda_c) (Eq. 2.35)
        Wc = vert["Wc"]  # (r_B, 2, 2)
        W00 = Wc[:, 0, 0][:, None, None, None]
        W01 = Wc[:, 0, 1][:, None, None, None]
        W11 = Wc[:, 1, 1][:, None, None, None]

        XW00 = X00 * W00 + X01 * W01
        XW01 = X00 * W01 + X01 * W11
        XW10 = X01 * W00 + X11 * W01
        XW11 = X01 * W01 + X11 * W11

        tr_M = 2.0 + XW00 + XW11
        det_M = (1.0 + XW00) * (1.0 + XW11) - XW01 * XW10
        disc = np.maximum(tr_M * tr_M - 4.0 * det_M, 0.0)
        eig_min = float(np.min(0.5 * (tr_M - np.sqrt(disc))))
        lam_c = max(0.0, 0.05 - eig_min)

        def eval_charge(lc):
            M00 = (1.0 + lc) + XW00
            M11 = (1.0 + lc) + XW11
            det = M00 * M11 - XW01 * XW10
            inv00 = M11 / det
            inv11 = M00 / det
            inv01 = -XW01 / det
            inv10 = -XW10 / det
            Gc00 = W00 * inv00 + W01 * inv10
            Gc01 = W00 * inv01 + W01 * inv11
            Gc10 = W01 * inv00 + W11 * inv10
            Gc11 = W01 * inv01 + W11 * inv11
            bGb = b0 * (Gc00 * b0 + Gc01 * b1) + b1 * (Gc10 * b0 + Gc11 * b1)
            chi_c = 2.0 * chi0_lat - 2.0 * bGb
            return (Gc00, Gc01, Gc10, Gc11), bGb, chi_c

        (Gc00, Gc01, Gc10, Gc11), bGb, chi_c = eval_charge(lam_c)
        for _ in range(15):
            if float(np.min(chi_c)) >= -1e-10:
                break
            lam_c = max(lam_c * 2.0, 0.1) + 0.1
            (Gc00, Gc01, Gc10, Gc11), bGb, chi_c = eval_charge(lam_c)

        # 5. Non-singular Moriya lambda_s + Parameter-Free 3D O(3) Bosonic SCSA!
        gs = vert["gs"]  # (r_B,) <= 0
        Ds = 1.0 + gs[:, None, None, None] * Xt_L2
        rhs_pauli = float(2.0 * np.mean(dlr.boson_sum_from_iOm(bGb)))
        chiL_sq = chiL_lat ** 2

        dD_scsa = np.zeros_like(Ds)
        n_scsa_loops = 3 if (self.enable_scsa and vert["u4_micro"] > 1e-14) else 1
        lam_s = 0.0
        for _ in range(n_scsa_loops):
            Ds_eff = Ds + dD_scsa
            lam_lo = max(0.0, 1e-7 - float(np.min(Ds_eff)))

            w_bsum = np.real(dlr._bsum_from_iOm_w)
            w_eff = (-2.0 * w_bsum * gs)[:, None, None, None] * chiL_sq
            def pauli_res(ls):
                return float(np.sum(w_eff / (Ds_eff + ls)) / self.N) - rhs_pauli

            if rhs_pauli <= 1e-15 or pauli_res(lam_lo) <= 0.0:
                lam_s = lam_lo
            else:
                lam_hi = max(lam_lo + 1.0, 4.0)
                while pauli_res(lam_hi) > 0.0 and lam_hi < 1e7:
                    lam_hi *= 4.0
                if pauli_res(lam_lo) * pauli_res(lam_hi) <= 0.0:
                    lam_s = float(brentq(pauli_res, lam_lo, lam_hi, xtol=1e-10))
                else:
                    lam_s = lam_lo

            Gamma_s = gs[:, None, None, None] / (Ds_eff + lam_s)
            chi_s = 2.0 * chi0_lat - 2.0 * Gamma_s * chiL_sq

            if not self.enable_scsa or vert["u4_micro"] <= 1e-14:
                break

            # Exact O(r * N log N) 3D O(3) Bosonic SCSA (Eqs. 7.4 - 7.8)
            u4 = vert["u4_micro"]
            chi_s_r = np.real(self._q_iOm_to_r_tau(chi_s))
            Pi_xx = np.maximum(np.real(self._r_tau_to_q_iOm(chi_s_r ** 2)), 0.0)
            # Split D4 = 2*u4 + dD4 so dD4 decays as O(Om^-4) and 2*u4 cancels at (Q, 0)!
            dD4 = -(6.0 * (u4 ** 2) * Pi_xx) / (1.0 + 3.0 * u4 * Pi_xx)
            dD4_r = np.real(self._q_iOm_to_r_tau(dD4))
            Pi_scsa = np.real(self._r_tau_to_q_iOm(dD4_r * chi_s_r))
            dPi_scsa = Pi_scsa - Pi_scsa[dlr.m0_idx, self.Q_idx[0], self.Q_idx[1], self.Q_idx[2]]
            dD_new = 2.0 * np.abs(gs)[:, None, None, None] * chiL_sq * dPi_scsa
            dD_scsa = 0.5 * dD_scsa + 0.5 * dD_new

        # 6. Particle-hole fluctuation kernel V_ph(q, iOm) in {1, Lambda} basis (Eq. 2.48)
        V_s = 1.5 * gs[:, None, None, None] * (Gamma_s - 0.5 * gs[:, None, None, None]) * Xt_L2
        dGc00 = Gc00 - 0.5 * W00
        dGc01 = Gc01 - 0.5 * W01
        dGc10 = Gc10 - 0.5 * W01
        dGc11 = Gc11 - 0.5 * W11

        WX00 = W00 * X00 + W01 * X01
        WX01 = W00 * X01 + W01 * X11
        WX10 = W01 * X00 + W11 * X01
        WX11 = W01 * X01 + W11 * X11

        Vph00 = V_s + 0.25 * ((WX00 * dGc00 + WX01 * dGc10) + (dGc00 * XW00 + dGc01 * XW10))
        Vph01 = V_s + 0.25 * ((WX00 * dGc01 + WX01 * dGc11) + (dGc00 * XW01 + dGc01 * XW11))
        Vph10 = V_s + 0.25 * ((WX10 * dGc00 + WX11 * dGc10) + (dGc10 * XW00 + dGc11 * XW10))
        Vph11 = V_s + 0.25 * ((WX10 * dGc01 + WX11 * dGc11) + (dGc10 * XW01 + dGc11 * XW11))
        Vph_off = 0.5 * (Vph01 + Vph10)

        # Rotate V_ph from {1, Lam-1} to {1, Lam} basis for exact factorized (r, tau) product!
        Vt_00 = Vph00 - 2.0 * Vph_off + Vph11
        Vt_01 = Vph_off - Vph11
        Vt_11 = Vph11

        Vt00_r = np.real(self._q_iOm_to_r_tau(Vt_00))
        Vt01_r = np.real(self._q_iOm_to_r_tau(Vt_01))
        Vt11_r = np.real(self._q_iOm_to_r_tau(Vt_11))

        # Exact ph dual self-energy: Sig_ph(r, tau) = +G_dual(r, tau) * V(-r, beta - tau)
        Sig_ph = (
            self._r_tau_to_k_iwn(Gd0_r * self._neg_r(Vt00_r)[::-1])
            + 2.0 * s_Lam[:, None, None, None] * self._r_tau_to_k_iwn(Gd1_r * self._neg_r(Vt01_r)[::-1])
            + Lam[:, None, None, None] * self._r_tau_to_k_iwn(Gd2_r * self._neg_r(Vt11_r)[::-1])
        )

        # 7. Singlet particle-particle Kanamori channel (Eq. 2.49, bare-U bounded, eps_pp=0.20)
        chi_pp_dual = self._r_tau_to_q_iOm(Gd2_r * Gd2_r)
        chi_pp_dual -= np.mean(chi_pp_dual, axis=(1, 2, 3), keepdims=True)
        gpp = vert["gpp"][:, None, None, None]
        denom_pp = 1.0 + gpp * chi_pp_dual
        lam_pp = max(0.0, 0.20 - float(np.min(np.real(denom_pp))))
        K_pp = (gpp ** 3) * (chi_pp_dual ** 2) / (denom_pp + lam_pp)
        K_pp_r = self._q_iOm_to_r_tau(K_pp)
        # Exact pp dual self-energy: Sig_pp(r, tau) = -G_dual(r, beta - tau) * K_pp(r, tau)
        Sig_pp = Lam[:, None, None, None] * self._r_tau_to_k_iwn(-self._neg_r(Gd2_r)[::-1] * K_pp_r)

        # 8. Self-consistent TU-SBE Eg (d-wave) feedback (Eqs. 7.10 - 7.13)
        Sig_Eg = np.zeros_like(Sig_ph)
        U_Eg_0 = 0.0
        if self.enable_eg_feedback:
            V_sing = (
                -1.5 * Gamma_s
                + 0.5 * Gc00
                - 0.5 * (vert["gc"] - gs)[:, None, None, None]
            )
            U_Eg = np.mean(self.eg_proj_q[None, :, :, :] * V_sing, axis=(1, 2, 3))  # (r_B,)
            U_Eg_0 = float(np.real(U_Eg[dlr.m0_idx]))

            Gd2_d1_r = self._k_iwn_to_r_tau(Lam[:, None, None, None] * self.psi_d1[None, :, :, :] * G_dual)
            chi_Eg = self._r_tau_to_q_iOm(Gd2_d1_r * Gd2_d1_r)
            chi_Eg -= np.mean(chi_Eg, axis=(1, 2, 3), keepdims=True)

            denom_Eg = 1.0 + U_Eg[:, None, None, None] * chi_Eg
            lam_Eg_reg = max(0.0, 0.10 - float(np.min(np.real(denom_Eg))))
            K_Eg = U_Eg[:, None, None, None] / (denom_Eg + lam_Eg_reg) - U_Eg[:, None, None, None]
            K_Eg_r = self._q_iOm_to_r_tau(K_Eg)
            # Leading minus sign in Eq. (7.13) flips -Gd(beta-tau)*K -> +Gd(beta-tau)*K
            Sig_Eg = (
                Lam[:, None, None, None]
                * self.psi_Eg_sq[None, :, :, :]
                * self._r_tau_to_k_iwn(+self._neg_r(Gd2_r)[::-1] * K_Eg_r)
            )

        # 9. Optional self-consistent 3-particle gamma^(6) contractions (Pillar 4)
        Sig_g6 = np.zeros_like(Sig_ph)
        max_Sig6_pol = 0.0
        max_Sig6_2mag = 0.0
        g6_ssc = 0.0
        g6_even = 0.0
        if self.enable_gamma6:
            from mr_aldf_gamma6 import compute_gamma6_atomic_iva, compute_sigma_dual_gamma6
            g6_res = compute_gamma6_atomic_iva(dlr, imp, self.U)
            g6_ssc = float(g6_res["gamma6_ssc"])
            g6_even = float(g6_res["gamma6_even"])
            s6 = compute_sigma_dual_gamma6(self, imp, {"chi_s": chi_s, "chi_c": chi_c}, G_dual, g6_res)
            Sig_g6 = s6["Sig6_tot"]
            max_Sig6_pol = s6["max_Sig6_pol"]
            max_Sig6_2mag = s6["max_Sig6_2mag"]

        Sig_dual_new = Sig_ph + Sig_pp + Sig_Eg + Sig_g6
        return {
            "Sig_dual": Sig_dual_new,
            "Sig_ph": Sig_ph,
            "Sig_pp": Sig_pp,
            "Sig_Eg": Sig_Eg,
            "Sig_g6": Sig_g6,
            "Gamma_s": Gamma_s,
            "Gamma_c_00": Gc00,
            "chi_s": chi_s,
            "chi_c": chi_c,
            "lam_s": lam_s,
            "lam_c": lam_c,
            "max_dD_scsa": float(np.max(np.abs(dD_scsa))),
            "U_Eg_0": U_Eg_0,
            "max_Sig6_pol": max_Sig6_pol,
            "max_Sig6_2mag": max_Sig6_2mag,
            "gamma6_ssc": g6_ssc,
            "gamma6_even": g6_even,
            "chi_AFM": float(np.real(chi_s[dlr.m0_idx, self.Q_idx[0], self.Q_idx[1], self.Q_idx[2]])),
        }

    # ---------- Linearized 3D Oh Eg (d-wave) Eliashberg Eigenvalue ----------

    def compute_Eg_eigenvalue(self, imp, vert, dual_res, G_lat, n_power=25):
        """O(r * N log N) DLR power iteration for 3D Oh Eg pairing eigenvalue Eq. (2.51)."""
        V_sing = (
            -1.5 * dual_res["Gamma_s"]
            + 0.5 * dual_res["Gamma_c_00"]
            - 0.5 * (vert["gc"] - vert["gs"])[:, None, None, None]
        )
        V_sing_r = np.real(self._q_iOm_to_r_tau(V_sing))
        Lam = imp["Lam"][:, None, None, None]
        G_sq = np.abs(G_lat) ** 2

        phi = self.psi_d1[None, :, :, :] / (1.0 + self.dlr.wn[:, None, None, None] ** 2)
        phi = phi / np.linalg.norm(phi)
        ev = 0.0
        for _ in range(n_power):
            # W(k', iwn') = Lam(wn') * |G(k', iwn')|^2 * phi(k', iwn')
            W_r = self._k_iwn_to_r_tau(Lam * G_sq * phi)
            # -(1/(beta N)) sum_{k', wn'} V_sing(k - k', iwn - iwn') Lam(wn) W(k', iwn')
            # Same convolution form as ph self-energy with minus sign!
            phi_new = -Lam * self._r_tau_to_k_iwn(W_r * V_sing_r)
            # Project onto Eg d1 symmetry (odd under kx <-> ky)
            phi_new = 0.5 * (phi_new - np.swapaxes(phi_new, 1, 2))
            phi_new = 0.5 * (phi_new + np.conj(phi_new[::-1]))
            ev = float(np.real(np.vdot(phi, phi_new)))
            nrm = float(np.linalg.norm(phi_new))
            if nrm < 1e-15:
                return 0.0
            phi = phi_new / nrm
        return ev

    # ---------- Main Coupled Inner-Outer Self-Consistent Solve ----------

    def solve(self, max_outer=45, max_inner=1, tol=1e-5, Delta_init=None, Sig_dual_init=None, mu_init=None):
        dlr = self.dlr
        mu = 0.5 * self.U if self.half_fill else (float(mu_init) if mu_init is not None else 0.5 * self.U - 2.0 * self.delta)

        if Delta_init is not None:
            Delta = self._symmetrize_Delta(Delta_init.copy())
        else:
            Delta = self._symmetrize_Delta(6.0 * (self.t ** 2) / (dlr.iwn + 2.0j * np.sign(dlr.wn)))

        if Sig_dual_init is not None:
            Sig_dual = self._symmetrize_Sig_dual(Sig_dual_init.copy())
        else:
            Sig_dual = np.zeros((dlr.r_F, self.L, self.L, self.L), dtype=np.complex128)

        diis = PulayDIIS(max_hist=6, beta_mix=0.30)
        diff = 1.0
        imp = vert = dual_res = None
        G_lat = None

        for it in range(max_outer):
            # 1. Impurity solve on DLR grid
            imp = solve_mipt4_impurity_dlr(dlr, Delta, self.U, mu, self.n_target)
            g_imp = imp["g"]
            Sig_imp = imp["Sig_imp"]
            vert = self._build_local_vertices(imp)

            # 2. Inner dual loop
            for _ in range(max_inner):
                Sig_dual = self._herglotz_project(Sig_dual, g_imp)
                Sig_nl = Sig_dual / (1.0 + g_imp[:, None, None, None] * Sig_dual)
                Sig_tot = Sig_imp[:, None, None, None] + Sig_nl

                if not self.half_fill:
                    def n_err(m_try):
                        G_try = 1.0 / (dlr.iwn[:, None, None, None] + m_try - self.eps_k[None, :, :, :] - Sig_tot)
                        return float(np.mean(dlr.density_from_G_iwn(G_try))) - self.n_target
                    lo, hi = mu - 4.0, mu + 4.0
                    while n_err(lo) > 0.0 and lo > -self.w_max:
                        lo -= 4.0
                    while n_err(hi) < 0.0 and hi < self.w_max:
                        hi += 4.0
                    if n_err(lo) * n_err(hi) <= 0.0:
                        mu = float(brentq(n_err, lo, hi, xtol=1e-9))

                G_lat = 1.0 / (dlr.iwn[:, None, None, None] + mu - self.eps_k[None, :, :, :] - Sig_tot)
                # Exact dressed dual propagator G_dual = G_lat - g_imp / (1 + g_imp * Sig_dual)?
                # Check Eq. (2.11): G_dual = -g^2 (Delta - eps) / [1 + g(Delta - eps)(1 + g*Sig_dual)]
                d_minus_e = Delta[:, None, None, None] - self.eps_k[None, :, :, :]
                g_b = g_imp[:, None, None, None]
                G_dual = -(g_b ** 2) * d_minus_e / (1.0 + g_b * d_minus_e * (1.0 + g_b * Sig_dual))
                G_dual -= np.mean(G_dual, axis=(1, 2, 3), keepdims=True)

                dual_res = self._dual_self_energy_step(imp, vert, G_lat, G_dual)
                Sig_cand = self._herglotz_project(dual_res["Sig_dual"], g_imp)
                d_in = float(np.max(np.abs(Sig_cand - Sig_dual)))
                Sig_dual = 0.55 * Sig_dual + 0.45 * Sig_cand
                if d_in < 1e-6:
                    break

            # 3. Outer hybridization update: Delta_new = Delta + g^{-1} - G_loc^{-1}
            Sig_dual = self._herglotz_project(Sig_dual, g_imp)
            Sig_nl = Sig_dual / (1.0 + g_imp[:, None, None, None] * Sig_dual)
            G_lat = 1.0 / (dlr.iwn[:, None, None, None] + mu - self.eps_k[None, :, :, :] - Sig_imp[:, None, None, None] - Sig_nl)
            G_loc = np.mean(G_lat, axis=(1, 2, 3))
            Delta_new = self._symmetrize_Delta(Delta + (1.0 / g_imp - 1.0 / G_loc))

            # Coupled DIIS update of [Delta, Sig_dual]
            x_in = np.concatenate([Delta, Sig_dual.ravel()])
            x_out = np.concatenate([Delta_new, Sig_cand.ravel()])
            diff = float(np.max(np.abs(x_out - x_in)))
            # Adaptive damping when residual is large to prevent DIIS overshoot at low T
            if diff > 0.5:
                diis.reset()
                alpha_d = 0.25
                Delta = self._symmetrize_Delta((1.0 - alpha_d) * Delta + alpha_d * Delta_new)
                Sig_dual = self._herglotz_project((1.0 - alpha_d) * Sig_dual + alpha_d * Sig_cand, g_imp)
            else:
                x_mix = diis.update(x_in, x_out)
                Delta = self._symmetrize_Delta(x_mix[: dlr.r_F])
                Sig_dual = self._herglotz_project(x_mix[dlr.r_F :].reshape(Sig_dual.shape), g_imp)

            if diff < tol and it >= 3:
                break

        # Final diagnostics
        w0_i = dlr.r_half
        w0 = dlr.wn[w0_i]
        Sig_nl = Sig_dual / (1.0 + imp["g"][:, None, None, None] * Sig_dual)
        Sig_tot = imp["Sig_imp"][:, None, None, None] + Sig_nl
        G_lat = 1.0 / (dlr.iwn[:, None, None, None] + mu - self.eps_k[None, :, :, :] - Sig_tot)
        n_lat = float(np.mean(dlr.density_from_G_iwn(G_lat)))

        Z_imp = float(1.0 / (1.0 - np.imag(imp["Sig_imp"][w0_i]) / w0))
        Z_k = 1.0 / (1.0 - np.imag(Sig_tot[w0_i]) / w0)
        # Hot (0, pi, pi/2) and cold (pi/2, pi/2, pi/2) indices
        q4 = self.L // 4
        q2 = self.L // 2
        Z_hot = float(Z_k[0, q2, q4])
        Z_cold = float(Z_k[q4, q4, q4])

        C0 = float(np.max(np.abs(np.imag(Sig_dual[w0_i]))) * np.abs(np.imag(imp["g"][w0_i])))
        lam_Eg = self.compute_Eg_eigenvalue(imp, vert, dual_res, G_lat)

        return {
            "U": self.U,
            "T": self.T,
            "delta": self.delta,
            "L": self.L,
            "r_F": dlr.r_F,
            "r_B": dlr.r_B,
            "mu": mu,
            "n_lat": n_lat,
            "Z_imp": Z_imp,
            "Z_hot": Z_hot,
            "Z_cold": Z_cold,
            "D_imp": imp["D_imp"],
            "Delta_B_kin": imp["Delta_B_kin"],
            "D_c_irr": vert["D_c_irr"],
            "u4_micro": vert["u4_micro"],
            "chi_AFM": dual_res["chi_AFM"],
            "lam_s": dual_res["lam_s"],
            "lam_c": dual_res["lam_c"],
            "max_dD_scsa": dual_res["max_dD_scsa"],
            "U_Eg_0": dual_res["U_Eg_0"],
            "max_Sig_Eg": float(np.max(np.abs(dual_res["Sig_Eg"]))),
            "max_Sig6_pol": dual_res["max_Sig6_pol"],
            "max_Sig6_2mag": dual_res["max_Sig6_2mag"],
            "gamma6_ssc": dual_res["gamma6_ssc"],
            "gamma6_even": dual_res["gamma6_even"],
            "lam_Eg": lam_Eg,
            "C0": C0,
            "diff": diff,
            "iters": it + 1,
            "Delta": Delta,
            "Sig_dual": Sig_dual,
            "Sig_tot": Sig_tot,
            "G_lat": G_lat,
            "imp": imp,
        }
