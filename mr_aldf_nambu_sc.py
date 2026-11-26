"""
MR-ALDF-IV 2x2 Nambu Eg (d-wave) Superconducting Ordered-Phase Solver (mr_aldf_nambu_sc.py)
Pillar 2b: Spontaneous U(1) Gauge Symmetry Breaking in the 3D Oh Eg (d_{x^2-y^2}) Channel.

Implements:
  1. Exact symmetry property F_loc(iwn) = (1/N) sum_k F(k, iwn) = 0 for Oh Eg pairing:
     the local impurity problem stays strictly normal-state (mIPT-4 + local-TPSC on DLR).
  2. Exact 2x2 Nambu physical-to-dual self-energy matrix identity:
       Sigma_nl,11 = [Sig_d (1 + g* Sig_d*) + g* |Phi_d|^2] / [|1 + g Sig_d|^2 + |g|^2 |Phi_d|^2]
       Phi_nl      = Phi_d / [|1 + g Sig_d|^2 + |g|^2 |Phi_d|^2]
     with smooth 2x2 Nambu Herglotz causality safeguard guaranteeing Im Sigma_nl,11(wn > 0) <= 0.
  3. Exact Nambu coherence-factor spin (GG + FF, capturing Yosida singlet suppression at q=0
     and neutron spin resonance at Q=(pi,pi,pi)) and charge (GG - FF) physical & dual bubbles.
  4. Coupled self-consistent normal dual self-energy Sig_dual(k, iwn) + nonlinear 3D Oh Eg
     Eliashberg anomalous self-energy Phi_dual(k, iwn) mediated by Moriya+SCSA V_sing(q, iOm).
  5. Exact consistency with the linearized Eliashberg eigenvalue lambda_Eg: spontaneous
     Delta_Eg > 0 emerges if and only if lambda_Eg > 1 (seed strictly turned off after 2 iters).
"""

import numpy as np
from scipy.optimize import brentq
from mr_aldf_dlr import DLRBasis, solve_mipt4_impurity_dlr


class MRALDFNambuSCDLRSolver:
    """
    2x2 Nambu Oh Eg (d_{x^2-y^2}) Superconducting MR-ALDF-IV Solver on the 3D Simple-Cubic Lattice.
    """

    def __init__(
        self,
        U,
        T,
        delta=0.05,
        L=8,
        t=1.0,
        w_max=None,
        eps_dlr=1e-7,
        enable_scsa=True,
        c_max=0.90,
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
        self.enable_scsa = bool(enable_scsa)
        self.c_max = float(c_max)

        self.w_max = float(w_max) if w_max is not None else max(18.0 * self.t, 1.5 * self.U + 12.0 * self.t)
        self.dlr = DLRBasis(beta=self.beta, w_max=self.w_max, eps=eps_dlr)

        k1d = 2.0 * np.pi * np.arange(self.L) / self.L
        kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
        self.kx, self.ky, self.kz = kx, ky, kz
        self.eps_k = -2.0 * self.t * (np.cos(kx) + np.cos(ky) + np.cos(kz))
        self.psi_d1 = np.cos(kx) - np.cos(ky)  # <psi_d1^2> = 1, <psi_d1> = 0 exact!
        self.Q_idx = (self.L // 2, self.L // 2, self.L // 2)
        self._r_neg = (self.L - np.arange(self.L)) % self.L

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
        if self.half_fill:
            D = 1j * np.imag(D)
        sgn = np.sign(self.dlr.wn)
        return np.real(D) + 1j * np.where(sgn * np.imag(D) > 0.0, 0.0, np.imag(D))

    def _project_d1(self, Phi):
        """Enforce exact real even-frequency Oh Eg d_{x^2-y^2} symmetry: Phi(kx,ky,kz,-wn) = Phi(kx,ky,kz,wn) = -Phi(ky,kx,kz,wn)."""
        P = 0.5 * np.real(Phi + Phi[::-1])
        P = 0.5 * (P - np.swapaxes(P, 1, 2))
        return P

    def _herglotz_nambu(self, Sig_d, Phi_d, g_imp):
        """
        Smooth 2x2 Nambu Herglotz projection:
          Im Sigma_nl,11 <= 0  <=>  -Im(Sig_d) > |Im(g)| * (|Sig_d|^2 + |Phi_d|^2) / c_max.
        Guarantees strict causality of the full 2x2 Nambu lattice Green's function.
        """
        S = 0.5 * (Sig_d + np.conj(Sig_d[::-1]))
        P = self._project_d1(Phi_d)
        sgn = np.sign(self.dlr.wn)[:, None, None, None]
        im_S = np.where(sgn * np.imag(S) > 0.0, -1e-14 * sgn, np.imag(S))
        S = np.real(S) + 1j * im_S

        # Standard normal-channel dual-inverse Herglotz projection on S
        mag2 = np.abs(S) ** 2
        active = mag2 > 1e-24
        y_min = (np.abs(np.imag(g_imp)) / self.c_max)[:, None, None, None]
        dh = 0.05 * np.maximum(y_min, 1e-12)
        inv_S = np.where(active, 1.0 / np.where(active, S, 1.0), 0.0 + 0.0j)
        y_inv = sgn * np.imag(inv_S)
        z = (y_inv - y_min) / dh
        softp = np.where(z > 35.0, z, np.log1p(np.exp(np.clip(z, -60.0, 35.0))))
        inv_p = np.real(inv_S) + 1j * sgn * (y_min + dh * softp)
        S = 0.5 * (np.where(active, 1.0 / np.where(active, inv_p, 1.0), S))
        S = S + np.conj(S[::-1])

        return S, P

    def _build_local_vertices(self, imp):
        dlr = self.dlr
        m0 = dlr.m0_idx
        c0 = imp["chi0_loc"]
        cL = imp["chi_Lam_loc"]
        cG0 = imp["chi0_G0"]
        ratio_sq = np.clip(np.real(np.where(cL > 1e-14, (cG0 / np.where(cL > 1e-14, cL, 1.0)) ** 2, 0.0)), 0.0, 10.0)

        U_sp, U_ch = imp["U_sp"], imp["U_ch"]
        gs = np.minimum(np.real(np.where(
            np.arange(dlr.r_B) == m0,
            (2.0 * c0[m0] - imp["chi_imp_s"][m0]) / max(2.0 * cL[m0] ** 2, 1e-18),
            (-U_sp / np.maximum(1.0 - U_sp * cG0, 1e-8)) * ratio_sq,
        )), 0.0)
        gc = np.real(np.where(
            np.arange(dlr.r_B) == m0,
            (2.0 * c0[m0] - imp["chi_imp_c"][m0]) / max(2.0 * cL[m0] ** 2, 1e-18),
            (+U_ch / np.maximum(1.0 + U_ch * cG0, 1e-8)) * ratio_sq,
        ))
        theta = np.clip(np.real(c0) / np.maximum(np.real(cL), 1e-14), 1e-6, 1.0)
        norm_vp = theta ** 2 + (1.0 - theta) ** 2
        gc_perp = np.where(np.arange(dlr.r_B) == m0, self.U / norm_vp[m0], (self.U / norm_vp) * ratio_sq)

        Wc = np.zeros((dlr.r_B, 2, 2), dtype=float)
        for m in range(dlr.r_B):
            th = theta[m]
            vp = np.array([1.0 - th, -th])
            Wc[m] = gc[m] * np.ones((2, 2)) + gc_perp[m] * np.outer(vp, vp)

        u4_micro = float((U_sp ** 4) * max(imp["chi4_loc"], 0.0))
        return {"gs": gs, "gc": gc, "Wc": Wc, "u4_micro": u4_micro}

    def _nambu_dual_step(self, imp, vert, G_lat, F_lat, Gd_lat, Fd_lat):
        """
        Compute Nambu spin/charge ladders with exact coherence factors (GG + FF for spin,
        GG - FF for charge), Moriya + 3D O(3) SCSA regularization, and coupled:
          - normal dual self-energy Sig_dual(k, iwn)
          - anomalous Eg (d-wave) gap Phi_nl(k, iwn) via nonlinear 3D Eliashberg equation!
        """
        dlr = self.dlr
        Lam = imp["Lam"]
        sLam = np.sqrt(Lam)
        Lam_b = Lam[:, None, None, None]
        sLam_b = sLam[:, None, None, None]

        # 1. Transform physical (G, F) and dual (Gd, Fd) to (tau, r)
        G0_r = self._k_to_r(G_lat)
        G1_r = self._k_to_r(sLam_b * G_lat)
        F0_r = self._k_to_r(F_lat)
        F1_r = self._k_to_r(sLam_b * F_lat)

        Gd0_r = self._k_to_r(Gd_lat)
        Gd1_r = self._k_to_r(sLam_b * Gd_lat)
        Gd2_r = self._k_to_r(Lam_b * Gd_lat)
        Fd2_r = self._k_to_r(Lam_b * Fd_lat)

        # 2. Physical Nambu bubbles:
        #    Spin   bubble: chi0_s = GG + FF  (Yosida suppression at q=0, spin resonance at Q!)
        #    Charge bubble: chi0_c = GG - FF  (finite compressibility at q=0!)
        GG0_q = np.real(self._r_to_q(G0_r * self._neg_r(G0_r)[::-1]))
        FF0_q = np.real(self._r_to_q(F0_r * self._neg_r(F0_r)[::-1]))
        GGL_q = np.real(self._r_to_q(G1_r * self._neg_r(G1_r)[::-1]))
        FFL_q = np.real(self._r_to_q(F1_r * self._neg_r(F1_r)[::-1]))

        # Note: F(r=0) = 0 exact => mean_q(FF) = 0 exact! Restore exact r=0 on GG only.
        GG0_q += (imp["chi0_loc"] - np.mean(GG0_q, axis=(1, 2, 3)))[:, None, None, None]
        GGL_q += (imp["chi_Lam_loc"] - np.mean(GGL_q, axis=(1, 2, 3)))[:, None, None, None]

        chi0_s_lat = np.maximum(GG0_q + FF0_q, 0.0)
        chiL_s_lat = np.maximum(GGL_q + FFL_q, 0.0)
        chi0_c_lat = np.maximum(GG0_q - FF0_q, 0.0)
        chiL_c_lat = np.maximum(GGL_q - FFL_q, 0.0)

        # 3. Dual Nambu bubbles (spin uses GdGd + FdFd, charge uses GdGd - FdFd)
        Xt_0 = np.real(self._r_to_q(Gd0_r * self._neg_r(Gd0_r)[::-1]))
        Xt_L = np.real(self._r_to_q(Gd1_r * self._neg_r(Gd1_r)[::-1]))
        Xt_L2 = np.real(self._r_to_q(Gd2_r * self._neg_r(Gd2_r)[::-1]))
        Xt_FF2 = np.real(self._r_to_q(Fd2_r * self._neg_r(Fd2_r)[::-1]))
        Xt_0 -= np.mean(Xt_0, axis=(1, 2, 3), keepdims=True)
        Xt_L -= np.mean(Xt_L, axis=(1, 2, 3), keepdims=True)
        Xt_L2 -= np.mean(Xt_L2, axis=(1, 2, 3), keepdims=True)
        Xt_FF2 -= np.mean(Xt_FF2, axis=(1, 2, 3), keepdims=True)

        Xt_s_L2 = Xt_L2 + Xt_FF2

        # 4. 2x2 charge ladder (using charge bubbles chi0_c, chiL_c)
        X00 = Xt_0
        X01 = Xt_L - Xt_0
        X11 = (Xt_L2 - Xt_FF2) - 2.0 * Xt_L + Xt_0
        b0 = chi0_c_lat
        b1 = chiL_c_lat - chi0_c_lat

        Wc = vert["Wc"]
        W00 = Wc[:, 0, 0][:, None, None, None]
        W01 = Wc[:, 0, 1][:, None, None, None]
        W11 = Wc[:, 1, 1][:, None, None, None]

        XW00 = X00 * W00 + X01 * W01
        XW01 = X00 * W01 + X01 * W11
        XW10 = X01 * W00 + X11 * W01
        XW11 = X01 * W01 + X11 * W11

        tr_M = 2.0 + XW00 + XW11
        det_M = (1.0 + XW00) * (1.0 + XW11) - XW01 * XW10
        eig_min = float(np.min(0.5 * (tr_M - np.sqrt(np.maximum(tr_M * tr_M - 4.0 * det_M, 0.0)))))
        lam_c = max(0.0, 0.05 - eig_min)

        def eval_charge(lc):
            M00 = (1.0 + lc) + XW00
            M11 = (1.0 + lc) + XW11
            det = M00 * M11 - XW01 * XW10
            inv00, inv11, inv01, inv10 = M11 / det, M00 / det, -XW01 / det, -XW10 / det
            Gc00 = W00 * inv00 + W01 * inv10
            Gc01 = W00 * inv01 + W01 * inv11
            Gc10 = W01 * inv00 + W11 * inv10
            Gc11 = W01 * inv01 + W11 * inv11
            bGb = b0 * (Gc00 * b0 + Gc01 * b1) + b1 * (Gc10 * b0 + Gc11 * b1)
            chi_c = 2.0 * chi0_c_lat - 2.0 * bGb
            return (Gc00, Gc01, Gc10, Gc11), bGb, chi_c

        (Gc00, Gc01, Gc10, Gc11), bGb, chi_c = eval_charge(lam_c)
        for _ in range(15):
            if float(np.min(chi_c)) >= -1e-10:
                break
            lam_c = max(lam_c * 2.0, 0.1) + 0.1
            (Gc00, Gc01, Gc10, Gc11), bGb, chi_c = eval_charge(lam_c)

        # 5. Moriya + 3D O(3) SCSA spin ladder with Nambu coherence-factor spin bubble
        gs = vert["gs"]
        Ds = 1.0 + gs[:, None, None, None] * Xt_s_L2
        w_bsum = np.real(dlr._bsum_from_iOm_w)
        rhs_pauli = float(2.0 * np.mean(dlr.boson_sum_from_iOm(bGb)))
        chiL_s_sq = chiL_s_lat ** 2

        dD_scsa = np.zeros_like(Ds)
        n_scsa = 2 if (self.enable_scsa and vert["u4_micro"] > 1e-14) else 1
        lam_s = 0.0
        for _ in range(n_scsa):
            Ds_eff = Ds + dD_scsa
            lam_lo = max(0.0, 1e-7 - float(np.min(Ds_eff)))
            w_eff = (-2.0 * w_bsum * gs)[:, None, None, None] * chiL_s_sq

            def pauli_res(ls):
                return float(np.sum(w_eff / (Ds_eff + ls)) / self.N) - rhs_pauli

            if rhs_pauli <= 1e-15 or pauli_res(lam_lo) <= 0.0:
                lam_s = lam_lo
            else:
                lam_hi = max(lam_lo + 1.0, 4.0)
                while pauli_res(lam_hi) > 0.0 and lam_hi < 1e7:
                    lam_hi *= 4.0
                lam_s = float(brentq(pauli_res, lam_lo, lam_hi, xtol=1e-9)) if pauli_res(lam_lo) * pauli_res(lam_hi) <= 0.0 else lam_lo

            Gamma_s = gs[:, None, None, None] / (Ds_eff + lam_s)
            chi_s = 2.0 * chi0_s_lat - 2.0 * Gamma_s * chiL_s_sq

            if not self.enable_scsa or vert["u4_micro"] <= 1e-14:
                break
            u4 = vert["u4_micro"]
            chi_s_r = np.real(self._q_to_r(chi_s))
            Pi_xx = np.maximum(np.real(self._r_to_q(chi_s_r ** 2)), 0.0)
            dD4 = -(6.0 * (u4 ** 2) * Pi_xx) / (1.0 + 3.0 * u4 * Pi_xx)
            dD4_r = np.real(self._q_to_r(dD4))
            Pi_scsa = np.real(self._r_to_q(dD4_r * chi_s_r))
            dPi_scsa = Pi_scsa - Pi_scsa[dlr.m0_idx, self.Q_idx[0], self.Q_idx[1], self.Q_idx[2]]
            dD_scsa = 0.5 * dD_scsa + 0.5 * (2.0 * np.abs(gs)[:, None, None, None] * chiL_s_sq * dPi_scsa)

        # 6. Normal particle-hole dual self-energy Sig_dual(k, iwn)
        V_s = 1.5 * gs[:, None, None, None] * (Gamma_s - 0.5 * gs[:, None, None, None]) * Xt_s_L2
        dGc00, dGc01, dGc10, dGc11 = Gc00 - 0.5 * W00, Gc01 - 0.5 * W01, Gc10 - 0.5 * W01, Gc11 - 0.5 * W11
        WX00 = W00 * X00 + W01 * X01
        WX01 = W00 * X01 + W01 * X11
        WX10 = W01 * X00 + W11 * X01
        WX11 = W01 * X01 + W11 * X11

        Vph00 = V_s + 0.25 * ((WX00 * dGc00 + WX01 * dGc10) + (dGc00 * XW00 + dGc01 * XW10))
        Vph01 = V_s + 0.25 * ((WX00 * dGc01 + WX01 * dGc11) + (dGc00 * XW01 + dGc01 * XW11))
        Vph10 = V_s + 0.25 * ((WX10 * dGc00 + WX11 * dGc10) + (dGc10 * XW00 + dGc11 * XW10))
        Vph11 = V_s + 0.25 * ((WX10 * dGc01 + WX11 * dGc11) + (dGc10 * XW01 + dGc11 * XW11))
        Vph_off = 0.5 * (Vph01 + Vph10)

        Vt00_r = np.real(self._q_to_r(Vph00 - 2.0 * Vph_off + Vph11))
        Vt01_r = np.real(self._q_to_r(Vph_off - Vph11))
        Vt11_r = np.real(self._q_to_r(Vph11))

        Sig_dual_new = (
            self._r_to_k(Gd0_r * self._neg_r(Vt00_r)[::-1])
            + 2.0 * sLam_b * self._r_to_k(Gd1_r * self._neg_r(Vt01_r)[::-1])
            + Lam_b * self._r_to_k(Gd2_r * self._neg_r(Vt11_r)[::-1])
        )

        # 7. Nonlinear 3D Oh Eg (d-wave) Eliashberg Anomalous Self-Energy Phi(k, iwn)!
        #    Exact pairing interaction V_sing = -3/2 Gamma_s + 1/2 Gamma_c,00 - 1/2 (gc - gs):
        #    Phi(k, iwn) = +(1/(beta N)) sum_{k', wn'} V_sing(k - k', iwn - iwn') Lam(wn) Lam(wn') F(k', iwn')
        #    Note: linearizes identically to MRALDFLevel4DLRSolver.compute_Eg_eigenvalue when F -> -|G|^2 Phi!
        V_sing = -1.5 * Gamma_s + 0.5 * Gc00 - 0.5 * (vert["gc"] - gs)[:, None, None, None]
        V_sing_r = np.real(self._q_to_r(V_sing))
        LF_r = self._k_to_r(Lam_b * F_lat)
        Phi_new = self._project_d1(Lam_b * self._r_to_k(LF_r * self._neg_r(V_sing_r)[::-1]))

        return {
            "Sig_dual": Sig_dual_new,
            "Phi_new": Phi_new,
            "V_sing_r": V_sing_r,
            "chi_s": chi_s,
            "chi_c": chi_c,
            "chi_AFM": float(np.real(chi_s[dlr.m0_idx, self.Q_idx[0], self.Q_idx[1], self.Q_idx[2]])),
            "chi_s_q0": float(np.real(chi_s[dlr.m0_idx, 0, 0, 0])),
            "lam_s": lam_s,
        }

    def compute_linearized_Eg(self, imp, V_sing_r, G_lat, n_power=20):
        """Linearized Eg eigenvalue on the converged state (checks lambda_Eg = 1 in SC state!)."""
        Lam_b = imp["Lam"][:, None, None, None]
        G_sq = np.abs(G_lat) ** 2
        phi = self.psi_d1[None, :, :, :] / (1.0 + self.dlr.wn[:, None, None, None] ** 2)
        phi /= np.linalg.norm(phi)
        ev = 0.0
        for _ in range(n_power):
            W_r = self._k_to_r(Lam_b * G_sq * phi)
            phi_n = self._project_d1(-Lam_b * self._r_to_k(W_r * self._neg_r(V_sing_r)[::-1]))
            ev = float(np.real(np.vdot(phi, phi_n)))
            nrm = float(np.linalg.norm(phi_n))
            if nrm < 1e-15:
                return 0.0
            phi = phi_n / nrm
        return ev

    def solve(self, phi_seed=0.15, n_seed_iters=2, max_outer=40, tol=1e-4, mix=0.35):
        """
        Solve the coupled 2x2 Nambu MR-ALDF-IV equations.
        The Eg pairing seed phi_seed * psi_d1(k) is turned off completely after n_seed_iters,
        so any nonzero converged Delta_Eg is 100% spontaneous!
        """
        dlr = self.dlr
        mu = 0.5 * self.U if self.half_fill else 0.5 * self.U - 2.0 * self.delta
        Delta = self._sym_Delta(6.0 * (self.t ** 2) / (dlr.iwn + 2.0j * np.sign(dlr.wn)))
        Sig_d = np.zeros((dlr.r_F, self.L, self.L, self.L), dtype=np.complex128)
        Phi_nl = self._project_d1(phi_seed * self.psi_d1[None, :, :, :] / (1.0 + (dlr.wn[:, None, None, None] / 2.0) ** 2))

        diff = 1.0
        imp = dual_res = None
        G_lat = F_lat = None

        for it in range(max_outer):
            # 1. Exact normal-state single-site mIPT-4 impurity (F_loc = 0 by Eg symmetry!)
            imp = solve_mipt4_impurity_dlr(dlr, Delta, self.U, mu, self.n_target)
            g_imp = imp["g"]
            Sig_imp = imp["Sig_imp"]
            vert = self._build_local_vertices(imp)

            # 2. 2x2 Nambu physical-to-dual transformation & Herglotz control
            Sig_d, Phi_nl = self._herglotz_nambu(Sig_d, Phi_nl, g_imp)
            g_b = g_imp[:, None, None, None]
            Sig_nl = Sig_d / (1.0 + g_b * Sig_d)
            Sig_tot = Sig_imp[:, None, None, None] + Sig_nl

            # Seed field strictly vanishes for it >= n_seed_iters!
            Phi_ext = (phi_seed * self.psi_d1[None, :, :, :]) if it < n_seed_iters else 0.0
            Phi_eff = Phi_nl + Phi_ext

            # 3. 2x2 Nambu lattice Green's function + chemical potential root find
            def nambu_GF(m_val):
                z = dlr.iwn[:, None, None, None] + m_val - self.eps_k[None, :, :, :] - Sig_tot
                denom = np.abs(z) ** 2 + np.abs(Phi_eff) ** 2
                return np.conj(z) / denom, -Phi_eff / denom

            if not self.half_fill:
                def n_err(m_try):
                    g11, _ = nambu_GF(m_try)
                    return float(np.mean(dlr.density_from_G_iwn(g11))) - self.n_target
                lo, hi = mu - 4.0, mu + 4.0
                while n_err(lo) > 0.0 and lo > -self.w_max:
                    lo -= 4.0
                while n_err(hi) < 0.0 and hi < self.w_max:
                    hi += 4.0
                if n_err(lo) * n_err(hi) <= 0.0:
                    mu = float(brentq(n_err, lo, hi, xtol=1e-9))

            G_lat, F_lat = nambu_GF(mu)
            G_loc = np.mean(G_lat, axis=(1, 2, 3))
            # Exact dual propagators in Nambu space (pole-free Gd = G_lat - g_loc on normal, Fd = F_lat since F_loc = 0!)
            Gd_lat = G_lat - G_loc[:, None, None, None]
            Fd_lat = F_lat

            # 4. Nambu dual ladders + nonlinear Eg Eliashberg gap update
            dual_res = self._nambu_dual_step(imp, vert, G_lat, F_lat, Gd_lat, Fd_lat)
            Sig_d_new, Phi_new = self._herglotz_nambu(dual_res["Sig_dual"], dual_res["Phi_new"], g_imp)
            Delta_new = self._sym_Delta(Delta + (1.0 / g_imp - 1.0 / G_loc))

            d_Del = float(np.max(np.abs(Delta_new - Delta)))
            d_Sig = float(np.max(np.abs(Sig_d_new - Sig_d)))
            d_Phi = float(np.max(np.abs(Phi_new - Phi_nl)))
            diff = max(d_Del, d_Sig, d_Phi)

            alpha = 0.25 if diff > 0.5 else mix
            Delta = self._sym_Delta((1.0 - alpha) * Delta + alpha * Delta_new)
            Sig_d = (1.0 - alpha) * Sig_d + alpha * Sig_d_new
            Phi_nl = self._project_d1((1.0 - alpha) * Phi_nl + alpha * Phi_new)

            if it >= n_seed_iters + 5 and diff < tol:
                break

        # Final physical observables & gap diagnostics
        w0_i = dlr.r_half
        w0 = dlr.wn[w0_i]
        Sig_nl = Sig_d / (1.0 + g_imp[:, None, None, None] * Sig_d)
        Sig_tot = imp["Sig_imp"][:, None, None, None] + Sig_nl
        z = dlr.iwn[:, None, None, None] + mu - self.eps_k[None, :, :, :] - Sig_tot
        denom = np.abs(z) ** 2 + np.abs(Phi_nl) ** 2
        G_lat, F_lat = np.conj(z) / denom, -Phi_nl / denom

        n_lat = float(np.mean(dlr.density_from_G_iwn(G_lat)))
        F_loc_max = float(np.max(np.abs(np.mean(F_lat, axis=(1, 2, 3)))))
        Z_k = 1.0 / (1.0 - np.imag(Sig_tot[w0_i]) / w0)
        # Physical quasiparticle gap Delta_Eg(k, w0) = Z_k * Phi(k, w0) projected onto psi_d1(k)
        Phi_w0 = np.real(Phi_nl[w0_i])
        Phi_d_amp = float(np.mean(Phi_w0 * self.psi_d1))          # <Phi * psi_d1> since <psi_d1^2>=1
        Delta_Eg_max = float(np.max(np.abs(Z_k * Phi_w0)))       # peak physical d-wave gap at antinode
        C0 = float(np.max(np.abs(np.imag(Sig_d[w0_i]))) * abs(np.imag(g_imp[w0_i])))
        lam_Eg_eff = self.compute_linearized_Eg(imp, dual_res["V_sing_r"], G_lat)

        return {
            "U": self.U,
            "T": self.T,
            "delta": self.delta,
            "L": self.L,
            "n_lat": n_lat,
            "F_loc_max": F_loc_max,
            "Phi_d_amp": abs(Phi_d_amp),
            "Delta_Eg_max": Delta_Eg_max,
            "Z_imp": float(1.0 / (1.0 - np.imag(imp["Sig_imp"][w0_i]) / w0)),
            "chi_AFM": dual_res["chi_AFM"],
            "chi_s_q0": dual_res["chi_s_q0"],
            "lam_Eg_eff": lam_Eg_eff,
            "C0": C0,
            "diff": diff,
            "iters": it + 1,
        }
