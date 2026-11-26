"""
MR-ALDF-IV Track IV-B Full Dual-Fermion Lattice Solver (mr_aldf_track4b.py) — Pillar 3c.

Plugs the deterministic symmetry-sector Lehmann Exact Diagonalization (ED) impurity
solver (mr_aldf_ed.py) and exact 4-point Rank-R SVD vertex analyzer (mr_aldf_ed_vertex.py)
directly into Step 1 & Step 2 of the 3D MR-ALDF-IV DLR dual-fermion loop.

Provides apples-to-apples Track IV-A (mIPT-4) vs. Track IV-B (ED) validation on the
identical 3D simple-cubic lattice, DLR frequency grid, Moriya+SCSA spin ladder,
rank-2/rank-R charge ladder, TU-SBE Eg feedback, and C^inf Herglotz projection.
"""

import numpy as np
from scipy.optimize import brentq
from mr_aldf_level4_dlr import MRALDFLevel4DLRSolver, PulayDIIS
from mr_aldf_ed import EDAndersonSolver, fit_bath_dlr
from mr_aldf_ed_vertex import compute_chi_ph_static_ed, extract_rank_R_vertices


class MRALDFTrack4BSolver(MRALDFLevel4DLRSolver):
    """
    Track IV-B MR-ALDF-IV solver: replaces Step 1 mIPT-4 with exact symmetry-sector
    Lehmann ED (N_b = 3-5) on the DLR grid, uses exact ED local susceptibilities
    chi_imp_{s,c,pp}(iOm) in the dual ladders, and extracts exact Rank-R (R = 6) SVD
    vertices and multi-line Gamma_c^irr divergence counts on the converged lattice state.
    """

    def __init__(
        self,
        U,
        T,
        delta=0.0,
        L=8,
        t=1.0,
        N_b=4,
        w_max=None,
        eps_dlr=1e-7,
        enable_scsa=True,
        enable_eg_feedback=True,
        enable_herglotz=True,
        c_max=0.92,
    ):
        super().__init__(
            U=U, T=T, delta=delta, L=L, t=t,
            w_max=w_max, eps_dlr=eps_dlr,
            enable_scsa=enable_scsa,
            enable_eg_feedback=enable_eg_feedback,
            enable_herglotz=enable_herglotz,
            c_max=c_max,
        )
        self.N_b = int(N_b)
        self.ed = EDAndersonSolver(N_b=self.N_b)
        self._last_bath_x = None

    def _solve_ed_impurity(self, Delta_iwn, mu):
        """
        Fit Delta_iwn -> (eps_b, V_b) on DLR, solve SIAM via exact Lehmann ED,
        and populate the full Track IV-A/IV-B impurity interface dictionary.
        """
        dlr = self.dlr
        eps_b, V_b, Delta_fit, fit_err, x_opt = fit_bath_dlr(
            dlr, Delta_iwn, N_b=self.N_b, half_fill=self.half_fill, init_params=self._last_bath_x
        )
        self._last_bath_x = x_opt

        ed_res = self.ed.solve(dlr, U=self.U, mu=mu, eps_b=eps_b, V_b=V_b, compute_chi=True)
        ed_res["fit_err"] = fit_err

        # Construct auxiliary Weiss propagator G0 on fitted bath for exact UV tail & u4_micro
        if self.half_fill:
            mu0 = 0.0
        else:
            lim_m0 = 0.75 * dlr.w_max
            def d_err(m0):
                return float(dlr.density_from_G_iwn(1.0 / (dlr.iwn + m0 - Delta_fit))) - ed_res["n_imp"]
            mu0 = float(brentq(d_err, -lim_m0, lim_m0, xtol=1e-10))

        G0_iwn = 1.0 / (dlr.iwn + mu0 - Delta_fit)
        G0_tau = np.real(dlr.iwn_to_tau(G0_iwn))
        chi0_G0 = np.maximum(np.real(dlr.tauB_to_iOm(G0_tau * G0_tau[::-1])), 1e-16)
        chi0_pp_G0 = dlr.tauB_to_iOm(G0_tau ** 2)

        # Effective static couplings matching exact ED static susceptibilities
        m0 = dlr.m0_idx
        cG0_0 = float(chi0_G0[m0])
        cs0 = max(float(np.real(ed_res["chi_imp_s"][m0])), 1e-12)
        cc0 = max(float(np.real(ed_res["chi_imp_c"][m0])), 1e-12)
        cpp0 = float(np.real(ed_res["chi_imp_pp"][m0]))
        c0pp0 = float(np.real(chi0_pp_G0[m0]))

        U_sp = float(np.clip((1.0 - 2.0 * cG0_0 / cs0) / max(cG0_0, 1e-12), 0.0, 0.9999 / max(cG0_0, 1e-12)))
        U_ch = float(np.clip((2.0 * cG0_0 / cc0 - 1.0) / max(cG0_0, 1e-12), 0.0, max(50.0 * self.U, 100.0)))
        U_pp = float(np.clip((c0pp0 / max(cpp0, 1e-12) - 1.0) / max(c0pp0, 1e-12), 0.0, max(50.0 * self.U, 100.0))) if cpp0 > 1e-12 else float(self.U)

        G0_sq = G0_iwn ** 2
        chi4_loc = float(np.real(dlr.fermion_prod_sum(G0_sq, G0_sq)))

        ed_res["mu0"] = mu0
        ed_res["G0_iwn"] = G0_iwn
        ed_res["chi0_G0"] = chi0_G0
        ed_res["chi0_pp_G0"] = chi0_pp_G0
        ed_res["U_sp"] = U_sp
        ed_res["U_ch"] = U_ch
        ed_res["U_pp"] = U_pp
        ed_res["chi4_loc"] = chi4_loc
        ed_res["Delta_B_kin"] = 0.0  # Exact ED needs no mIPT kinetic correction
        return ed_res

    def solve(
        self,
        max_outer=24,
        max_inner=1,
        tol=2e-4,
        Delta_init=None,
        Sig_dual_init=None,
        mu_init=None,
        extract_rankR_final=True,
        n_wn_vertex=12,
    ):
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

        diis = PulayDIIS(max_hist=5, beta_mix=0.35)
        diff = 1.0
        imp = vert = dual_res = None
        G_lat = None

        for it in range(max_outer):
            # 1. Track IV-B exact Lehmann ED impurity solve on DLR grid
            imp = self._solve_ed_impurity(Delta, mu)
            g_imp = imp["g"]
            Sig_imp = imp["Sig_imp"]
            Delta_eff = imp["Delta_fit"]
            vert = self._build_local_vertices(imp)

            # 2. Inner dual loop (identical Level-IV dual ladders, Moriya+SCSA, TU-SBE Eg)
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
                d_minus_e = Delta_eff[:, None, None, None] - self.eps_k[None, :, :, :]
                g_b = g_imp[:, None, None, None]
                G_dual = -(g_b ** 2) * d_minus_e / (1.0 + g_b * d_minus_e * (1.0 + g_b * Sig_dual))
                G_dual -= np.mean(G_dual, axis=(1, 2, 3), keepdims=True)

                dual_res = self._dual_self_energy_step(imp, vert, G_lat, G_dual)
                Sig_cand = self._herglotz_project(dual_res["Sig_dual"], g_imp)
                d_in = float(np.max(np.abs(Sig_cand - Sig_dual)))
                Sig_dual = 0.55 * Sig_dual + 0.45 * Sig_cand
                if d_in < 1e-6:
                    break

            # 3. Outer hybridization update using exact Delta_fit
            Sig_dual = self._herglotz_project(Sig_dual, g_imp)
            Sig_nl = Sig_dual / (1.0 + g_imp[:, None, None, None] * Sig_dual)
            G_lat = 1.0 / (dlr.iwn[:, None, None, None] + mu - self.eps_k[None, :, :, :] - Sig_imp[:, None, None, None] - Sig_nl)
            G_loc = np.mean(G_lat, axis=(1, 2, 3))
            Delta_new = self._symmetrize_Delta(Delta_eff + (1.0 / g_imp - 1.0 / G_loc))

            x_in = np.concatenate([Delta, Sig_dual.ravel()])
            x_out = np.concatenate([Delta_new, Sig_cand.ravel()])
            diff = float(np.max(np.abs(x_out - x_in)))

            if diff > 0.5:
                diis.reset()
                alpha_d = 0.30
                Delta = self._symmetrize_Delta((1.0 - alpha_d) * Delta + alpha_d * Delta_new)
                Sig_dual = self._herglotz_project((1.0 - alpha_d) * Sig_dual + alpha_d * Sig_cand, g_imp)
            else:
                x_mix = diis.update(x_in, x_out)
                Delta = self._symmetrize_Delta(x_mix[: dlr.r_F])
                Sig_dual = self._herglotz_project(x_mix[dlr.r_F :].reshape(Sig_dual.shape), g_imp)

            if diff < tol and it >= 3:
                break

        # Final physical observables
        w0_i = dlr.r_half
        w0 = dlr.wn[w0_i]
        Sig_nl = Sig_dual / (1.0 + imp["g"][:, None, None, None] * Sig_dual)
        Sig_tot = imp["Sig_imp"][:, None, None, None] + Sig_nl
        G_lat = 1.0 / (dlr.iwn[:, None, None, None] + mu - self.eps_k[None, :, :, :] - Sig_tot)
        n_lat = float(np.mean(dlr.density_from_G_iwn(G_lat)))

        Z_imp = float(1.0 / (1.0 - np.imag(imp["Sig_imp"][w0_i]) / w0))
        Z_k = 1.0 / (1.0 - np.imag(Sig_tot[w0_i]) / w0)
        q4, q2 = self.L // 4, self.L // 2
        Z_hot, Z_cold = float(Z_k[0, q2, q4]), float(Z_k[q4, q4, q4])
        C0 = float(np.max(np.abs(np.imag(Sig_dual[w0_i]))) * abs(np.imag(imp["g"][w0_i])))
        lam_Eg = self.compute_Eg_eigenvalue(imp, vert, dual_res, G_lat)

        # Optional exact 4-point Rank-R SVD vertex & multi-line Gamma_c^irr analysis on converged impurity
        n_div_lines = None
        svd_c_R2 = svd_c_R6 = svd_s_R6 = None
        eig_norm_c0 = None
        if extract_rankR_final and self.N_b <= 3:
            half_nw = n_wn_vertex // 2
            wn_v = (2 * np.arange(-half_nw, half_nw) + 1) * np.pi / self.beta
            chi_res = compute_chi_ph_static_ed(self.ed, imp, self.beta, wn_v)
            v_res = extract_rank_R_vertices(chi_res, self.U, rank=6)
            n_div_lines = v_res["n_div_lines"]
            eig_norm_c0 = float(v_res["eigs_norm_c"][0])
            svd_c_R2 = float(v_res["svd_c"]["err_by_rank"][2])
            svd_c_R6 = float(v_res["svd_c"]["err_by_rank"][6])
            svd_s_R6 = float(v_res["svd_s"]["err_by_rank"][6])

        return {
            "track": f"IV-B (ED, Nb={self.N_b})",
            "U": self.U,
            "T": self.T,
            "delta": self.delta,
            "L": self.L,
            "N_b": self.N_b,
            "mu": mu,
            "n_lat": n_lat,
            "n_imp": imp["n_imp"],
            "Z_imp": Z_imp,
            "Z_hot": Z_hot,
            "Z_cold": Z_cold,
            "D_imp": imp["D_imp"],
            "D_c_irr_R2": vert["D_c_irr"],
            "n_div_lines_ED": n_div_lines,
            "eig_norm_c0_ED": eig_norm_c0,
            "svd_c_err_R2": svd_c_R2,
            "svd_c_err_R6": svd_c_R6,
            "svd_s_err_R6": svd_s_R6,
            "chi_AFM": dual_res["chi_AFM"],
            "lam_s": dual_res["lam_s"],
            "max_dD_scsa": dual_res["max_dD_scsa"],
            "U_Eg_0": dual_res["U_Eg_0"],
            "lam_Eg": lam_Eg,
            "C0": C0,
            "fit_err": imp["fit_err"],
            "diff": diff,
            "iters": it + 1,
            "Delta": Delta,
            "Sig_dual": Sig_dual,
            "imp": imp,
        }
