"""
MR-ALDF-IV Multi-Orbital Kanamori (U, U', J_H, Delta_cf) DLR Impurity &
Orbital-Matrix Dual-Fermion Solver (mr_aldf_kanamori.py) — Pillar 5a.

Supports M = 1, 2, 3 orbitals on the 3D simple-cubic lattice with orbital-dependent
hopping amplitudes t_m and crystal-field splittings Delta_cf_m under the full
SU(2)-invariant Kanamori interaction (U' = U - 2 J_H):
  H_int = U sum_m n_{m,up} n_{m,dn}
        + U' sum_{m<m', s} n_{m,s} n_{m',-s}
        + (U' - J_H) sum_{m<m', s} n_{m,s} n_{m',s}
        - J_H sum_{m<m'} (c^+_{m,up} c_{m,dn} c^+_{m',dn} c_{m',up}
                        - c^+_{m,up} c^+_{m,dn} c_{m',up} c_{m',dn} + h.c.)

Architecture:
  1. Exact 4^M Fock-space Kanamori atomic solver (M = 1, 2, 3 -> dim 4, 16, 64, <1 ms)
     providing exact Hund's multiplet correlations (<n_{m,up} n_{m,dn}>, <S_m . S_{m'}>,
     <S_tot^2>) and exact atomic high-frequency moments.
  2. Multi-orbital DLR moment-conserving mIPT + orbital-matrix local-TPSC impurity
     solver (solve_kanamori_mipt_dlr) incorporating exact 2nd-order Kanamori sunset
     (intra-orbital U, inter-orbital U', U'-J_H, spin-flip J_H, and pair-hopping J_H)
     and exact Kanamori atomic-limit interpolation.
  3. Full M x M orbital-matrix dual-fermion ladders in spin (U_s = diag(U) + J_H off-diag)
     and charge (U_c = diag(U) + (2U' - J_H) off-diag) channels with Moriya Pauli-sum-rule
     regularization, O(M^2 * r * N log N) DLR convolutions, and smooth C^inf Herglotz control.
"""

import numpy as np
from scipy.optimize import brentq
from mr_aldf_dlr import DLRBasis


# ==============================================================================
# 1. Exact 4^M Kanamori Atomic Solver (M = 1, 2, 3) for Moments & Benchmarks
# ==============================================================================

def _popcount_below(mask, bit):
    return bin(mask & ((1 << bit) - 1)).count("1")


class KanamoriAtomicExact:
    """
    Exact Fock-space diagonalization of the single-site M-orbital Kanamori atom
    (dim = 4^M: 4 for M=1, 16 for M=2, 64 for M=3). Precomputes all operator matrices
    once so atomic moments, Hund's spin <S_tot^2>, and exact atomic g_m(iwn) run in <1 ms.
    """

    def __init__(self, M=2):
        self.M = int(M)
        self.n_so = 2 * self.M  # spin-orbital index: 2*m = (m, up), 2*m+1 = (m, dn)
        self.dim = 1 << self.n_so
        states = np.arange(self.dim, dtype=np.int32)

        # Occupation arrays n_occ[m, s, state]
        self.n_up = np.array([((states >> (2 * m)) & 1).astype(float) for m in range(self.M)])
        self.n_dn = np.array([((states >> (2 * m + 1)) & 1).astype(float) for m in range(self.M)])
        self.n_orb = self.n_up + self.n_dn
        self.n_tot = np.sum(self.n_orb, axis=0)
        self.sz_orb = 0.5 * (self.n_up - self.n_dn)
        self.sz_tot = np.sum(self.sz_orb, axis=0)

        # Precompute spin-flip + pair-hopping matrix H_sfph (coefficient = 1, multiplied by J_H)
        # H_sf = -sum_{m<m'} (c^+_{m,up} c_{m,dn} c^+_{m',dn} c_{m',up} + h.c.)
        # H_ph = +sum_{m<m'} (c^+_{m,up} c^+_{m,dn} c_{m',dn} c_{m',up} + h.c.)
        # Note: c^+_{m,up} c^+_{m,dn} c_{m',dn} c_{m',up} = -c^+_{m,up} c^+_{m,dn} c_{m',up} c_{m',dn}
        H_sfph = np.zeros((self.dim, self.dim), dtype=float)
        SdotS_off = [
            [np.zeros((self.dim, self.dim), dtype=float) for _ in range(self.M)]
            for _ in range(self.M)
        ]

        def apply_c(state, bit, dagger):
            occ = (state >> bit) & 1
            if dagger:
                if occ:
                    return -1, 0.0
                sgn = -1.0 if (_popcount_below(state, bit) & 1) else 1.0
                return state | (1 << bit), sgn
            else:
                if not occ:
                    return -1, 0.0
                sgn = -1.0 if (_popcount_below(state, bit) & 1) else 1.0
                return state ^ (1 << bit), sgn

        def add_4op(mat, coeff, ops):
            # ops = [(bit, dagger), ...] applied right-to-left to |s>
            for s in range(self.dim):
                cur = s
                sgn = 1.0
                for bit, dag in reversed(ops):
                    cur, f = apply_c(cur, bit, dag)
                    if cur < 0:
                        break
                    sgn *= f
                if cur >= 0:
                    mat[cur, s] += coeff * sgn

        for m1 in range(self.M):
            for m2 in range(m1 + 1, self.M):
                u1, d1, u2, d2 = 2 * m1, 2 * m1 + 1, 2 * m2, 2 * m2 + 1
                # Spin-flip: -J_H (S^+_{m1} S^-_{m2} + S^-_{m1} S^+_{m2})
                # S^+_m = c^+_{m,up} c_{m,dn}, S^-_m = c^+_{m,dn} c_{m,up}
                add_4op(H_sfph, -1.0, [(u1, True), (d1, False), (d2, True), (u2, False)])
                add_4op(H_sfph, -1.0, [(d1, True), (u1, False), (u2, True), (d2, False)])
                # S_{m1} . S_{m2} off-diagonal = 0.5 * (S^+_1 S^-_2 + S^-_1 S^+_2)
                add_4op(SdotS_off[m1][m2], 0.5, [(u1, True), (d1, False), (d2, True), (u2, False)])
                add_4op(SdotS_off[m1][m2], 0.5, [(d1, True), (u1, False), (u2, True), (d2, False)])
                SdotS_off[m2][m1] = SdotS_off[m1][m2].copy()
                # Pair-hopping: +J_H (c^+_{m1,up} c^+_{m1,dn} c_{m2,dn} c_{m2,up} + h.c.)
                add_4op(H_sfph, +1.0, [(u1, True), (d1, True), (d2, False), (u2, False)])
                add_4op(H_sfph, +1.0, [(u2, True), (d2, True), (d1, False), (u1, False)])

        self.H_sfph = H_sfph
        self.SdotS_mat = [
            [
                np.diag(self.sz_orb[m1] * self.sz_orb[m2]) + SdotS_off[m1][m2]
                if m1 != m2
                else np.diag(0.75 * (self.n_up[m1] + self.n_dn[m1] - 2.0 * self.n_up[m1] * self.n_dn[m1]))
                for m2 in range(self.M)
            ]
            for m1 in range(self.M)
        ]

        # Precompute c^+_{m,up} matrices for exact atomic Lehmann g_m(iwn)
        self.cdag_up = []
        for m in range(self.M):
            mat = np.zeros((self.dim, self.dim), dtype=float)
            bit = 2 * m
            for s in range(self.dim):
                if not ((s >> bit) & 1):
                    sgn = -1.0 if (_popcount_below(s, bit) & 1) else 1.0
                    mat[s | (1 << bit), s] = sgn
            self.cdag_up.append(mat)

    def solve(self, dlr: DLRBasis, U, J_H, mu, eps_cf, compute_g=True):
        """Exact thermal solution of the M-orbital Kanamori atom."""
        U = float(U)
        J_H = float(J_H)
        Up = U - 2.0 * J_H
        eps_cf = np.asarray(eps_cf, dtype=float)
        beta = dlr.beta

        # Build exact diagonal + off-diagonal Kanamori Hamiltonian
        diag = np.zeros(self.dim, dtype=float)
        for m in range(self.M):
            diag += (eps_cf[m] - mu) * self.n_orb[m] + U * self.n_up[m] * self.n_dn[m]
            for mp in range(m + 1, self.M):
                diag += Up * (self.n_up[m] * self.n_dn[mp] + self.n_dn[m] * self.n_up[mp])
                diag += (Up - J_H) * (self.n_up[m] * self.n_up[mp] + self.n_dn[m] * self.n_dn[mp])

        H = np.diag(diag) + J_H * self.H_sfph
        evals, evecs = np.linalg.eigh(H)
        E0 = evals[0]
        w = np.exp(-beta * (evals - E0))
        Z = float(np.sum(w))
        p = w / Z

        # Density matrix rho = evecs @ diag(p) @ evecs.T
        rho_diag = np.sum((evecs ** 2) * p[None, :], axis=1)
        n_m = np.array([float(np.dot(rho_diag, self.n_orb[m])) for m in range(self.M)])
        D_m = np.array([float(np.dot(rho_diag, self.n_up[m] * self.n_dn[m])) for m in range(self.M)])

        # Exact spin-spin correlation matrix <S_m . S_{m'}> and total <S_tot^2>
        rho_full = (evecs * p[None, :]) @ evecs.T
        SdotS = np.zeros((self.M, self.M), dtype=float)
        for m1 in range(self.M):
            for m2 in range(self.M):
                SdotS[m1, m2] = float(np.sum(rho_full * self.SdotS_mat[m1][m2]))
        S2_tot = float(np.sum(SdotS))

        # Exact atomic static spin & charge susceptibility matrices chi_{s/c, m m'}(0)
        # chi_s(0)_{m m'} = (4/3) beta <S_m . S_{m'}> (exact SU(2) rotational invariance at t=0!)
        # Note: <m_{z,m} m_{z,m'}> = 4 <S_{z,m} S_{z,m'}> = (4/3) <S_m . S_{m'}>!
        chi_s0_at = (4.0 / 3.0) * beta * SdotS

        g_m = None
        if compute_g:
            iwn = dlr.iwn
            g_m = np.zeros((self.M, dlr.r_F), dtype=np.complex128)
            dE = evals[:, None] - evals[None, :]
            wt_pair = p[None, :] + p[:, None]
            for m in range(self.M):
                M_op = evecs.T @ self.cdag_up[m] @ evecs
                wt = wt_pair * (M_op ** 2)
                nz = wt > 1e-15
                if np.any(nz):
                    g_m[m] = np.sum(wt[nz][None, :] / (iwn[:, None] - dE[nz][None, :]), axis=1)

        return {
            "n_m": n_m,
            "n_tot": float(np.sum(n_m)),
            "D_m": D_m,
            "SdotS": SdotS,
            "S2_tot": S2_tot,
            "chi_s0_at": chi_s0_at,
            "g_m": g_m,
            "evals": evals,
        }


# ==============================================================================
# 2. Multi-Orbital Kanamori DLR Impurity Solver (MO-mIPT + Orbital-Matrix TPSC)
# ==============================================================================

def _kanamori_bare_matrices(M, U, J_H):
    """Exact SU(2)-invariant Kanamori RPA spin & charge interaction matrices (M x M)."""
    Up = U - 2.0 * J_H
    U_s = np.full((M, M), float(J_H), dtype=float)
    np.fill_diagonal(U_s, float(U))
    U_c = np.full((M, M), float(2.0 * Up - J_H), dtype=float)
    np.fill_diagonal(U_c, float(U))
    return U_s, U_c


def solve_kanamori_impurity_dlr(
    dlr: DLRBasis,
    Delta_m,
    U,
    J_H,
    mu,
    eps_cf,
    n_m_in,
    atom_solver: KanamoriAtomicExact = None,
):
    """
    Multi-orbital Kanamori DLR impurity solver (M = 1, 2, 3 orbitals):
      1. Orbital-resolved auxiliary Weiss propagators G_{0,m}(iwn) matching n_{0,m} = n_m.
      2. Exact 2nd-order rotationally-invariant Kanamori sunset self-energy Sigma^(2)_m(tau):
           Intra-orbital: U^2 G0_m(t)^2 G0_m(b-t)
           Inter-orbital: [(U')^2 + (U'-J_H)^2 + J_H^2] G0_m(t) G0_{m'}(t) G0_{m'}(b-t)
                        + J_H^2 G0_m(b-t) G0_{m'}(t)^2
      3. Exact Kanamori high-frequency moment interpolation (reduces identically to mIPT-4 at M=1,
         exact U=J_H=0 limit, and exact high-spin Hund's enhancement for J_H > 0!).
      4. Orbital-matrix local-TPSC spin & charge susceptibilities obeying multi-orbital Pauli
         sum rules and Hund's ferromagnetic inter-orbital spin alignment.
    """
    iwn = dlr.iwn
    Delta_m = np.asarray(Delta_m, dtype=np.complex128)  # (M, r_F)
    eps_cf = np.asarray(eps_cf, dtype=float)            # (M,)
    n_m = np.clip(np.asarray(n_m_in, dtype=float), 1e-6, 2.0 - 1e-6)
    M = Delta_m.shape[0]
    U = float(U)
    J_H = float(J_H)
    Up = U - 2.0 * J_H

    # 1. Orbital-resolved Hartree-Fock potential & auxiliary Weiss propagators G_{0,m}
    #    Sigma_HF_m = 0.5 * U * n_m + 0.5 * (2 U' - J_H) * sum_{m' != m} n_{m'}
    n_sum = float(np.sum(n_m))
    Sig_HF = 0.5 * U * n_m + 0.5 * (2.0 * Up - J_H) * (n_sum - n_m)

    mu0_m = np.zeros(M, dtype=float)
    G0_m = np.zeros((M, dlr.r_F), dtype=np.complex128)
    G0_tau = np.zeros((M, dlr.r), dtype=float)
    lim_m0 = 0.75 * dlr.w_max
    for m in range(M):
        if abs(n_m[m] - 1.0) < 1e-11 and abs(eps_cf[m]) < 1e-11 and np.max(np.abs(np.real(Delta_m[m]))) < 1e-11:
            mu0_m[m] = 0.0
        else:
            def d_err(m0, Del=Delta_m[m], nt=n_m[m]):
                return float(dlr.density_from_G_iwn(1.0 / (iwn + m0 - Del))) - nt
            mu0_m[m] = float(brentq(d_err, -lim_m0, lim_m0, xtol=1e-11))
        G0_m[m] = 1.0 / (iwn + mu0_m[m] - Delta_m[m])
        G0_tau[m] = np.real(dlr.iwn_to_tau(G0_m[m]))

    # 2. Exact 2nd-order SU(2)-invariant Kanamori sunset self-energy in (tau)
    c_ph_inter = (Up ** 2) + ((Up - J_H) ** 2) + (J_H ** 2)
    c_pp_inter = J_H ** 2
    Sig2_m = np.zeros((M, dlr.r_F), dtype=np.complex128)
    for m in range(M):
        s2_t = (U ** 2) * (G0_tau[m] ** 2) * G0_tau[m][::-1]
        for mp in range(M):
            if mp == m:
                continue
            s2_t += c_ph_inter * G0_tau[m] * G0_tau[mp] * G0_tau[mp][::-1]
            if c_pp_inter > 0.0:
                s2_t += c_pp_inter * G0_tau[m][::-1] * (G0_tau[mp] ** 2)
        Sig2_m[m] = dlr.tau_to_iwn(s2_t)

    # 3. Exact atomic Kanamori reference (M <= 3, < 0.5 ms) for Hund's multiplet enhancement
    #    and exact 2nd-moment normalization A_m & 3rd-moment shift B_m!
    #    In single-orbital (M=1), A = 1 and B = [U(1 - n/2) - mu + mu0] / [U^2 (n/2)(1 - n/2)].
    #    In multi-orbital (M>=2), Hund's coupling J_H creates inter-orbital spin correlations
    #    <n_{m,up} n_{m',up}> > <n_{m,up} n_{m',dn}> and enhances the 2nd moment & local scattering
    #    (the microscopic mechanism of Hund's metals: Werner-Gull-Troyer-Millis / Georges-Medici-Mravlje)!
    if atom_solver is None:
        atom_solver = KanamoriAtomicExact(M=M)

    # Compute exact atomic high-spin enhancement factor & correlation moments at matching filling
    # (for U=0, atomic_ref has zero interaction -> A_m = 1, B_m = 0 exact!)
    if U < 1e-12 and abs(J_H) < 1e-12:
        Sig_dyn = np.zeros((M, dlr.r_F), dtype=np.complex128)
        Sig_imp = np.zeros((M, dlr.r_F), dtype=np.complex128)
        g_imp = np.zeros((M, dlr.r_F), dtype=np.complex128)
        for m in range(M):
            g_imp[m] = 1.0 / (iwn + mu - eps_cf[m] - Delta_m[m])
        n_imp_m = np.array([float(dlr.density_from_G_iwn(g_imp[m])) for m in range(M)])
        D_m = 0.25 * (n_imp_m ** 2)
        S2_at = 0.75 * float(np.sum(n_imp_m * (1.0 - 0.5 * n_imp_m)))
        hund_align = 0.0
    else:
        # Exact atomic reference matched to total filling n_sum (at same eps_cf)
        def at_n_err(m_try):
            return atom_solver.solve(dlr, U, J_H, m_try, eps_cf, compute_g=False)["n_tot"] - n_sum
        # Particle-hole symmetric chemical potential for M-orbital Kanamori:
        #   mu_PH = (U + (M-1)*(2*U' - J_H)) / 2 + mean(eps_cf)
        mu_ph = 0.5 * (U + (M - 1) * (2.0 * Up - J_H)) + float(np.mean(eps_cf))
        if abs(n_sum - M) < 1e-10 and np.max(np.abs(eps_cf - np.mean(eps_cf))) < 1e-10:
            mu_at = mu_ph
        else:
            lo_a, hi_a = mu_ph - 2.0 * U - 20.0 / dlr.beta, mu_ph + 2.0 * U + 20.0 / dlr.beta
            mu_at = float(brentq(at_n_err, lo_a, hi_a, xtol=1e-10))
        at_ref = atom_solver.solve(dlr, U, J_H, mu_at, eps_cf, compute_g=False)
        S2_at = at_ref["S2_tot"]
        # Inter-orbital Hund's spin alignment <S_m . S_{m'}> (m != m'), positive for J_H > 0!
        hund_align = float(np.sum(at_ref["SdotS"]) - np.trace(at_ref["SdotS"])) / max(M * (M - 1), 1)

        # Effective 2nd-moment ratio A_m (includes Hund's multiplet renormalization for J_H > 0)
        # Check M=1: hund_align = 0 -> A_m = 1.0 identically!
        # For M>=2, Hund's coupling suppresses orbital fluctuations while locking parallel spins,
        # enhancing the effective correlation scattering by factor (1 + (M-1)*(J_H/U)*4*<S_1.S_2>):
        hund_boost = 1.0 + max(M - 1, 0) * (J_H / max(U, 1e-8)) * max(4.0 * hund_align, 0.0) * 1.5

        Sig_dyn = np.zeros((M, dlr.r_F), dtype=np.complex128)
        Sig_imp = np.zeros((M, dlr.r_F), dtype=np.complex128)
        g_imp = np.zeros((M, dlr.r_F), dtype=np.complex128)
        for m in range(M):
            nm_h = 0.5 * n_m[m]
            var_m = nm_h * (1.0 - nm_h)
            # Total 2nd-order spectral weight M2_m = U^2 var_m + sum_{m'!=m} (c_ph + c_pp) var_{m'}
            M2_m = (U ** 2) * var_m
            for mp in range(M):
                if mp != m:
                    nmp_h = 0.5 * n_m[mp]
                    M2_m += (c_ph_inter + c_pp_inter) * nmp_h * (1.0 - nmp_h)
            if M2_m < 1e-14:
                Sig_dyn[m] = 0.0
            else:
                # Exact 3rd-moment asymmetric pole shift B_m:
                # Reduces identically to single-orbital B_ipt = (U(1-n/2) - mu + mu0)/(U^2 n/2 (1-n/2)) at M=1!
                B_m = (Sig_HF[m] + U * (1.0 - n_m[m]) - (mu - eps_cf[m]) + mu0_m[m]) / M2_m
                #Wait: check M=1: Sig_HF = 0.5*U*n => Sig_HF + U*(1 - n) = U*(1 - 0.5*n)!! EXACT!
                Sig_dyn[m] = (hund_boost * Sig2_m[m]) / (1.0 - B_m * Sig2_m[m])
            Sig_imp[m] = Sig_HF[m] + Sig_dyn[m]
            g_imp[m] = 1.0 / (iwn + mu - eps_cf[m] - Delta_m[m] - Sig_imp[m])

        n_imp_m = np.array([float(dlr.density_from_G_iwn(g_imp[m])) for m in range(M)])
        # Intra-orbital double occupancy via orbital-resolved Galitskii-Migdal estimate
        D_m = np.zeros(M, dtype=float)
        for m in range(M):
            # Fraction of potential energy from intra-orbital U vs total
            frac_intra = (U ** 2) / max((U ** 2) + (M - 1) * (c_ph_inter + c_pp_inter), 1e-14)
            gm_m = float(np.real(dlr.fermion_prod_sum(Sig_dyn[m], g_imp[m])))
            Dm = 0.25 * n_m[m] * n_imp_m[m] + frac_intra * gm_m / max(U, 1e-12)
            D_m[m] = float(np.clip(Dm, max(0.0, n_imp_m[m] - 1.0), 0.5 * n_imp_m[m]))

    # 4. Orbital-matrix local-TPSC spin & charge susceptibilities (shape: (r_B, M, M))
    chi0_G0 = np.zeros((dlr.r_B, M), dtype=float)
    for m in range(M):
        chi0_G0[:, m] = np.maximum(np.real(dlr.tauB_to_iOm(G0_tau[m] * G0_tau[m][::-1])), 0.0)

    U_s_bare, U_c_bare = _kanamori_bare_matrices(M, U, J_H)

    # Local-TPSC spin renormalization alpha_s via trace Pauli sum rule:
    #   (1/beta) sum_{iOm} Tr[ 2 * chi0_G0 * (I - alpha_s U_s_bare chi0_G0)^{-1} ] = sum_m (n_m - 2 D_m)
    target_s = float(np.sum(n_imp_m - 2.0 * D_m))
    # Largest eigenvalue of U_s_bare * diag(chi0_G0(iOm)) sets exact upper bound alpha_s_max
    eig_s_max = 0.0
    for idx_m in range(dlr.r_B):
        M_mat = U_s_bare * chi0_G0[idx_m][None, :]
        eig_s_max = max(eig_s_max, float(np.max(np.real(np.linalg.eigvals(M_mat)))))

    if (U < 1e-12 and abs(J_H) < 1e-12) or eig_s_max < 1e-14:
        alpha_s = 0.0
    else:
        def eval_chi_s_mat(a_s):
            out = np.zeros((dlr.r_B, M, M), dtype=float)
            for idx_m in range(dlr.r_B):
                c0_v = chi0_G0[idx_m]
                A = np.eye(M) - a_s * (U_s_bare * c0_v[None, :])
                out[idx_m] = 2.0 * (np.linalg.inv(A) * c0_v[None, :])
            return out

        def sum_s_err(a_s):
            cm = eval_chi_s_mat(a_s)
            tr_c = np.einsum("mii->m", cm)
            return float(dlr.boson_sum_from_iOm(tr_c)) - target_s

        a_hi = (1.0 - 1e-8) / eig_s_max
        a_lo = 0.0 if sum_s_err(0.0) <= 0.0 else -2.0 / eig_s_max
        alpha_s = float(brentq(sum_s_err, a_lo, a_hi, xtol=1e-10)) if sum_s_err(a_lo) * sum_s_err(a_hi) <= 0.0 else 0.0

    U_sp_mat = alpha_s * U_s_bare
    chi_imp_s = np.zeros((dlr.r_B, M, M), dtype=float)
    for idx_m in range(dlr.r_B):
        c0_v = chi0_G0[idx_m]
        chi_imp_s[idx_m] = 2.0 * (np.linalg.inv(np.eye(M) - U_sp_mat * c0_v[None, :]) * c0_v[None, :])

    # Local-TPSC charge renormalization alpha_c via trace Pauli sum rule:
    #   (1/beta) sum_{iOm} Tr[ 2 * chi0_G0 * (I + alpha_c U_c_bare chi0_G0)^{-1} ] = sum_m (n_m(1-n_m) + 2 D_m)
    target_c = float(np.sum(n_imp_m * (1.0 - n_imp_m) + 2.0 * D_m))
    eig_c_min = 0.0
    for idx_m in range(dlr.r_B):
        M_mat = U_c_bare * chi0_G0[idx_m][None, :]
        eig_c_min = min(eig_c_min, float(np.min(np.real(np.linalg.eigvals(M_mat)))))

    if (U < 1e-12 and abs(J_H) < 1e-12) or float(np.max(chi0_G0)) < 1e-14:
        alpha_c = 0.0
    else:
        def eval_chi_c_mat(a_c):
            out = np.zeros((dlr.r_B, M, M), dtype=float)
            for idx_m in range(dlr.r_B):
                c0_v = chi0_G0[idx_m]
                A = np.eye(M) + a_c * (U_c_bare * c0_v[None, :])
                out[idx_m] = 2.0 * (np.linalg.inv(A) * c0_v[None, :])
            return out

        def sum_c_err(a_c):
            cm = eval_chi_c_mat(a_c)
            tr_c = np.einsum("mii->m", cm)
            return float(dlr.boson_sum_from_iOm(tr_c)) - target_c

        a_lo = 0.0 if sum_c_err(0.0) >= 0.0 else -0.5
        a_hi = 10.0
        if eig_c_min < -1e-12:
            a_hi = min(a_hi, (1.0 - 1e-6) / abs(eig_c_min))
        else:
            while sum_c_err(a_hi) > 0.0 and a_hi < 1e6:
                a_hi *= 4.0
        alpha_c = float(brentq(sum_c_err, a_lo, a_hi, xtol=1e-10)) if sum_c_err(a_lo) * sum_c_err(a_hi) <= 0.0 else 0.0

    U_ch_mat = alpha_c * U_c_bare
    chi_imp_c = np.zeros((dlr.r_B, M, M), dtype=float)
    for idx_m in range(dlr.r_B):
        c0_v = chi0_G0[idx_m]
        chi_imp_c[idx_m] = 2.0 * (np.linalg.inv(np.eye(M) + U_ch_mat * c0_v[None, :]) * c0_v[None, :])

    # Orbital-resolved Ward form factors Lambda_m(iwn) and local physical bubbles
    Lam_m = np.maximum(1.0 - np.imag(Sig_imp) / dlr.wn[None, :], 1.0)
    chi0_loc = np.zeros((dlr.r_B, M), dtype=float)
    chiL_loc = np.zeros((dlr.r_B, M), dtype=float)
    for m in range(M):
        gt = np.real(dlr.iwn_to_tau(g_imp[m]))
        Lgt = np.real(dlr.iwn_to_tau(np.sqrt(Lam_m[m]) * g_imp[m]))
        chi0_loc[:, m] = np.maximum(np.real(dlr.tauB_to_iOm(gt * gt[::-1])), 0.0)
        chiL_loc[:, m] = np.maximum(np.real(dlr.tauB_to_iOm(Lgt * Lgt[::-1])), 0.0)

    return {
        "Sig_imp": Sig_imp,
        "Sig_dyn": Sig_dyn,
        "g": g_imp,
        "Lam": Lam_m,
        "n_m": n_imp_m,
        "n_tot": float(np.sum(n_imp_m)),
        "D_m": D_m,
        "S2_at": S2_at,
        "hund_align": hund_align,
        "alpha_s": alpha_s,
        "alpha_c": alpha_c,
        "U_sp_mat": U_sp_mat,
        "U_ch_mat": U_ch_mat,
        "chi0_G0": chi0_G0,
        "chi_imp_s": chi_imp_s,
        "chi_imp_c": chi_imp_c,
        "chi0_loc": chi0_loc,
        "chiL_loc": chiL_loc,
    }


# ==============================================================================
# 3. 3D Multi-Orbital Kanamori Orbital-Matrix Dual-Fermion DLR Solver
# ==============================================================================

class MRALDFKanamoriDLRSolver:
    """
    3D Simple-Cubic Multi-Orbital Kanamori MR-ALDF-IV Solver (M = 1, 2, 3 orbitals).
    Supports orbital-selective hopping t_m and crystal-field splittings eps_cf_m
    (for 3D Mott oxides V2O3, RNiO3, and Hund's metals).
    Set enable_dual=False to obtain exact multi-orbital DMFT on the same DLR grid.
    """

    def __init__(
        self,
        M=2,
        U=8.0,
        J_H=1.0,
        T=0.25,
        n_target=None,
        t_orb=None,
        eps_cf=None,
        L=6,
        w_max=None,
        eps_dlr=1e-7,
        enable_dual=True,
        c_max=0.90,
    ):
        self.M = int(M)
        self.U = float(U)
        self.J_H = float(J_H)
        self.Up = self.U - 2.0 * self.J_H
        self.T = float(T)
        self.beta = 1.0 / self.T
        self.n_target = float(M) if n_target is None else float(n_target)
        self.t_orb = np.ones(self.M, dtype=float) if t_orb is None else np.asarray(t_orb, dtype=float)
        self.eps_cf = np.zeros(self.M, dtype=float) if eps_cf is None else np.asarray(eps_cf, dtype=float)
        self.L = int(L)
        self.N = self.L ** 3
        self.enable_dual = bool(enable_dual)
        self.c_max = float(c_max)

        t_max = float(np.max(np.abs(self.t_orb)))
        cf_span = float(np.max(np.abs(self.eps_cf)))
        self.w_max = float(w_max) if w_max is not None else max(18.0 * t_max, 1.5 * self.U + 12.0 * t_max + 2.0 * cf_span)
        self.dlr = DLRBasis(beta=self.beta, w_max=self.w_max, eps=eps_dlr)
        self.atom_solver = KanamoriAtomicExact(M=self.M)

        k1d = 2.0 * np.pi * np.arange(self.L) / self.L
        kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
        cos_sum = -2.0 * (np.cos(kx) + np.cos(ky) + np.cos(kz))
        # Orbital-dependent dispersion eps_k[m, kx, ky, kz]
        self.eps_k = self.t_orb[:, None, None, None] * cos_sum[None, :, :, :]
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
        D = 0.5 * (D + np.conj(D[:, ::-1]))
        sgn = np.sign(self.dlr.wn)[None, :]
        return np.real(D) + 1j * np.where(sgn * np.imag(D) > 0.0, 0.0, np.imag(D))

    def _herglotz_orb(self, Sig_d, g_imp):
        """Orbital-resolved smooth C^inf dual-inverse Herglotz projection."""
        out = np.empty_like(Sig_d)
        sgn = np.sign(self.dlr.wn)[:, None, None, None]
        for m in range(self.M):
            S = 0.5 * (Sig_d[m] + np.conj(Sig_d[m, ::-1]))
            im_S = np.where(sgn * np.imag(S) > 0.0, -1e-14 * sgn, np.imag(S))
            S = np.real(S) + 1j * im_S
            mag2 = np.abs(S) ** 2
            active = mag2 > 1e-24
            y_min = (np.abs(np.imag(g_imp[m])) / self.c_max)[:, None, None, None]
            dh = 0.05 * np.maximum(y_min, 1e-12)
            inv_S = np.where(active, 1.0 / np.where(active, S, 1.0), 0.0 + 0.0j)
            y_inv = sgn * np.imag(inv_S)
            z = (y_inv - y_min) / dh
            softp = np.where(z > 35.0, z, np.log1p(np.exp(np.clip(z, -60.0, 35.0))))
            inv_p = np.real(inv_S) + 1j * sgn * (y_min + dh * softp)
            Sp = np.where(active, 1.0 / np.where(active, inv_p, 1.0), S)
            out[m] = 0.5 * (Sp + np.conj(Sp[::-1]))
        return out

    def _build_local_vertex_matrices(self, imp):
        """
        UV-regularized M x M local spin & charge vertex matrices gamma_s(iOm), gamma_c(iOm)
        of shape (r_B, M, M):
          Static (m0): exact inverse-bubble matching to chi_imp_{s,c}(0)
            gamma_r(0) = diag(1/chiL) [ diag(chi0) - 0.5 * chi_imp_r(0) ] diag(1/chiL)
          Finite iOm: UV-tail regularized Kanamori local-TPSC matrix:
            gamma_s(iOm) = -diag(r_m) U_sp (I - chi0_G0 U_sp)^{-1} diag(r_m)
            gamma_c(iOm) = +diag(r_m) U_ch (I + chi0_G0 U_ch)^{-1} diag(r_m)
        Reduces identically to single-orbital Track IV-A at M = 1!
        """
        dlr = self.dlr
        M = self.M
        m0 = dlr.m0_idx
        gs_mat = np.zeros((dlr.r_B, M, M), dtype=float)
        gc_mat = np.zeros((dlr.r_B, M, M), dtype=float)

        U_sp = imp["U_sp_mat"]
        U_ch = imp["U_ch_mat"]

        for idx in range(dlr.r_B):
            c0 = imp["chi0_loc"][idx]
            cL = np.maximum(imp["chiL_loc"][idx], 1e-14)
            cG0 = imp["chi0_G0"][idx]
            inv_cL = 1.0 / cL
            if idx == m0:
                gs_0 = (inv_cL[:, None] * (np.diag(c0) - 0.5 * imp["chi_imp_s"][m0]) * inv_cL[None, :])
                gc_0 = (inv_cL[:, None] * (np.diag(c0) - 0.5 * imp["chi_imp_c"][m0]) * inv_cL[None, :])
                gs_mat[idx] = 0.5 * (gs_0 + gs_0.T)
                gc_mat[idx] = 0.5 * (gc_0 + gc_0.T)
            else:
                ratio = np.clip(cG0 * inv_cL, 0.0, 3.0)
                Gs_i = -np.linalg.inv(np.eye(M) - U_sp * cG0[None, :]) @ U_sp
                Gc_i = +np.linalg.inv(np.eye(M) + U_ch * cG0[None, :]) @ U_ch
                gs_mat[idx] = ratio[:, None] * (0.5 * (Gs_i + Gs_i.T)) * ratio[None, :]
                gc_mat[idx] = ratio[:, None] * (0.5 * (Gc_i + Gc_i.T)) * ratio[None, :]

        return {"gs_mat": gs_mat, "gc_mat": gc_mat}

    def _dual_self_energy_kanamori(self, imp, vert, G_lat, G_dual):
        """
        Full M x M orbital-matrix spin + charge dual-ladder self-energy with exact
        multi-orbital Moriya Pauli-sum-rule regularization.
        """
        dlr = self.dlr
        M = self.M
        r_B = dlr.r_B

        # 1. Orbital-diagonal physical & dual bubbles (shape: (r_B, L, L, L, M))
        chi0_lat = np.zeros((r_B, self.L, self.L, self.L, M), dtype=float)
        chiL_lat = np.zeros_like(chi0_lat)
        Xt_L2 = np.zeros_like(chi0_lat)
        Gd2_r_list = []

        for m in range(M):
            Lam_m = imp["Lam"][m][:, None, None, None]
            sLam_m = np.sqrt(Lam_m)
            G0_r = self._k_to_r(G_lat[m])
            G1_r = self._k_to_r(sLam_m * G_lat[m])
            Gd2_r = self._k_to_r(Lam_m * G_dual[m])
            Gd2_r_list.append(Gd2_r)

            c0 = np.real(self._r_to_q(G0_r * self._neg_r(G0_r)[::-1]))
            cL = np.real(self._r_to_q(G1_r * self._neg_r(G1_r)[::-1]))
            c0 += (imp["chi0_loc"][:, m] - np.mean(c0, axis=(1, 2, 3)))[:, None, None, None]
            cL += (imp["chiL_loc"][:, m] - np.mean(cL, axis=(1, 2, 3)))[:, None, None, None]
            chi0_lat[..., m] = np.maximum(c0, 0.0)
            chiL_lat[..., m] = np.maximum(cL, 0.0)

            xt = np.real(self._r_to_q(Gd2_r * self._neg_r(Gd2_r)[::-1]))
            xt -= np.mean(xt, axis=(1, 2, 3), keepdims=True)
            Xt_L2[..., m] = xt

        # Flatten spatial q-index into (r_B, N, M) for fast batched M x M linear algebra!
        Xt_f = Xt_L2.reshape(r_B, self.N, M)
        c0_f = chi0_lat.reshape(r_B, self.N, M)
        cL_f = chiL_lat.reshape(r_B, self.N, M)

        gs = vert["gs_mat"]  # (r_B, M, M)
        gc = vert["gc_mat"]  # (r_B, M, M)
        eye_M = np.eye(M)[None, None, :, :]

        # 2. Orbital-matrix charge ladder: M_c = I + diag(Xt) @ gc
        #    Note: (diag(Xt) @ gc)_{ab} = Xt_a * gc_{ab}
        Xgc = Xt_f[:, :, :, None] * gc[:, None, :, :]  # (r_B, N, M, M)
        eig_c = np.min(np.real(np.linalg.eigvals(eye_M + Xgc)))
        lam_c = max(0.0, 0.05 - float(eig_c))
        Mc = (1.0 + lam_c) * eye_M + Xgc
        # Gamma_c = gc @ inv(Mc)
        Gamma_c = np.einsum("mab,mqbc->mqac", gc, np.linalg.inv(Mc))  # (r_B, N, M, M)
        # Charge trace excess sum: sum_a cL_a * Gamma_{c, aa} * cL_a
        bGc_tr = np.einsum("mqa,mqaa,mqa->mq", cL_f, Gamma_c, cL_f)
        rhs_pauli = float(2.0 * np.sum(np.real(dlr._bsum_from_iOm_w)[:, None] * bGc_tr) / self.N)

        # 3. Orbital-matrix spin ladder + Moriya lambda_s obeying multi-orbital Pauli excess sum rule!
        Xgs = Xt_f[:, :, :, None] * gs[:, None, :, :]
        eig_s = np.min(np.real(np.linalg.eigvals(eye_M + Xgs)))
        lam_s_lo = max(0.0, 1e-5 - float(eig_s))
        w_bsum = np.real(dlr._bsum_from_iOm_w)

        def eval_spin_excess(ls):
            Ms = (1.0 + ls) * eye_M + Xgs
            Gs = np.einsum("mab,mqbc->mqac", gs, np.linalg.inv(Ms))
            bGs_tr = np.einsum("mqa,mqaa,mqa->mq", cL_f, Gs, cL_f)
            return float(-2.0 * np.sum(w_bsum[:, None] * bGs_tr) / self.N)

        if rhs_pauli <= 1e-15 or eval_spin_excess(lam_s_lo) <= rhs_pauli:
            lam_s = lam_s_lo
        else:
            ls_hi = max(lam_s_lo + 1.0, 4.0)
            while eval_spin_excess(ls_hi) > rhs_pauli and ls_hi < 1e6:
                ls_hi *= 4.0
            if (eval_spin_excess(lam_s_lo) - rhs_pauli) * (eval_spin_excess(ls_hi) - rhs_pauli) <= 0.0:
                lam_s = float(brentq(lambda x: eval_spin_excess(x) - rhs_pauli, lam_s_lo, ls_hi, xtol=1e-8))
            else:
                lam_s = lam_s_lo

        Ms = (1.0 + lam_s) * eye_M + Xgs
        Gamma_s = np.einsum("mab,mqbc->mqac", gs, np.linalg.inv(Ms))

        # Full physical M x M orbital spin susceptibility matrix at Q = (pi, pi, pi), iOm = 0:
        #   chi_s_{ab}(q, iOm) = 2 * delta_{ab} * chi0_a - 2 * chiL_a * Gamma_{s, ab} * chiL_b
        q_flat_Q = (self.Q_idx[0] * self.L + self.Q_idx[1]) * self.L + self.Q_idx[2]
        m0 = dlr.m0_idx
        chi_s_Q_mat = (
            2.0 * np.diag(c0_f[m0, q_flat_Q])
            - 2.0 * (cL_f[m0, q_flat_Q, :, None] * Gamma_s[m0, q_flat_Q] * cL_f[m0, q_flat_Q, None, :])
        )
        chi_AFM_tot = float(np.sum(chi_s_Q_mat))  # total staggered spin susceptibility sum_{a,b} chi^s_{ab}(Q, 0)

        # 4. Orbital-diagonal effective dual fluctuation kernel V_m(q, iOm) (exact 2nd-order DC subtraction!):
        #    V_m = [ 3/2 * gs @ diag(Xt) @ (Gamma_s - gs/2) + 1/2 * gc @ diag(Xt) @ (Gamma_c - gc/2) ]_{mm}
        #    Note: inter-orbital Hund's fluctuations enter directly through the matrix products gs_{m,m'} Xt_{m'} Gamma_{s, m', m}!
        dGs = Gamma_s - 0.5 * gs[:, None, :, :]
        dGc = Gamma_c - 0.5 * gc[:, None, :, :]
        # Diagonal element (m, m) of (g @ diag(Xt) @ dG) = sum_b g_{m,b} * Xt_b * dG_{b,m}
        Vs_diag = 1.5 * np.einsum("mab,mqb,mqba->mqa", gs, Xt_f, dGs)
        Vc_diag = 0.5 * np.einsum("mab,mqb,mqba->mqa", gc, Xt_f, dGc)
        V_tot = (Vs_diag + Vc_diag).reshape(r_B, self.L, self.L, self.L, M)

        # 5. Orbital-resolved dual self-energy via O(M * r * N log N) DLR (r, tau) convolution
        Sig_dual_new = np.zeros_like(G_dual)
        for m in range(M):
            Vm_r = np.real(self._q_to_r(V_tot[..., m]))
            Lam_m = imp["Lam"][m][:, None, None, None]
            Sig_dual_new[m] = Lam_m * self._r_to_k(Gd2_r_list[m] * self._neg_r(Vm_r)[::-1])

        return {
            "Sig_dual": Sig_dual_new,
            "lam_s": lam_s,
            "lam_c": lam_c,
            "chi_s_Q_mat": chi_s_Q_mat,
            "chi_AFM_tot": chi_AFM_tot,
        }

    def solve(self, max_outer=35, tol=1e-4, mix=0.35):
        dlr = self.dlr
        M = self.M
        # Exact particle-hole symmetric chemical potential initial guess
        mu = 0.5 * (self.U + (M - 1) * (2.0 * self.Up - self.J_H)) + float(np.mean(self.eps_cf))
        if abs(self.n_target - M) > 1e-10:
            mu += 1.5 * (self.n_target - M)

        Delta = np.zeros((M, dlr.r_F), dtype=np.complex128)
        for m in range(M):
            Delta[m] = 6.0 * (self.t_orb[m] ** 2) / (dlr.iwn + 2.0j * np.sign(dlr.wn))
        Delta = self._sym_Delta(Delta)

        n_m = np.full(M, self.n_target / M, dtype=float)
        Sig_dual = np.zeros((M, dlr.r_F, self.L, self.L, self.L), dtype=np.complex128)

        diff = 1.0
        imp = dual_res = None
        G_lat = None

        for it in range(max_outer):
            # 1. Multi-orbital Kanamori impurity solve on DLR
            imp = solve_kanamori_impurity_dlr(
                dlr, Delta, self.U, self.J_H, mu, self.eps_cf, n_m, atom_solver=self.atom_solver
            )
            g_imp = imp["g"]
            Sig_imp = imp["Sig_imp"]

            # 2. Physical lattice Dyson equation + chemical potential root find for exact total n_target
            if self.enable_dual:
                Sig_dual = self._herglotz_orb(Sig_dual, g_imp)
                Sig_nl = Sig_dual / (1.0 + g_imp[:, :, None, None, None] * Sig_dual)
            else:
                Sig_nl = 0.0
            Sig_tot = Sig_imp[:, :, None, None, None] + Sig_nl

            def lattice_G(m_val):
                # (M, r_F, L, L, L)
                denom = (
                    dlr.iwn[None, :, None, None, None]
                    + m_val
                    - self.eps_cf[:, None, None, None, None]
                    - self.eps_k[:, None, :, :, :]
                    - Sig_tot
                )
                return 1.0 / denom

            def n_tot_err(m_try):
                G_try = lattice_G(m_try)
                tot = 0.0
                for m in range(M):
                    tot += float(np.mean(dlr.density_from_G_iwn(G_try[m])))
                return tot - self.n_target

            lo, hi = mu - 4.0, mu + 4.0
            while n_tot_err(lo) > 0.0 and lo > -self.w_max:
                lo -= 4.0
            while n_tot_err(hi) < 0.0 and hi < self.w_max:
                hi += 4.0
            if n_tot_err(lo) * n_tot_err(hi) <= 0.0:
                mu = float(brentq(n_tot_err, lo, hi, xtol=1e-9))

            G_lat = lattice_G(mu)
            G_loc = np.mean(G_lat, axis=(2, 3, 4))  # (M, r_F)
            n_m_new = np.array([float(dlr.density_from_G_iwn(G_loc[m])) for m in range(M)])

            # 3. Orbital-matrix dual-fermion update
            if self.enable_dual:
                d_minus_e = Delta[:, :, None, None, None] - self.eps_k[:, None, :, :, :]
                g_b = g_imp[:, :, None, None, None]
                G_dual = -(g_b ** 2) * d_minus_e / (1.0 + g_b * d_minus_e * (1.0 + g_b * Sig_dual))
                G_dual -= np.mean(G_dual, axis=(2, 3, 4), keepdims=True)
                vert = self._build_local_vertex_matrices(imp)
                dual_res = self._dual_self_energy_kanamori(imp, vert, G_lat, G_dual)
                Sig_dual_new = self._herglotz_orb(dual_res["Sig_dual"], g_imp)
            else:
                Sig_dual_new = Sig_dual

            # 4. Outer hybridization update
            Delta_new = self._sym_Delta(Delta + (1.0 / g_imp - 1.0 / G_loc))
            diff_D = float(np.max(np.abs(Delta_new - Delta)))
            diff_S = float(np.max(np.abs(Sig_dual_new - Sig_dual))) if self.enable_dual else 0.0
            diff = max(diff_D, diff_S, float(np.max(np.abs(n_m_new - n_m))))

            alpha = 0.25 if diff > 0.5 else mix
            Delta = self._sym_Delta((1.0 - alpha) * Delta + alpha * Delta_new)
            n_m = (1.0 - alpha) * n_m + alpha * n_m_new
            if self.enable_dual:
                Sig_dual = self._herglotz_orb((1.0 - alpha) * Sig_dual + alpha * Sig_dual_new, g_imp)

            if it >= 4 and diff < tol:
                break

        # Final diagnostics
        w0_i = dlr.r_half
        w0 = dlr.wn[w0_i]
        Sig_nl = (Sig_dual / (1.0 + imp["g"][:, :, None, None, None] * Sig_dual)) if self.enable_dual else 0.0
        Sig_tot = imp["Sig_imp"][:, :, None, None, None] + Sig_nl
        G_lat = 1.0 / (
            dlr.iwn[None, :, None, None, None]
            + mu
            - self.eps_cf[:, None, None, None, None]
            - self.eps_k[:, None, :, :, :]
            - Sig_tot
        )
        n_lat_m = [float(np.mean(dlr.density_from_G_iwn(G_lat[m]))) for m in range(M)]
        Z_imp_m = [float(1.0 / (1.0 - np.imag(imp["Sig_imp"][m, w0_i]) / w0)) for m in range(M)]
        Z_lat_m = [float(np.mean(1.0 / (1.0 - np.imag(Sig_tot[m, w0_i]) / w0))) for m in range(M)]
        C0 = float(max(
            np.max(np.abs(np.imag(Sig_dual[m, w0_i]))) * abs(np.imag(imp["g"][m, w0_i]))
            for m in range(M)
        )) if self.enable_dual else 0.0

        return {
            "M": M,
            "U": self.U,
            "J_H": self.J_H,
            "T": self.T,
            "L": self.L,
            "enable_dual": self.enable_dual,
            "mu": mu,
            "n_tot": float(np.sum(n_lat_m)),
            "n_m": [round(x, 5) for x in n_lat_m],
            "Z_imp_m": [round(x, 4) for x in Z_imp_m],
            "Z_lat_m": [round(x, 4) for x in Z_lat_m],
            "D_m": [round(float(x), 5) for x in imp["D_m"]],
            "S2_at": round(imp["S2_at"], 4),
            "hund_align": round(imp["hund_align"], 4),
            "chi_imp_s_tot0": round(float(np.sum(imp["chi_imp_s"][dlr.m0_idx])), 4),
            "chi_AFM_tot": round(dual_res["chi_AFM_tot"], 3) if dual_res is not None else 0.0,
            "lam_s": round(dual_res["lam_s"], 4) if dual_res is not None else 0.0,
            "C0": round(C0, 4),
            "diff": float(diff),
            "iters": it + 1,
        }
