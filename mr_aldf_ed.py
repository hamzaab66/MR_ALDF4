"""
MR-ALDF-IV Track IV-B Deterministic Symmetry-Sector Lehmann Exact Diagonalization (ED)
Anderson Impurity Solver (mr_aldf_ed.py) — Pillar 3a.

Exact, zero-noise, QMC-free solver for the Single-Impurity Anderson Model (SIAM):
  H = -mu (n_{0,up} + n_{0,dn}) + U n_{0,up} n_{0,dn}
      + sum_{l=1}^{N_b} eps_l (n_{l,up} + n_{l,dn})
      + sum_{l=1}^{N_b} sum_s V_l (c_{0,s}^dag c_{l,s} + c_{l,s}^dag c_{0,s})

Features:
  1. Exact (N_up, N_dn) Fock-sector block diagonalization (N_b = 4 => max dim 100;
     N_b = 5 => max dim 400) exploiting exact spin-flip symmetry N_up <-> N_dn.
  2. Vectorized finite-temperature Lehmann evaluation of:
       - Single-particle impurity Green's function g_imp(iwn) and Sigma_imp(iwn)
       - Exact thermal density n_imp and double occupancy D_imp = <n_{0,up} n_{0,dn}>
       - Exact local spin chi_s(iOm), charge chi_c(iOm), and pairing chi_pp(iOm)
         susceptibilities obeying exact Pauli sum rules to machine precision!
  3. Smooth DLR-weighted nonlinear least-squares bath discretization fit
     Delta_fit(iwn) = sum_l V_l^2 / (iwn - eps_l) with exact particle-hole symmetry
     enforcement at half-filling and warm-start continuation across DMFT/DF loops.
"""

import numpy as np
from itertools import combinations
from scipy.optimize import least_squares
from mr_aldf_dlr import DLRBasis


def _spin_basis(n_orb, n_part):
    """Sorted list of integer bitmasks for n_part fermions in n_orb orbitals (bit 0 = impurity)."""
    masks = []
    for comb in combinations(range(n_orb), n_part):
        m = 0
        for site in comb:
            m |= (1 << site)
        masks.append(m)
    masks = np.array(masks, dtype=np.int32)
    idx_map = {int(m): i for i, m in enumerate(masks)}
    return masks, idx_map


def _popcount_below(mask, site):
    """Number of occupied bits strictly below `site` in `mask` (Jordan-Wigner sign)."""
    return bin(mask & ((1 << site) - 1)).count("1")


class EDAndersonSolver:
    """
    Reusable symmetry-sector Lehmann ED solver for fixed number of bath sites N_b.
    Precomputes all Fock-sector bitmasks, single-spin hopping indices, and impurity
    creation/annihilation operator maps once at initialization for millisecond solves.
    """

    def __init__(self, N_b=4):
        self.N_b = int(N_b)
        self.N_s = 1 + self.N_b  # orbital 0 = impurity, 1..N_b = bath

        # Precompute single-spin bases for n = 0 .. N_s
        self.masks = {}
        self.idx_map = {}
        self.dims = {}
        for n in range(self.N_s + 1):
            m, im = _spin_basis(self.N_s, n)
            self.masks[n] = m
            self.idx_map[n] = im
            self.dims[n] = len(m)

        # Precompute single-spin hopping transitions (0 <-> l) inside each n-particle sector
        # hop_tables[n][l] = (rows, cols, signs) for c_0^dag c_l + h.c. (upper/lower both included)
        self.hop_tables = {}
        for n in range(self.N_s + 1):
            self.hop_tables[n] = {}
            ms = self.masks[n]
            im = self.idx_map[n]
            for l in range(1, self.N_s):
                rows, cols, signs = [], [], []
                bit0 = 1
                bitl = 1 << l
                for i, state in enumerate(ms):
                    # c_0^dag c_l |state>: requires bit l = 1, bit 0 = 0
                    if (state & bitl) and not (state & bit0):
                        # Remove l, then add 0: sign = (-1)^(popcount_below(state, l)) * (+1 for bit 0)
                        sgn = -1.0 if (_popcount_below(int(state), l) & 1) else 1.0
                        new_state = (int(state) ^ bitl) | bit0
                        j = im[new_state]
                        rows.append(j)
                        cols.append(i)
                        signs.append(sgn)
                        # Hermitian conjugate c_l^dag c_0
                        rows.append(i)
                        cols.append(j)
                        signs.append(sgn)
                self.hop_tables[n][l] = (
                    np.array(rows, dtype=np.int32),
                    np.array(cols, dtype=np.int32),
                    np.array(signs, dtype=np.float64),
                )

        # Precompute impurity occupation mask n_0 in each single-spin sector
        self.n0_vec = {n: ((self.masks[n] & 1).astype(np.float64)) for n in range(self.N_s + 1)}
        # Precompute bath occupation matrix occ[n] of shape (dim_n, N_b)
        self.bath_occ = {}
        for n in range(self.N_s + 1):
            occ = np.zeros((self.dims[n], self.N_b), dtype=np.float64)
            for l in range(self.N_b):
                occ[:, l] = ((self.masks[n] >> (l + 1)) & 1).astype(np.float64)
            self.bath_occ[n] = occ

        # Precompute impurity creation operator c_{0}^dag: n -> n + 1 (always sign +1 because bit 0 is LSB!)
        # For spin-up in |i_up, i_dn> with ordering (c_{up}^dag ...)(c_{dn}^dag ...)|0>,
        # c_{0,up}^dag acts on the leftmost slot => sign = +1!
        # c_{0,dn}^dag hops past N_up spin-up fermions => sign = (-1)^{N_up}!
        self.cdag0_map = {}
        for n in range(self.N_s):
            src_idx, dst_idx = [], []
            im_next = self.idx_map[n + 1]
            for i, state in enumerate(self.masks[n]):
                if not (int(state) & 1):
                    src_idx.append(i)
                    dst_idx.append(im_next[int(state) | 1])
            self.cdag0_map[n] = (np.array(src_idx, dtype=np.int32), np.array(dst_idx, dtype=np.int32))

    def _build_h1_spin(self, n, mu, eps_b, V_b):
        """Single-spin Hamiltonian matrix H1(n) of shape (dim_n, dim_n)."""
        d = self.dims[n]
        H1 = np.zeros((d, d), dtype=np.float64)
        diag = -mu * self.n0_vec[n] + self.bath_occ[n] @ eps_b
        np.fill_diagonal(H1, diag)
        for l_idx in range(self.N_b):
            V_l = float(V_b[l_idx])
            if abs(V_l) > 0.0:
                rows, cols, signs = self.hop_tables[n][l_idx + 1]
                if len(rows) > 0:
                    H1[rows, cols] += V_l * signs
        return H1

    def diagonalize(self, U, mu, eps_b, V_b):
        """
        Diagonalize all (N_up, N_dn) sectors using exact spin-flip symmetry (N_up >= N_dn).
        Returns eigensystem dictionary keyed by (n_up, n_dn): (evals, evecs).
        """
        U = float(U)
        mu = float(mu)
        eps_b = np.asarray(eps_b, dtype=np.float64)
        V_b = np.asarray(V_b, dtype=np.float64)

        H1 = {n: self._build_h1_spin(n, mu, eps_b, V_b) for n in range(self.N_s + 1)}
        sectors = {}
        E_min = np.inf

        for n_up in range(self.N_s + 1):
            d_up = self.dims[n_up]
            H_up = H1[n_up]
            n0_up = self.n0_vec[n_up]
            for n_dn in range(n_up + 1):
                d_dn = self.dims[n_dn]
                H_dn = H1[n_dn]
                n0_dn = self.n0_vec[n_dn]

                # Full sector Hamiltonian in Kronecker basis |i_up> x |i_dn> (size d_up * d_dn)
                # H = H_up (x) I + I (x) H_dn + U * diag(n0_up (x) n0_dn)
                H_sec = np.kron(H_up, np.eye(d_dn)) + np.kron(np.eye(d_up), H_dn)
                if abs(U) > 0.0:
                    H_sec[np.arange(d_up * d_dn), np.arange(d_up * d_dn)] += U * np.outer(n0_up, n0_dn).ravel()

                evals, evecs = np.linalg.eigh(H_sec)
                if evals[0] < E_min:
                    E_min = float(evals[0])
                sectors[(n_up, n_dn)] = (evals, evecs)

                if n_up != n_dn:
                    # Spin-flip partner (n_dn, n_up): reshape (d_up, d_dn, k) -> transpose (d_dn, d_up, k)
                    ev_swap = evecs.reshape(d_up, d_dn, -1).transpose(1, 0, 2).reshape(d_dn * d_up, -1)
                    sectors[(n_dn, n_up)] = (evals, ev_swap)

        return sectors, E_min

    def solve(self, dlr: DLRBasis, U, mu, eps_b, V_b, compute_chi=True, boltz_cut=1e-11):
        """
        Solve SIAM via exact symmetry-sector Lehmann representation on the DLR grid.
        Returns g_imp, Sig_imp, n_imp, D_imp, Lam, chi_imp_{s,c,pp}, and full eigensystem
        for reuse in Step 3b (exact Rank-R SVD 4-point local vertex extraction).
        """
        beta = dlr.beta
        eps_b = np.asarray(eps_b, dtype=np.float64)
        V_b = np.asarray(V_b, dtype=np.float64)
        sectors, E_min = self.diagonalize(U, mu, eps_b, V_b)

        # Partition function Z and thermally active eigenstates
        Z = 0.0
        active = {}
        for sec, (evals, _) in sectors.items():
            w = np.exp(-beta * (evals - E_min))
            Z += float(np.sum(w))
            keep = np.where(w > boltz_cut)[0]
            if len(keep) > 0:
                active[sec] = (keep, w[keep])

        # Exact thermal observables n_imp and D_imp = <n_{0,up} n_{0,dn}>
        n_up_avg = 0.0
        D_imp = 0.0
        for (n_up, n_dn), (keep, w_k) in active.items():
            _, evecs = sectors[(n_up, n_dn)]
            d_up, d_dn = self.dims[n_up], self.dims[n_dn]
            psi2 = (evecs[:, keep] ** 2).reshape(d_up, d_dn, -1)
            n0_u = self.n0_vec[n_up][:, None, None]
            n0_d = self.n0_vec[n_dn][None, :, None]
            n_up_avg += float(np.sum(np.sum(psi2 * n0_u, axis=(0, 1)) * w_k))
            D_imp += float(np.sum(np.sum(psi2 * n0_u * n0_d, axis=(0, 1)) * w_k))
        n_up_avg /= Z
        n_imp = 2.0 * n_up_avg
        D_imp /= Z

        # Exact Lehmann single-particle Green's function g_imp(iwn) for spin-up:
        #   g(iwn) = (1/Z) sum_{a, b} (e^{-b E_a} + e^{-b E_b}) |<b| c_{0,up}^dag |a>|^2 / (iwn - (E_b - E_a))
        iwn = dlr.iwn
        g_imp = np.zeros(dlr.r_F, dtype=np.complex128)
        for n_up in range(self.N_s):
            src_u, dst_u = self.cdag0_map[n_up]
            if len(src_u) == 0:
                continue
            d_up = self.dims[n_up]
            d_up1 = self.dims[n_up + 1]
            for n_dn in range(self.N_s + 1):
                sec_a = (n_up, n_dn)
                sec_b = (n_up + 1, n_dn)
                act_a = active.get(sec_a)
                act_b = active.get(sec_b)
                if act_a is None and act_b is None:
                    continue
                evals_a, evecs_a = sectors[sec_a]
                evals_b, evecs_b = sectors[sec_b]
                d_dn = self.dims[n_dn]

                # Select only states participating in thermally active pairs
                ka = act_a[0] if act_a is not None else np.array([], dtype=np.int32)
                kb = act_b[0] if act_b is not None else np.array([], dtype=np.int32)
                # Note: either a or b must be active -> compute full block with active rows/cols
                # To avoid double counting, evaluate full matrix M_{ba} = <b| c_{0,up}^dag |a>
                # on active_a x all_b AND (inactive_a) x active_b!
                Ea_all = evals_a
                Eb_all = evals_b
                wa_all = np.exp(-beta * (Ea_all - E_min))
                wb_all = np.exp(-beta * (Eb_all - E_min))

                # Reshape evecs to (d_up, d_dn, dim_tot) and contract along (src_u/dst_u, d_dn)
                Va = evecs_a.reshape(d_up, d_dn, -1)[src_u, :, :].reshape(len(src_u) * d_dn, -1)
                Vb = evecs_b.reshape(d_up1, d_dn, -1)[dst_u, :, :].reshape(len(dst_u) * d_dn, -1)

                if len(ka) > 0:
                    # Active a -> all b
                    M_ab = Vb.T @ Va[:, ka]  # (dim_b, len(ka))
                    wt = (wa_all[ka][None, :] + wb_all[:, None]) * (M_ab ** 2)
                    dE = Eb_all[:, None] - Ea_all[ka][None, :]
                    mask_nz = wt > 1e-16
                    if np.any(mask_nz):
                        wt_nz = wt[mask_nz]
                        dE_nz = dE[mask_nz]
                        g_imp += np.sum(wt_nz[None, :] / (iwn[:, None] - dE_nz[None, :]), axis=1)

                if len(kb) > 0:
                    # Inactive a -> active b (so no pair is counted twice)
                    inact_a = np.where(wa_all <= boltz_cut)[0]
                    if len(inact_a) > 0:
                        M_ba = Vb[:, kb].T @ Va[:, inact_a]  # (len(kb), len(inact_a))
                        wt = (wa_all[inact_a][None, :] + wb_all[kb][:, None]) * (M_ba ** 2)
                        dE = Eb_all[kb][:, None] - Ea_all[inact_a][None, :]
                        mask_nz = wt > 1e-16
                        if np.any(mask_nz):
                            wt_nz = wt[mask_nz]
                            dE_nz = dE[mask_nz]
                            g_imp += np.sum(wt_nz[None, :] / (iwn[:, None] - dE_nz[None, :]), axis=1)
        g_imp /= Z

        # Exact hybridization Delta_fit(iwn) and impurity self-energy Sigma_imp(iwn)
        Delta_fit = np.sum((V_b[None, :] ** 2) / (iwn[:, None] - eps_b[None, :]), axis=1)
        G0_inv = iwn + mu - Delta_fit
        Sig_imp = G0_inv - 1.0 / g_imp
        Lam = np.maximum(1.0 - np.imag(Sig_imp) / dlr.wn, 1.0)

        # Local physical bubbles on g_imp (matching Track IV-A interface!)
        g_tau = np.real(dlr.iwn_to_tau(g_imp))
        Lam_g_tau = np.real(dlr.iwn_to_tau(np.sqrt(Lam) * g_imp))
        chi0_loc = np.real(dlr.tauB_to_iOm(g_tau * g_tau[::-1]))
        chi_Lam_loc = np.real(dlr.tauB_to_iOm(Lam_g_tau * Lam_g_tau[::-1]))
        chi_pp_Lam_loc = dlr.tauB_to_iOm(Lam_g_tau ** 2)

        phi0_g = g_imp
        phi1_g = (Lam - 1.0) * g_imp
        X0_loc_static = np.array([
            [float(np.real(-dlr.fermion_prod_sum(phi0_g, phi0_g))), float(np.real(-dlr.fermion_prod_sum(phi0_g, phi1_g)))],
            [float(np.real(-dlr.fermion_prod_sum(phi0_g, phi1_g))), float(np.real(-dlr.fermion_prod_sum(phi1_g, phi1_g)))],
        ], dtype=float)

        chi_imp_s = np.zeros(dlr.r_B, dtype=np.float64)
        chi_imp_c = np.zeros(dlr.r_B, dtype=np.float64)
        chi_imp_pp = np.zeros(dlr.r_B, dtype=np.complex128)

        if compute_chi:
            iOm = dlr.iOm
            m0 = dlr.m0_idx
            # Exact Lehmann bosonic susceptibilities:
            #   chi_O(iOm) = int_0^beta dtau e^{iOm tau} <O(tau) O^dag(0)>_conn
            #              = (1/Z) sum_{a,b} |<b|O^dag|a>|^2 * [ (w_a - w_b)/(E_b - E_a - iOm) if |dE|>1e-12 else beta*w_a*delta_{m,0} ]
            # Operators:
            #   Spin:   O_s = n_{0,up} - n_{0,dn}  (gives chi_s = <m_z m_z>, Pauli sum = n - 2D, exact match to Track IV-A!)
            #   Charge: O_c = n_{0,up} + n_{0,dn} - n_imp (gives chi_c = <dn dn>, Pauli sum = n(1-n) + 2D, exact match!)
            for (n_up, n_dn), (evals, evecs) in sectors.items():
                act = active.get((n_up, n_dn))
                if act is None:
                    continue
                ka, wa_k = act
                d_up, d_dn = self.dims[n_up], self.dims[n_dn]
                n0_u = self.n0_vec[n_up][:, None]
                n0_d = self.n0_vec[n_dn][None, :]
                op_s = (n0_u - n0_d).ravel()
                op_c = (n0_u + n0_d - n_imp).ravel()
                w_all = np.exp(-beta * (evals - E_min))

                # Evaluate M_ba = evecs^T diag(op) evecs[:, ka]
                # To avoid double counting active-active pairs, weight factor is:
                # For a in ka, b in all: if b in ka we see both (a,b) and (b,a); simpler:
                # Use exact formula summing over a in ka, b in ALL with (w_a - w_b)/(dE - iOm)?
                # Wait: (w_a - w_b)/(E_b - E_a - iOm) = (w_b - w_a)/(E_a - E_b + iOm)!
                # Sum over ALL (a, b) of (w_a - w_b)/(E_b - E_a - iOm) = 2 * sum_{a} w_a sum_b 1/(E_b - E_a - iOm)?
                # Careful! Let's use the exact single-active-index form (zero double counting!):
                #   chi_O(iOm) = (1/Z) sum_{a active, b all} w_a |M_{ba}|^2 * [ 1/(dE - iOm) + 1/(dE + iOm) ]  for |dE| > 1e-12
                #              + (beta/Z) sum_{a active, b: |dE|<=1e-12} w_a |M_{ba}|^2 * delta_{m, 0}
                # Check: int_0^beta dtau e^{iOm tau} e^{(E_a - E_b) tau} = (e^{(E_a - E_b) beta} - 1) / (E_a - E_b + iOm)
                #        = (w_b - w_a) / (E_a - E_b + iOm) = (w_a - w_b) / (E_b - E_a - iOm).
                # Summing over ALL (a, b):
                #   sum_{a,b} w_a |M_{ba}|^2 / (E_b - E_a - iOm) - sum_{a,b} w_b |M_{ba}|^2 / (E_b - E_a - iOm)
                # Relabel a <-> b in 2nd term (using |M_{ab}|^2 = |M_{ba}|^2 for Hermitian O!):
                #   = sum_{a,b} w_a |M_{ba}|^2 [ 1/(E_b - E_a - iOm) + 1/(E_b - E_a + iOm) ]!
                # EXACT and only sums over active `a`!
                V_act = evecs[:, ka]
                dE = evals[:, None] - evals[ka][None, :]  # (dim, len(ka))
                deg = np.abs(dE) < 1e-11
                safe_dE = np.where(deg, 1.0, dE)
                # Precompute kernel K_ba(iOm) = 2*dE / (dE^2 + Om^2) for non-deg, beta*delta_{m,0} for deg
                for op_vec, target_chi in ((op_s, chi_imp_s), (op_c, chi_imp_c)):
                    M2 = (evecs.T @ (op_vec[:, None] * V_act)) ** 2  # (dim, len(ka))
                    wt = wa_k[None, :] * M2
                    if np.any(deg):
                        target_chi[m0] += float(beta * np.sum(wt[deg]))
                    nd = (~deg) & (wt > 1e-16)
                    if np.any(nd):
                        wt_nd = wt[nd]
                        dE_nd = safe_dE[nd]
                        # Real symmetric: 1/(dE - iOm) + 1/(dE + iOm) = 2*dE / (dE^2 + Om^2)
                        target_chi += np.sum(
                            (2.0 * wt_nd * dE_nd)[None, :] / (dE_nd[None, :] ** 2 + dlr.Om[:, None] ** 2),
                            axis=1,
                        )
            chi_imp_s /= Z
            chi_imp_c /= Z

            # Exact Lehmann singlet pairing susceptibility chi_pp(iOm) = int_0^beta dtau e^{iOm tau} <Delta(tau) Delta^dag(0)>
            # with Delta = c_{0,dn} c_{0,up} (maps (n_up+1, n_dn+1) -> (n_up, n_dn)),
            # Delta^dag = c_{0,up}^dag c_{0,dn}^dag (maps (n_up, n_dn) -> (n_up+1, n_dn+1))!
            # Exact sum rule: (1/beta) sum_m chi_pp(iOm_m) = 0.5 * (<Delta Delta^dag> + <Delta^dag Delta>)
            #               = 0.5 * (<(1-n_up)(1-n_dn)> + <n_up n_dn>) = 0.5*(1 - n) + D_imp!
            for n_up in range(self.N_s):
                src_u, dst_u = self.cdag0_map[n_up]
                if len(src_u) == 0:
                    continue
                d_up, d_up1 = self.dims[n_up], self.dims[n_up + 1]
                for n_dn in range(self.N_s):
                    src_d, dst_d = self.cdag0_map[n_dn]
                    if len(src_d) == 0:
                        continue
                    sec_a = (n_up, n_dn)
                    sec_b = (n_up + 1, n_dn + 1)
                    act_a = active.get(sec_a)
                    act_b = active.get(sec_b)
                    if act_a is None and act_b is None:
                        continue
                    Ea, Va_full = sectors[sec_a]
                    Eb, Vb_full = sectors[sec_b]
                    d_dn, d_dn1 = self.dims[n_dn], self.dims[n_dn + 1]
                    wa_all = np.exp(-beta * (Ea - E_min))
                    wb_all = np.exp(-beta * (Eb - E_min))

                    # Sub-slices matching c_{0,up}^dag c_{0,dn}^dag:
                    Va = Va_full.reshape(d_up, d_dn, -1)[np.ix_(src_u, src_d, np.arange(Va_full.shape[1]))].reshape(len(src_u) * len(src_d), -1)
                    Vb = Vb_full.reshape(d_up1, d_dn1, -1)[np.ix_(dst_u, dst_d, np.arange(Vb_full.shape[1]))].reshape(len(dst_u) * len(dst_d), -1)

                    # chi_pp(iOm) = (1/Z) sum_{a,b} |<b|Delta^dag|a>|^2 * (w_a - w_b) / (E_b - E_a - iOm)
                    # Note: pair creation b has E_b - E_a ~ U - 2*mu -> 1/(-iOm + U - 2*mu) at t=0!
                    # Let's check sign: int_0^beta dtau e^{iOm tau} <Delta(tau) Delta^dag(0)>
                    # = (1/Z) sum_{a,b} w_a |<b|Delta^dag|a>|^2 int_0^beta dtau e^{(iOm + E_a - E_b) tau}
                    # = (1/Z) sum_{a,b} |<b|Delta^dag|a>|^2 (w_b - w_a) / (iOm + E_a - E_b)
                    # = (1/Z) sum_{a,b} |<b|Delta^dag|a>|^2 (w_a - w_b) / (E_b - E_a - iOm).
                    # Atomic check: a = |0> (E_a=0, w_a=1), b = |ud> (E_b=U-2mu, w_b=e^{-b(U-2mu)}),
                    # (w_a - w_b)/Z = (1 - n_imp), denom = -iOm + U - 2mu => (1 - n_imp)/(-iOm + U - 2mu)! EXACT!
                    ka = act_a[0] if act_a is not None else np.array([], dtype=np.int32)
                    kb = act_b[0] if act_b is not None else np.array([], dtype=np.int32)

                    if len(ka) > 0:
                        M2 = (Vb.T @ Va[:, ka]) ** 2
                        dE = Eb[:, None] - Ea[ka][None, :]
                        dw = (wa_all[ka][None, :] - wb_all[:, None]) * M2
                        deg = np.abs(dE) < 1e-11
                        if np.any(deg):
                            chi_imp_pp[m0] += float(beta * np.sum((wa_all[ka][None, :] * M2)[deg]))
                        nd = (~deg) & (np.abs(dw) > 1e-16)
                        if np.any(nd):
                            chi_imp_pp += np.sum(dw[nd][None, :] / (dE[nd][None, :] - iOm[:, None]), axis=1)

                    if len(kb) > 0:
                        inact_a = np.where(wa_all <= boltz_cut)[0]
                        if len(inact_a) > 0:
                            M2 = (Vb[:, kb].T @ Va[:, inact_a]) ** 2
                            dE = Eb[kb][:, None] - Ea[inact_a][None, :]
                            dw = (wa_all[inact_a][None, :] - wb_all[kb][:, None]) * M2
                            deg = np.abs(dE) < 1e-11
                            if np.any(deg):
                                chi_imp_pp[m0] += float(beta * np.sum((wa_all[inact_a][None, :] * M2)[deg]))
                            nd = (~deg) & (np.abs(dw) > 1e-16)
                            if np.any(nd):
                                chi_imp_pp += np.sum(dw[nd][None, :] / (dE[nd][None, :] - iOm[:, None]), axis=1)
            chi_imp_pp /= Z

        return {
            "eps_b": eps_b,
            "V_b": V_b,
            "Delta_fit": Delta_fit,
            "Sig_imp": Sig_imp,
            "g": g_imp,
            "Lam": Lam,
            "n_imp": float(n_imp),
            "D_imp": float(D_imp),
            "chi_imp_s": chi_imp_s,
            "chi_imp_c": chi_imp_c,
            "chi_imp_pp": chi_imp_pp,
            "chi0_loc": chi0_loc,
            "chi_Lam_loc": chi_Lam_loc,
            "chi_pp_Lam_loc": chi_pp_Lam_loc,
            "X0_loc_static": X0_loc_static,
            "Z_part": float(Z),
            "E_min": float(E_min),
            "sectors": sectors,
            "active": active,
        }


def fit_bath_dlr(dlr: DLRBasis, Delta_iwn, N_b=4, half_fill=True, init_params=None, weight_power=0.75):
    """
    Fit continuous hybridization Delta(iwn) on the positive DLR Matsubara grid to
      Delta_fit(iwn) = sum_{l=1}^{N_b} V_l^2 / (iwn - eps_l)
    using nonlinear least squares with low-frequency emphasis 1 / |wn|^weight_power.
    Enforces exact particle-hole symmetry when half_fill=True.
    """
    pos = dlr.wn > 0.0
    wn_p = dlr.wn[pos]
    iwn_p = 1j * wn_p
    D_tgt = Delta_iwn[pos]
    w_fit = 1.0 / (wn_p ** weight_power)
    w_fit /= np.linalg.norm(w_fit)

    if half_fill:
        n_pairs = N_b // 2
        has_zero = (N_b % 2) == 1

        def unpack(x):
            eps_pos = x[:n_pairs]
            V_pairs = x[n_pairs : 2 * n_pairs]
            eps_list = []
            V_list = []
            for ep, vp in zip(eps_pos, V_pairs):
                eps_list.extend([ep, -ep])
                V_list.extend([vp, vp])
            if has_zero:
                eps_list.append(0.0)
                V_list.append(x[-1])
            return np.array(eps_list, dtype=float), np.array(V_list, dtype=float)

        if init_params is None:
            eps0 = np.linspace(0.6, 3.2, n_pairs)
            V0 = np.full(n_pairs, np.sqrt(3.0 / max(n_pairs + 0.5 * has_zero, 1)))
            x0 = np.concatenate([eps0, V0, [1.2] if has_zero else []])
        else:
            x0 = np.asarray(init_params, dtype=float)

        lo = np.concatenate([np.full(n_pairs, 0.02), np.zeros(n_pairs + int(has_zero))])
        hi = np.concatenate([np.full(n_pairs, 15.0), np.full(n_pairs + int(has_zero), 10.0)])

        def res(x):
            eb, Vb = unpack(x)
            D_fit = np.sum((Vb[None, :] ** 2) / (iwn_p[:, None] - eb[None, :]), axis=1)
            diff = w_fit * (D_fit - D_tgt)
            return np.imag(diff)

        opt = least_squares(res, np.clip(x0, lo + 1e-4, hi - 1e-4), bounds=(lo, hi), ftol=1e-11, xtol=1e-11, max_nfev=400)
        eps_b, V_b = unpack(opt.x)
    else:
        def unpack(x):
            return x[:N_b], x[N_b:]

        if init_params is None:
            eps0 = np.linspace(-2.5, 2.5, N_b)
            V0 = np.full(N_b, np.sqrt(6.0 / N_b))
            x0 = np.concatenate([eps0, V0])
        else:
            x0 = np.asarray(init_params, dtype=float)

        lo = np.concatenate([np.full(N_b, -15.0), np.zeros(N_b)])
        hi = np.concatenate([np.full(N_b, 15.0), np.full(N_b, 10.0)])

        def res(x):
            eb, Vb = unpack(x)
            D_fit = np.sum((Vb[None, :] ** 2) / (iwn_p[:, None] - eb[None, :]), axis=1)
            diff = w_fit * (D_fit - D_tgt)
            return np.concatenate([np.real(diff), np.imag(diff)])

        opt = least_squares(res, np.clip(x0, lo + 1e-4, hi - 1e-4), bounds=(lo, hi), ftol=1e-11, xtol=1e-11, max_nfev=500)
        eps_b, V_b = unpack(opt.x)

    D_fit_full = np.sum((V_b[None, :] ** 2) / (dlr.iwn[:, None] - eps_b[None, :]), axis=1)
    fit_err = float(np.max(np.abs(D_fit_full - Delta_iwn)))
    return eps_b, V_b, D_fit_full, fit_err, opt.x
