"""
QMC-Free Unified Level-III MR-ALDF Solver (MR-ALDF-III) for the 3D Simple-Cubic Hubbard Model.

Implements the complete Table 1 unified closed MR-ALDF-III self-consistent loop:
  - Exact fourth-spectral-moment interpolating perturbation theory (mIPT-4) with
    correlated kinetic band shift Delta B_kin and tail-subtracted Galitskii-Migdal D_imp
  - (s, c, pp) local-TPSC susceptibilities on the tail-corrected Weiss bubble
  - UV-tail-regularized Rank-2 Ward-constrained local charge vertex W_c^reg (detecting D_c^irr = 0)
  - Non-singular dual-denominator Moriya lambda_s via UV-tail-free Pauli excess sum rule
  - Three-channel (s + c + pp) dual self-energy in O(N log N) time via 3D FFTs
  - Linearized 3D O_h E_g (d-wave) pairing eigenvalue diagnostic via FFT power iteration
  - Iterative O(N log N) Maki-Thompson dual current vertex corrections and Delta_Ward
  - Coupled inner-outer Pulay/DIIS mixing with Theorem 5.1 causality monitoring
  - 50-digit Vidberg-Serene Pade analytic continuation on the O_h IBZ and optical conductivity
"""

import numpy as np
_trapz = getattr(np, "trapezoid", np.trapz)
from scipy.optimize import brentq

try:
    import mpmath as mp
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False


# ==============================================================================
# Pulay / DIIS Accelerator & Analytic Continuation Utilities
# ==============================================================================

class PulayDIIS:
    """Pulay/DIIS fixed-point accelerator for complex state vectors."""

    def __init__(self, m_max=6, reg=1e-12):
        self.m_max = int(m_max)
        self.reg = float(reg)
        self.xs = []
        self.rs = []

    def reset(self):
        self.xs.clear()
        self.rs.clear()

    def update(self, x_in, x_out, beta_mix=0.4):
        r = x_out - x_in
        self.xs.append(x_in.copy())
        self.rs.append(r.copy())
        if len(self.xs) > self.m_max:
            self.xs.pop(0)
            self.rs.pop(0)
        m = len(self.xs)
        if m < 2:
            return x_in + beta_mix * r

        B = np.zeros((m + 1, m + 1), dtype=float)
        for i in range(m):
            for j in range(i, m):
                v = float(np.real(np.vdot(self.rs[i], self.rs[j])))
                B[i, j] = B[j, i] = v
        B[m, :m] = -1.0
        B[:m, m] = -1.0
        scale = max(float(np.max(np.abs(np.diag(B[:m, :m])))), 1e-30)
        B[:m, :m] += self.reg * scale * np.eye(m)

        rhs = np.zeros(m + 1, dtype=float)
        rhs[m] = -1.0
        try:
            c = np.linalg.solve(B, rhs)[:m]
            if np.max(np.abs(c)) > 25.0 or not np.all(np.isfinite(c)):
                return x_in + beta_mix * r
            x_bar = np.zeros_like(x_in)
            r_bar = np.zeros_like(r)
            for i in range(m):
                x_bar += c[i] * self.xs[i]
                r_bar += c[i] * self.rs[i]
            return x_bar + beta_mix * r_bar
        except np.linalg.LinAlgError:
            return x_in + beta_mix * r


def select_pade_indices(Nf, n_low=12, n_high=8):
    """
    Hybrid consecutive low-frequency plus geometrically spaced high-frequency
    positive Matsubara indices in [0, Nf).
    """
    n_low = min(int(n_low), int(Nf))
    low = list(range(n_low))
    if n_low >= Nf or n_high <= 0:
        return np.array(low, dtype=int)
    hi = np.unique(
        np.round(np.geomspace(n_low, Nf - 1, num=int(n_high))).astype(int)
    )
    idx = np.unique(np.concatenate([low, hi]))
    return idx.astype(int)


def pade_eval_mpmath(z_nodes, u_vals, z_eval, dps=50):
    """50-digit Vidberg-Serene Pade analytic continuation."""
    if not HAS_MPMATH:
        raise ImportError("mpmath is required for high-precision Pade continuation.")
    with mp.workdps(dps):
        N = len(z_nodes)
        z = [mp.mpc(float(np.real(v)), float(np.imag(v))) for v in z_nodes]
        g = [[mp.mpc(0)] * N for _ in range(N)]
        for i in range(N):
            g[0][i] = mp.mpc(float(np.real(u_vals[i])), float(np.imag(u_vals[i])))
        for p in range(1, N):
            for i in range(p, N):
                denom = (z[i] - z[p - 1]) * g[p - 1][i]
                if abs(denom) < mp.mpf("1e-40"):
                    g[p][i] = mp.mpc(0)
                else:
                    g[p][i] = (g[p - 1][p - 1] - g[p - 1][i]) / denom
        a = [g[p][p] for p in range(N)]

        out = np.empty(len(z_eval), dtype=np.complex128)
        for idx, zv in enumerate(z_eval):
            zm = mp.mpc(float(np.real(zv)), float(np.imag(zv)))
            A_prev, A_curr = mp.mpc(0), a[0]
            B_prev, B_curr = mp.mpc(1), mp.mpc(1)
            for p in range(1, N):
                fac = (zm - z[p - 1]) * a[p]
                A_next = A_curr + fac * A_prev
                B_next = B_curr + fac * B_prev
                A_prev, A_curr = A_curr, A_next
                B_prev, B_curr = B_curr, B_next
            val = A_curr / B_curr if abs(B_curr) > mp.mpf("1e-45") else mp.mpc(0)
            out[idx] = complex(float(mp.re(val)), float(mp.im(val)))
        return out


# ==============================================================================
# Unified QMC-Free Level-III MR-ALDF Solver
# ==============================================================================

class MRALDFLevel3Solver:
    """
    QMC-free Moriya-Renormalized Analytical Ladder Dual-Fermion solver (Levels II & III)
    on the 3D simple-cubic lattice.
    """

    def __init__(
        self,
        U,
        T,
        delta=0.0,
        t=1.0,
        L=8,
        Nf=40,
        Nb=16,
        Ntau=512,
        level=3,
        eps_c=0.02,
        eps_pp=0.05,
    ):
        self.U = float(U)
        self.T = float(T)
        self.beta = 1.0 / self.T
        self.delta = float(delta)
        self.n_target = 1.0 - self.delta
        self.t = float(t)
        self.L = int(L)
        self.N = self.L ** 3
        self.Nf = int(Nf)
        self.Nb = int(Nb)
        self.Ntau = int(Ntau)
        if self.Ntau < 4 * self.Nf:
            self.Ntau = 1 << int(np.ceil(np.log2(max(4 * self.Nf, 256))))
        self.level = int(level)
        self.eps_c = float(eps_c)
        self.eps_pp = float(eps_pp)

        # Fermionic Matsubara grids: working window [-Nf, Nf) and full FFT grid [-Ntau/2, Ntau/2)
        self.n_f = np.arange(-self.Nf, self.Nf, dtype=int)
        self.wn = (2.0 * self.n_f + 1.0) * np.pi / self.beta
        self.iwn = 1j * self.wn

        self.n_fft = np.arange(-self.Ntau // 2, self.Ntau // 2, dtype=int)
        self.wn_fft = (2.0 * self.n_fft + 1.0) * np.pi / self.beta
        self.iwn_fft = 1j * self.wn_fft
        self.fft_offset = self.Ntau // 2 - self.Nf

        # Midpoint imaginary-time grid tau_j = (j + 1/2) * beta / Ntau
        self.dtau = self.beta / self.Ntau
        self.tau = (np.arange(self.Ntau, dtype=float) + 0.5) * self.dtau

        # Bosonic Matsubara grid: working window [-Nb, Nb] and full FFT grid [-Ntau/2, Ntau/2)
        self.m_b = np.arange(-self.Nb, self.Nb + 1, dtype=int)
        self.Om_b = 2.0 * self.m_b * np.pi / self.beta
        self.m_b_fft = np.arange(-self.Ntau // 2, self.Ntau // 2, dtype=int)
        self.Om_b_fft = 2.0 * self.m_b_fft * np.pi / self.beta

        # Precompute phase factors for midpoint-tau <-> Matsubara FFTs
        j_idx = np.arange(self.Ntau, dtype=float)
        k_idx = np.arange(self.Ntau, dtype=float)
        self._f_in_phase = np.exp(-1j * np.pi * k_idx / self.Ntau)
        self._f_out_phase = np.exp(1j * np.pi * (self.Ntau - 1.0) * (j_idx + 0.5) / self.Ntau)
        self._fi_in_phase = np.conj(self._f_out_phase)
        self._fi_out_phase = np.conj(self._f_in_phase)

        self._bi_in_phase = np.exp(-1j * np.pi * (j_idx + 0.5))
        self._bi_out_phase = np.exp(1j * np.pi * k_idx / self.Ntau)

        # 3D simple-cubic momentum grid
        k1d = 2.0 * np.pi * np.arange(self.L) / self.L
        kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
        self.kx, self.ky, self.kz = kx, ky, kz
        self.eps_k = -2.0 * self.t * (np.cos(kx) + np.cos(ky) + np.cos(kz))
        self.M2 = 6.0 * (self.t ** 2)
        self.vx = 2.0 * self.t * np.sin(kx)
        self.vy = 2.0 * self.t * np.sin(ky)
        self.vz = 2.0 * self.t * np.sin(kz)
        self.v2 = self.vx ** 2 + self.vy ** 2 + self.vz ** 2

        # O_h E_g (d-wave) normalized form factors: <psi_d^2>_k = 1
        self.psi_d1 = np.cos(kx) - np.cos(ky)
        self.psi_d2 = (2.0 * np.cos(kz) - np.cos(kx) - np.cos(ky)) / np.sqrt(3.0)

        # O_h Irreducible Brillouin Zone (IBZ) map
        self._build_oh_ibz()

        # Spatial reversal map -r mod L
        self._r_neg = (self.L - np.arange(self.L)) % self.L

        # Initial physical/dual state
        self.mu = 0.5 * self.U if abs(self.delta) < 1e-12 else 0.5 * self.U - 2.0 * self.t * self.delta
        if self.t > 0.0:
            self.Delta = self.M2 / (self.iwn + 1j * np.sign(self.wn) * max(2.0 * self.t, 0.5 * self.U))
        else:
            self.Delta = np.zeros(2 * self.Nf, dtype=np.complex128)
        self.Sigma_tilde = np.zeros((2 * self.Nf, self.L, self.L, self.L), dtype=np.complex128)
        self.Sigma_tilde_ph = np.zeros_like(self.Sigma_tilde)
        self.Sigma_tilde_pp = np.zeros_like(self.Sigma_tilde)

        # Caches
        self.last_imp = None
        self.last_vert = None
        self.last_moriya = None
        self.last_V_ph = None
        self.last_Sig_lat = None
        self.last_G_tilde = None
        self.last_G_lat = None

    def _build_oh_ibz(self):
        """Construct the O_h irreducible Brillouin zone coordinates, weights, and inverse map."""
        L = self.L
        rep_map = {}
        coords = []
        counts = []
        inv_map = np.empty((L, L, L), dtype=int)
        for ix in range(L):
            ax = min(ix, L - ix)
            for iy in range(L):
                ay = min(iy, L - iy)
                for iz in range(L):
                    az = min(iz, L - iz)
                    key = tuple(sorted((ax, ay, az)))
                    if key not in rep_map:
                        rep_map[key] = len(coords)
                        coords.append(key)
                        counts.append(0)
                    p = rep_map[key]
                    inv_map[ix, iy, iz] = p
                    counts[p] += 1
        self.ibz_coords = coords
        self.ibz_inv_map = inv_map
        self.ibz_weights = np.array(counts, dtype=float) / float(self.N)
        self.ibz_eps = np.array([self.eps_k[c] for c in coords], dtype=float)

    # ---------- Midpoint-tau <-> Matsubara FFT Helpers ----------

    def _neg_r(self, arr_r):
        idx = self._r_neg
        return arr_r[:, idx, :, :][:, :, idx, :][:, :, :, idx]

    def _interp_delta_to_fft(self, Delta_small):
        """Embed working-window Delta(iwn) onto full Ntau FFT grid with M2 / iwn tail."""
        if self.t == 0.0:
            Del_fft = np.zeros(self.Ntau, dtype=np.complex128)
        else:
            Del_fft = self.M2 / self.iwn_fft
        s0 = self.fft_offset
        Del_fft[s0 : s0 + 2 * self.Nf] = Delta_small
        return Del_fft

    def _fermion_iwn_to_tau(self, f_iwn, c1=1.0, xi=0.0):
        """Transform fermionic f(iwn_fft) -> f(tau_j) with analytic tail subtraction."""
        if c1 != 0.0:
            g_ref = c1 / (self.iwn_fft - xi)
            diff = f_iwn - g_ref
            bx = self.beta * xi
            if bx >= 0.0:
                ref_tau = -c1 * np.exp(-xi * self.tau) / (1.0 + np.exp(-bx))
            else:
                ref_tau = -c1 * np.exp(xi * (self.beta - self.tau)) / (1.0 + np.exp(bx))
        else:
            diff = f_iwn
            ref_tau = 0.0
        ft = np.fft.fft(diff * self._f_in_phase) * self._f_out_phase / self.beta
        return ft + ref_tau

    def _fermion_tau_to_iwn(self, f_tau):
        """Transform fermionic f(tau_j) -> f(iwn_fft) on midpoint grid."""
        return self.beta * np.fft.ifft(f_tau * self._fi_in_phase) * self._fi_out_phase

    def _boson_tau_to_iOm_fft(self, f_tau):
        """Transform bosonic f(tau_j) -> f(iOm_m) on full [-Ntau/2, Ntau/2) grid."""
        return self.beta * np.fft.ifft(f_tau * self._bi_in_phase) * self._bi_out_phase

    def _extract_small_f(self, arr_fft):
        s0 = self.fft_offset
        return arr_fft[s0 : s0 + 2 * self.Nf]

    def _extract_small_b(self, arr_fft):
        m0 = self.Ntau // 2
        return arr_fft[m0 - self.Nb : m0 + self.Nb + 1]

    # ---------- Step 1: Analytical QMC-Free Impurity Closure (mIPT-4 + Local-TPSC) ----------

    def solve_impurity(self, Delta_small, mu, n_target, max_bkin_iter=20):
        """
        Fourth-spectral-moment interpolating perturbation theory (mIPT-4) with
        correlated kinetic band shift Delta B_kin, tail-subtracted Galitskii-Migdal
        D_imp (Eq. 2.64), and (s, c, pp) local-TPSC susceptibilities (Eqs. 2.44, 2.62-2.63).
        """
        U = self.U
        beta = self.beta
        n_val = float(n_target)
        half_fill = abs(n_val - 1.0) < 1e-12
        Del_fft = self._interp_delta_to_fft(Delta_small)

        # 1. Determine shifted Weiss chemical potential mu0 via n(G0_tilde) = n_target
        if half_fill:
            mu0 = 0.0
        else:
            def dens_G0(m0):
                G0 = 1.0 / (self.iwn_fft + m0 - Del_fft)
                G0_ref = 1.0 / (self.iwn_fft + m0)
                bx = -beta * m0
                f_ref = 1.0 / (1.0 + np.exp(np.clip(bx, -500.0, 500.0)))
                n_est = 2.0 * (f_ref + float(np.sum(np.real(G0 - G0_ref))) / beta)
                return n_est - n_val

            lo, hi = -max(15.0 * max(self.t, 1.0), 2.0 * U + 10.0), max(15.0 * max(self.t, 1.0), 2.0 * U + 10.0)
            while dens_G0(lo) > 0.0 and lo > -1e4:
                lo *= 2.0
            while dens_G0(hi) < 0.0 and hi < 1e4:
                hi *= 2.0
            mu0 = float(brentq(dens_G0, lo, hi, xtol=1e-12))

        G0_fft = 1.0 / (self.iwn_fft + mu0 - Del_fft)

        # 2. Second-order sunset Sigma^(2)(iwn)
        if self.t == 0.0:
            Sig2_fft = (U ** 2) * (0.5 * n_val) * (1.0 - 0.5 * n_val) / (self.iwn_fft + mu0)
            G0_tau = np.real(self._fermion_iwn_to_tau(G0_fft, c1=1.0, xi=-mu0))
        else:
            G0_tau = np.real(self._fermion_iwn_to_tau(G0_fft, c1=1.0, xi=-mu0))
            G0_rev = G0_tau[::-1]
            Sig2_tau = (U ** 2) * (G0_tau ** 2) * G0_rev
            Sig2_fft = self._fermion_tau_to_iwn(Sig2_tau)
            Sig2_fft = 0.5 * (Sig2_fft + np.conj(Sig2_fft[::-1]))
            if half_fill:
                Sig2_fft = 1j * np.imag(Sig2_fft)

        # 3. Fourth-moment mIPT-4 self-energy with self-consistent Delta B_kin (Eqs. 2.57-2.60)
        denom_b = (U ** 2) * (0.5 * n_val) * (1.0 - 0.5 * n_val)
        Delta_B_kin = 0.0
        use_bkin = (self.level >= 3) and (not half_fill) and (U > 1e-12) and (self.t > 0.0)

        for _ in range(max_bkin_iter if use_bkin else 1):
            if U < 1e-12 or denom_b < 1e-14:
                Sig_dyn_fft = np.zeros_like(self.iwn_fft)
            else:
                B_ipt = (U * (1.0 - 0.5 * n_val) - mu + mu0 + Delta_B_kin) / denom_b
                Sig_dyn_fft = Sig2_fft / (1.0 - B_ipt * Sig2_fft)
            Sig_imp_fft = 0.5 * U * n_val + Sig_dyn_fft
            g_fft = 1.0 / (self.iwn_fft + mu - Del_fft - Sig_imp_fft)
            if not use_bkin:
                break
            bracket = ((2.0 / U) * Sig_imp_fft - 1.0) * g_fft - (n_val - 1.0) * G0_fft
            dB_new = float(np.sum(np.real(Del_fft * bracket)) / beta) / (0.5 * n_val * (1.0 - 0.5 * n_val))
            if abs(dB_new - Delta_B_kin) < 1e-11:
                Delta_B_kin = dB_new
                break
            Delta_B_kin = 0.5 * Delta_B_kin + 0.5 * dB_new

        # Impurity density with tail subtraction
        xi_hf = 0.5 * U * n_val - mu
        g_ref_fft = 1.0 / (self.iwn_fft - xi_hf)
        bx_hf = beta * xi_hf
        f_hf = 1.0 / (1.0 + np.exp(np.clip(bx_hf, -500.0, 500.0)))
        n_imp = 2.0 * (f_hf + float(np.sum(np.real(g_fft - g_ref_fft))) / beta)

        # 4. Tail-subtracted Galitskii-Migdal double occupancy (Eq. 2.64)
        if U < 1e-12:
            D_imp = 0.25 * n_val * n_imp
        else:
            c2 = (U ** 2) * (0.5 * n_val) * (1.0 - 0.5 * n_val)
            alpha = max(0.5 * U, self.t, 0.5)
            tail_sum = -c2 * np.tanh(0.5 * beta * alpha) / (2.0 * alpha * U)
            tail_term = c2 / ((self.iwn_fft ** 2) - alpha ** 2)
            corr = float(np.sum(np.real(Sig_dyn_fft * g_fft - tail_term)) / (beta * U))
            D_imp = 0.25 * n_val * n_imp + tail_sum + corr
            D_imp = float(np.clip(D_imp, max(0.0, n_val - 1.0), 0.5 * n_val))

        # 5. Local Weiss bubbles & local-TPSC (s, c, pp) susceptibilities (Eqs. 2.44, 2.62-2.63)
        m0_fft = self.Ntau // 2
        if self.t == 0.0:
            chi0_G0_fft = np.zeros(self.Ntau, dtype=float)
            chi0_G0_fft[m0_fft] = beta * (0.5 * n_val) * (1.0 - 0.5 * n_val)
            chi0_pp_G0_fft = (1.0 - n_val) / (-1j * self.Om_b_fft - 2.0 * mu0 + 1e-30)
            if half_fill:
                chi0_pp_G0_fft = np.zeros(self.Ntau, dtype=np.complex128)
                chi0_pp_G0_fft[m0_fft] = 0.25 * beta
        else:
            G0_rev = G0_tau[::-1]
            chi0_G0_fft = np.maximum(np.real(self._boson_tau_to_iOm_fft(G0_tau * G0_rev)), 0.0)
            chi0_pp_G0_fft = self._boson_tau_to_iOm_fft(G0_tau ** 2)
            if half_fill:
                chi0_pp_G0_fft = np.real(chi0_pp_G0_fft) + 0.0j

        # Spin local-TPSC
        target_s = n_val - 2.0 * D_imp
        c0_max = float(np.max(chi0_G0_fft))
        if U < 1e-12 or c0_max < 1e-14:
            U_sp = 0.0
        elif self.t == 0.0:
            U_sp = (1.0 - 2.0 * chi0_G0_fft[m0_fft] / max(beta * target_s, 1e-30)) / chi0_G0_fft[m0_fft]
        else:
            def sum_s(usp):
                return float(np.sum(2.0 * chi0_G0_fft / (1.0 - usp * chi0_G0_fft)) / beta) - target_s
            u_hi = (1.0 - 1e-10) / c0_max
            u_lo = 0.0 if sum_s(0.0) <= 0.0 else -10.0 * U
            if sum_s(u_lo) * sum_s(u_hi) <= 0.0:
                U_sp = float(brentq(sum_s, u_lo, u_hi, xtol=1e-11))
            else:
                U_sp = float(U)
        chi_imp_s_fft = 2.0 * chi0_G0_fft / (1.0 - U_sp * chi0_G0_fft)

        # Charge local-TPSC
        target_c = n_val * (1.0 - n_val) + 2.0 * D_imp
        if U < 1e-12 or c0_max < 1e-14:
            U_ch = 0.0
        elif self.t == 0.0:
            if target_c > 1e-18:
                U_ch = (2.0 * chi0_G0_fft[m0_fft] / (beta * target_c) - 1.0) / chi0_G0_fft[m0_fft]
            else:
                U_ch = 1e12
        else:
            def sum_c(uch):
                return float(np.sum(2.0 * chi0_G0_fft / (1.0 + uch * chi0_G0_fft)) / beta) - target_c
            u_lo = 0.0 if sum_c(0.0) >= 0.0 else -(1.0 - 1e-9) / c0_max
            u_hi = max(10.0 * U, 50.0)
            while sum_c(u_hi) > 0.0 and u_hi < 1e9:
                u_hi *= 4.0
            if sum_c(u_lo) * sum_c(u_hi) <= 0.0:
                U_ch = float(brentq(sum_c, u_lo, u_hi, xtol=1e-11))
            else:
                U_ch = float(U)
        chi_imp_c_fft = 2.0 * chi0_G0_fft / (1.0 + U_ch * chi0_G0_fft)

        # Singlet particle-particle local-TPSC (Eq. 2.43-2.44)
        target_pp = 0.5 * (1.0 - n_val) + D_imp
        if U < 1e-12:
            U_pp = 0.0
        elif self.t == 0.0:
            if half_fill:
                c_pp0 = float(np.real(chi0_pp_G0_fft[m0_fft]))
                U_pp = (c_pp0 / max(beta * target_pp, 1e-30) - 1.0) / max(c_pp0, 1e-30)
            else:
                U_pp = (U - 2.0 * mu + 2.0 * mu0) / (1.0 - n_val)
        else:
            def sum_pp(upp):
                x_pp = chi0_pp_G0_fft / (1.0 + upp * chi0_pp_G0_fft)
                if half_fill:
                    return float(np.sum(np.real(x_pp)) / beta) - target_pp
                E_pp = -2.0 * mu0 + upp * (1.0 - n_val)
                if abs(E_pp) < 1e-8:
                    E_pp = 1e-8 if E_pp >= 0 else -1e-8
                ref_iOm = (1.0 - n_val) / (-1j * self.Om_b_fft + E_pp)
                bE = 0.5 * beta * E_pp
                coth_val = np.sign(bE) if abs(bE) > 300.0 else 1.0 / np.tanh(bE)
                ref_sum = 0.5 * (1.0 - n_val) * coth_val
                return ref_sum + float(np.sum(np.real(x_pp - ref_iOm)) / beta) - target_pp

            c_pp0 = float(np.real(chi0_pp_G0_fft[m0_fft]))
            u_lo = -(1.0 - 1e-7) / max(c_pp0, 1e-12) if c_pp0 > 0 else -10.0
            u_hi = max(10.0 * U, 50.0)
            while sum_pp(u_hi) > 0.0 and u_hi < 1e9:
                u_hi *= 4.0
            if sum_pp(u_lo) * sum_pp(u_hi) <= 0.0:
                U_pp = float(brentq(sum_pp, u_lo, u_hi, xtol=1e-11))
            else:
                U_pp = float(U)
        chi_imp_pp_fft = chi0_pp_G0_fft / (1.0 + U_pp * chi0_pp_G0_fft)

        # 6. Local bubbles on full-Ntau physical g(iwn) and Ward factor Lambda(nu) (Eq. 2.17)
        Lam_fft = np.maximum(1.0 - np.imag(Sig_imp_fft) / self.wn_fft, 1.0)
        sqrt_Lam_fft = np.sqrt(Lam_fft)
        g_tau = np.real(self._fermion_iwn_to_tau(g_fft, c1=1.0, xi=xi_hf))
        Lam_g_tau = np.real(self._fermion_iwn_to_tau(sqrt_Lam_fft * g_fft, c1=1.0, xi=xi_hf))

        chi0_loc_fft = np.maximum(np.real(self._boson_tau_to_iOm_fft(g_tau * g_tau[::-1])), 0.0)
        chi_Lam_loc_fft = np.maximum(np.real(self._boson_tau_to_iOm_fft(Lam_g_tau * Lam_g_tau[::-1])), 0.0)
        chi_pp_Lam_loc_fft = self._boson_tau_to_iOm_fft(Lam_g_tau ** 2)
        if half_fill:
            chi_pp_Lam_loc_fft = np.real(chi_pp_Lam_loc_fft) + 0.0j

        g_sq_sum = float(np.real(-np.sum(g_fft * g_fft) / beta))
        Lam_g_sq_sum = float(np.real(-np.sum(Lam_fft * g_fft * g_fft) / beta))
        phi1_fft = Lam_fft - 1.0
        X0_00 = g_sq_sum
        X0_01 = float(np.real(-np.sum(phi1_fft * g_fft * g_fft) / beta))
        X0_11 = float(np.real(-np.sum((phi1_fft ** 2) * g_fft * g_fft) / beta))
        X0_loc_0 = np.array([[X0_00, X0_01], [X0_01, X0_11]], dtype=float)

        chi0_loc = self._extract_small_b(chi0_loc_fft).copy()
        chi_Lam_loc = self._extract_small_b(chi_Lam_loc_fft).copy()
        chi0_loc[self.Nb] = max(g_sq_sum, 0.0)
        chi_Lam_loc[self.Nb] = max(Lam_g_sq_sum, 0.0)
        b_loc = np.stack([chi0_loc, chi_Lam_loc - chi0_loc], axis=1)

        return {
            "mu0": mu0,
            "g": self._extract_small_f(g_fft),
            "Sig_imp": self._extract_small_f(Sig_imp_fft),
            "Sig_dyn": self._extract_small_f(Sig_dyn_fft),
            "Lam_small": self._extract_small_f(Lam_fft),
            "n_imp": n_imp,
            "D_imp": D_imp,
            "Delta_B_kin": Delta_B_kin,
            "U_sp": U_sp,
            "U_ch": U_ch,
            "U_pp": U_pp,
            "chi0_G0": self._extract_small_b(chi0_G0_fft),
            "chi0_pp_G0": self._extract_small_b(chi0_pp_G0_fft),
            "chi_imp_s": self._extract_small_b(chi_imp_s_fft),
            "chi_imp_c": self._extract_small_b(chi_imp_c_fft),
            "chi_imp_pp": self._extract_small_b(chi_imp_pp_fft),
            "chi0_loc": chi0_loc,
            "chi_Lam_loc": chi_Lam_loc,
            "chi_pp_Lam_loc": self._extract_small_b(chi_pp_Lam_loc_fft),
            "b_loc": b_loc,
            "X0_loc_0": X0_loc_0,
            "Sig_imp_fft": Sig_imp_fft,
            "g_fft": g_fft,
        }

    # ---------- Step 2: UV-Regularized Rank-2 Local Vertex Closure ----------

    def compute_local_vertex(self, imp):
        """
        UV-tail-regularized Rank-2 Ward-constrained local vertices W_s^reg, W_c^reg
        (Eqs. 2.25-2.29) and singlet pair vertex gamma_pp^reg (Eq. 2.45).
        """
        n_om = 2 * self.Nb + 1
        m0 = self.Nb
        c0 = imp["chi0_loc"]
        cL = imp["chi_Lam_loc"]
        cG0 = imp["chi0_G0"]
        U_sp, U_ch, U_pp = imp["U_sp"], imp["U_ch"], imp["U_pp"]

        denom_L = np.where(np.abs(cL) > 1e-14, cL, 1e-14)
        ratio_sq = (cG0 / denom_L) ** 2

        gamma_s = np.empty(n_om, dtype=float)
        if self.U < 1e-12:
            gamma_s[:] = 0.0
        else:
            gamma_s[:] = (-U_sp / np.maximum(1.0 - U_sp * cG0, 1e-12)) * ratio_sq
            gamma_s[m0] = (2.0 * c0[m0] - imp["chi_imp_s"][m0]) / max(2.0 * cL[m0] ** 2, 1e-18)
            gamma_s = np.minimum(gamma_s, 0.0)

        gamma_c = np.empty(n_om, dtype=float)
        if self.U < 1e-12:
            gamma_c[:] = 0.0
        else:
            gamma_c[:] = (+U_ch / np.maximum(1.0 + U_ch * cG0, 1e-12)) * ratio_sq
            gamma_c[m0] = (2.0 * c0[m0] - imp["chi_imp_c"][m0]) / max(2.0 * cL[m0] ** 2, 1e-18)

        theta = np.clip(c0 / np.maximum(cL, 1e-14), 1e-6, 1.0)
        norm_vp = theta ** 2 + (1.0 - theta) ** 2
        gamma_c_perp = np.zeros(n_om, dtype=float)
        if self.level >= 3 and self.U > 1e-12:
            gamma_c_perp[:] = (self.U / norm_vp) * ratio_sq
            gamma_c_perp[m0] = self.U / norm_vp[m0]

        W_c = np.zeros((n_om, 2, 2), dtype=float)
        for im in range(n_om):
            th = theta[im]
            vp = np.array([1.0 - th, -th], dtype=float)
            W_c[im] = gamma_c[im] * np.ones((2, 2)) + gamma_c_perp[im] * np.outer(vp, vp)

        D_c_irr = float(np.linalg.det(np.eye(2) - imp["X0_loc_0"] @ W_c[m0]))

        chi0_pp = imp["chi0_pp_G0"]
        denom_pp = np.where(np.abs(imp["chi_pp_Lam_loc"]) > 1e-12, imp["chi_pp_Lam_loc"], 1e-12)
        ratio_pp_sq = (chi0_pp / denom_pp) ** 2
        if self.level >= 3 and self.U > 1e-12:
            gamma_pp = (U_pp / (1.0 + U_pp * chi0_pp)) * ratio_pp_sq
        else:
            gamma_pp = np.zeros(n_om, dtype=np.complex128)

        return {
            "gamma_s": gamma_s,
            "gamma_c": gamma_c,
            "gamma_c_perp": gamma_c_perp,
            "W_c": W_c,
            "gamma_pp": gamma_pp,
            "D_c_irr": D_c_irr,
        }

    # ---------- Step 3: 3D-FFT Bubbles with Exact Local Restoration ----------

    def compute_bubbles(self, G_tilde, G_lat, imp):
        """
        Evaluate 2x2 dual bubble X_tilde(q, iOm), scalar Ward bubble chi_t_L2(q, iOm),
        lattice bubbles (chi0_lat, b_lat, chiL_lat) with exact r=0 restoration,
        and dual singlet pair bubble chi_pp_tilde(P, iOm) via 3D FFTs.
        """
        Nf, Nb, L, beta = self.Nf, self.Nb, self.L, self.beta
        n_om = 2 * Nb + 1
        Lam = imp["Lam_small"]
        s_Lam = np.sqrt(np.maximum(Lam, 0.0))

        G0_r = np.fft.ifftn(G_lat, axes=(1, 2, 3))
        G1_r = np.fft.ifftn(s_Lam[:, None, None, None] * G_lat, axes=(1, 2, 3))

        Gt0_r = np.fft.ifftn(G_tilde, axes=(1, 2, 3))
        Gt1_r = np.fft.ifftn(s_Lam[:, None, None, None] * G_tilde, axes=(1, 2, 3))
        Gt2_r = np.fft.ifftn(Lam[:, None, None, None] * G_tilde, axes=(1, 2, 3))

        G0_nr = self._neg_r(G0_r)
        G1_nr = self._neg_r(G1_r)
        Gt0_nr = self._neg_r(Gt0_r)
        Gt1_nr = self._neg_r(Gt1_r)
        Gt2_nr = self._neg_r(Gt2_r)

        chi0_lat = np.zeros((n_om, L, L, L), dtype=float)
        chiL_lat = np.zeros((n_om, L, L, L), dtype=float)
        Xt_0 = np.zeros((n_om, L, L, L), dtype=float)
        Xt_L = np.zeros((n_om, L, L, L), dtype=float)
        Xt_L2 = np.zeros((n_om, L, L, L), dtype=float)
        chi_pp_tilde = np.zeros((n_om, L, L, L), dtype=np.complex128)

        for im, m in enumerate(self.m_b):
            i_lo = max(0, -m)
            i_hi = min(2 * Nf, 2 * Nf - m)
            if i_hi > i_lo:
                s1 = slice(i_lo, i_hi)
                s2 = slice(i_lo + m, i_hi + m)
                c0_r = -np.sum(G0_nr[s1] * G0_r[s2], axis=0) / beta
                cL_r = -np.sum(G1_nr[s1] * G1_r[s2], axis=0) / beta
                chi0_lat[im] = np.real(np.fft.fftn(c0_r))
                chiL_lat[im] = np.real(np.fft.fftn(cL_r))

                x0_r = -np.sum(Gt0_nr[s1] * Gt0_r[s2], axis=0) / beta
                xL_r = -np.sum(Gt1_nr[s1] * Gt1_r[s2], axis=0) / beta
                xL2_r = -np.sum(Gt2_nr[s1] * Gt2_r[s2], axis=0) / beta
                Xt_0[im] = np.real(np.fft.fftn(x0_r))
                Xt_L[im] = np.real(np.fft.fftn(xL_r))
                Xt_L2[im] = np.real(np.fft.fftn(xL2_r))

            p_lo = max(0, m)
            p_hi = min(2 * Nf, 2 * Nf + m)
            if p_hi > p_lo:
                i_idx = np.arange(p_lo, p_hi)
                j_idx = 2 * Nf - 1 + m - i_idx
                pp_r = np.sum(Gt2_r[i_idx] * Gt2_r[j_idx], axis=0) / beta
                chi_pp_tilde[im] = np.fft.fftn(pp_r)

        chi0_lat += (imp["chi0_loc"] - np.mean(chi0_lat, axis=(1, 2, 3)))[:, None, None, None]
        chiL_lat += (imp["chi_Lam_loc"] - np.mean(chiL_lat, axis=(1, 2, 3)))[:, None, None, None]
        chi0_lat = np.maximum(chi0_lat, 0.0)
        chiL_lat = np.maximum(chiL_lat, 0.0)

        Xt_0 -= np.mean(Xt_0, axis=(1, 2, 3), keepdims=True)
        Xt_L -= np.mean(Xt_L, axis=(1, 2, 3), keepdims=True)
        Xt_L2 -= np.mean(Xt_L2, axis=(1, 2, 3), keepdims=True)
        chi_pp_tilde -= np.mean(chi_pp_tilde, axis=(1, 2, 3), keepdims=True)

        X_tilde = np.empty((n_om, 2, 2, L, L, L), dtype=float)
        X_tilde[:, 0, 0] = Xt_0
        X_tilde[:, 0, 1] = Xt_L - Xt_0
        X_tilde[:, 1, 0] = X_tilde[:, 0, 1]
        X_tilde[:, 1, 1] = Xt_L2 - 2.0 * Xt_L + Xt_0

        b_lat = np.empty((n_om, 2, L, L, L), dtype=float)
        b_lat[:, 0] = chi0_lat
        b_lat[:, 1] = chiL_lat - chi0_lat

        return X_tilde, Xt_L2, chi0_lat, b_lat, chiL_lat, chi_pp_tilde

    # ---------- Step 4: Non-Singular Moriya Spin & Safeguarded 2x2 Charge Ladders ----------

    def solve_moriya_and_charge(self, vert, X_tilde, chi_t_L2, chi0_lat, b_lat, chiL_lat, n_val):
        """Level-III Moriya lambda_s via UV-tail-free excess Pauli sum rule (Eq. 2.41)."""
        W_c = vert["W_c"]
        Xt_T = np.transpose(X_tilde, (0, 3, 4, 5, 1, 2))
        M_c = np.eye(2)[None, None, None, None, :, :] + np.einsum(
            "...ab,...bc->...ac", Xt_T, W_c[:, None, None, None, :, :]
        )

        tr = M_c[..., 0, 0] + M_c[..., 1, 1]
        det = M_c[..., 0, 0] * M_c[..., 1, 1] - M_c[..., 0, 1] * M_c[..., 1, 0]
        disc = np.maximum(tr ** 2 - 4.0 * det, 0.0)
        lam_min_c = float(np.min(0.5 * (tr - np.sqrt(disc))))
        lam_c_floor = max(0.0, max(self.eps_c, 0.05, 1e-6 * abs(lam_min_c)) - lam_min_c)

        b_T = np.transpose(b_lat, (0, 2, 3, 4, 1))
        Wc_b = W_c[:, None, None, None, :, :]

        def _eval_charge(lam_c_val):
            M_reg = M_c.copy()
            M_reg[..., 0, 0] += lam_c_val
            M_reg[..., 1, 1] += lam_c_val
            det_reg = M_reg[..., 0, 0] * M_reg[..., 1, 1] - M_reg[..., 0, 1] * M_reg[..., 1, 0]
            M_inv = np.empty_like(M_reg)
            M_inv[..., 0, 0] = M_reg[..., 1, 1] / det_reg
            M_inv[..., 1, 1] = M_reg[..., 0, 0] / det_reg
            M_inv[..., 0, 1] = -M_reg[..., 0, 1] / det_reg
            M_inv[..., 1, 0] = -M_reg[..., 1, 0] / det_reg
            Gc = np.einsum("...ab,...bc->...ac", Wc_b, M_inv)
            bGb_val = np.einsum("...a,...ab,...b->...", b_T, Gc, b_T)
            xc = 2.0 * chi0_lat - 2.0 * bGb_val
            return Gc, bGb_val, xc

        tilde_lam_c = lam_c_floor
        Gamma_c_mat, bGb, chi_c = _eval_charge(tilde_lam_c)
        if float(np.min(chi_c)) < 0.0:
            lo_c = tilde_lam_c
            hi_c = max(lo_c + 1.0, 2.0 * max(abs(lo_c), 1.0))
            while float(np.min(_eval_charge(hi_c)[2])) < 0.0 and hi_c < 1e8:
                hi_c *= 2.0
            if float(np.min(_eval_charge(hi_c)[2])) >= 0.0:
                tilde_lam_c = float(
                    brentq(lambda lc: float(np.min(_eval_charge(lc)[2])), lo_c, hi_c, xtol=1e-6)
                ) + 1e-5
            else:
                tilde_lam_c = hi_c
            Gamma_c_mat, bGb, chi_c = _eval_charge(tilde_lam_c)

        rhs_excess = float(2.0 * np.sum(bGb) / (self.beta * self.N))
        gs = vert["gamma_s"][:, None, None, None]
        D_s = 1.0 + gs * chi_t_L2
        d_min = float(np.min(D_s))
        lam_min_s = -d_min + max(1e-8, 1e-6 * abs(d_min))

        def spin_excess_denom(lam_tilde):
            Gs = gs / (D_s + lam_tilde)
            lhs = float(-2.0 * np.sum(Gs * (chiL_lat ** 2)) / (self.beta * self.N))
            return lhs - rhs_excess

        if self.U < 1e-12 or rhs_excess <= 0.0:
            lam_s = max(0.0, lam_min_s)
            mode = "ladder-denom"
        else:
            lam_lo = max(0.0, lam_min_s)
            if spin_excess_denom(lam_lo) < 0.0:
                lam_lo = lam_min_s
            lam_hi = max(lam_lo + 1.0, 10.0 * max(abs(lam_lo), 1.0))
            while spin_excess_denom(lam_hi) > 0.0 and lam_hi < 1e16:
                lam_hi *= 4.0
            if spin_excess_denom(lam_lo) * spin_excess_denom(lam_hi) <= 0.0:
                lam_s = float(brentq(spin_excess_denom, lam_lo, lam_hi))
                mode = "ladder-denom"
            else:
                lam_s = max(0.0, lam_min_s)
                mode = "ladder-denom-clamp"

        Gamma_s_lam = gs / (D_s + lam_s)
        chi_s_lam = 2.0 * chi0_lat - 2.0 * Gamma_s_lam * (chiL_lat ** 2)

        return {
            "lam_s": lam_s,
            "mode_s": mode,
            "lam_min_c": lam_min_c,
            "tilde_lam_c": tilde_lam_c,
            "Gamma_s_lam": Gamma_s_lam,
            "Gamma_c_mat": Gamma_c_mat,
            "chi_s_lam": chi_s_lam,
            "chi_c": chi_c,
        }

    # ---------- Step 5: Three-Channel (s + c + pp) Dual Self-Energy ----------

    def compute_dual_self_energy(self, G_tilde, imp, vert, moriya, X_tilde, chi_t_L2, chi_pp_tilde):
        """
        Evaluate three-channel Level-III dual self-energy (Eqs. 2.46-2.49):
          Sigma_tilde = Sigma_tilde_ph + Sigma_tilde_pp
        Returns (Sig_tot, Sig_ph, Sig_pp, V_ph).
        """
        Nf, Nb, L, beta = self.Nf, self.Nb, self.L, self.beta
        Lam = imp["Lam_small"]
        s_Lam = np.sqrt(np.maximum(Lam, 0.0))

        gs = vert["gamma_s"][:, None, None, None]
        Gs = moriya["Gamma_s_lam"]
        V_s = 1.5 * gs * (Gs - 0.5 * gs) * chi_t_L2

        Wc = vert["W_c"][:, None, None, None, :, :]
        Xt = np.transpose(X_tilde, (0, 3, 4, 5, 1, 2))
        Gc = moriya["Gamma_c_mat"]
        dGc = Gc - 0.5 * Wc

        WX = np.einsum("...ab,...bc->...ac", Wc, Xt)
        XW = np.einsum("...ab,...bc->...ac", Xt, Wc)
        V_c = 0.25 * (
            np.einsum("...ab,...bc->...ac", WX, dGc)
            + np.einsum("...ab,...bc->...ac", dGc, XW)
        )
        V_ph = V_s[..., None, None] * np.ones((2, 2)) + V_c
        V_off = 0.5 * (V_ph[..., 0, 1] + V_ph[..., 1, 0])
        V_ph[..., 0, 1] = V_off
        V_ph[..., 1, 0] = V_off

        Vt_00 = V_ph[..., 0, 0] - 2.0 * V_off + V_ph[..., 1, 1]
        Vt_01 = V_off - V_ph[..., 1, 1]
        Vt_11 = V_ph[..., 1, 1]

        Vt00_r = np.fft.ifftn(Vt_00, axes=(1, 2, 3))
        Vt01_r = np.fft.ifftn(Vt_01, axes=(1, 2, 3))
        Vt11_r = np.fft.ifftn(Vt_11, axes=(1, 2, 3))

        Gt0_r = np.fft.ifftn(G_tilde, axes=(1, 2, 3))
        Gt1_r = np.fft.ifftn(s_Lam[:, None, None, None] * G_tilde, axes=(1, 2, 3))
        Gt2_r = np.fft.ifftn(Lam[:, None, None, None] * G_tilde, axes=(1, 2, 3))

        Vt00_nr = self._neg_r(Vt00_r)
        Vt01_nr = self._neg_r(Vt01_r)
        Vt11_nr = self._neg_r(Vt11_r)

        Sig_ph = np.zeros_like(G_tilde)
        for iw in range(2 * Nf):
            m_lo = max(-Nb, -iw)
            m_hi = min(Nb, 2 * Nf - 1 - iw)
            if m_hi >= m_lo:
                im_sl = slice(m_lo + Nb, m_hi + Nb + 1)
                iw2_sl = slice(iw + m_lo, iw + m_hi + 1)
                acc_r = (
                    np.sum(Gt0_r[iw2_sl] * Vt00_nr[im_sl], axis=0)
                    + (2.0 * s_Lam[iw]) * np.sum(Gt1_r[iw2_sl] * Vt01_nr[im_sl], axis=0)
                    + Lam[iw] * np.sum(Gt2_r[iw2_sl] * Vt11_nr[im_sl], axis=0)
                )
                Sig_ph[iw] = np.fft.fftn(acc_r) / beta

        Sig_pp = np.zeros_like(G_tilde)
        if self.level >= 3 and self.U > 1e-12:
            gpp = vert["gamma_pp"][:, None, None, None]
            denom_pp = 1.0 + gpp * chi_pp_tilde
            lam_pp = max(0.0, self.eps_pp - float(np.min(np.real(denom_pp))))
            K_pp = (gpp ** 3) * (chi_pp_tilde ** 2) / (denom_pp + lam_pp)
            K_pp_r = np.fft.ifftn(K_pp, axes=(1, 2, 3))
            Gt2_nr = self._neg_r(Gt2_r)
            for iw in range(2 * Nf):
                m_lo = max(-Nb, iw - (2 * Nf - 1))
                m_hi = min(Nb, iw)
                if m_hi >= m_lo:
                    im_idx = np.arange(m_lo + Nb, m_hi + Nb + 1)
                    iw2_idx = (2 * Nf - 1) + np.arange(m_lo, m_hi + 1) - iw
                    acc_r = np.sum(Gt2_nr[iw2_idx] * K_pp_r[im_idx], axis=0)
                    Sig_pp[iw] = Lam[iw] * (np.fft.fftn(acc_r) / beta)

        return Sig_ph + Sig_pp, Sig_ph, Sig_pp, V_ph

    # ---------- Step 6: Hartree-Tail-Subtracted Chemical Potential ----------

    def find_chemical_potential(self, Sig_lat):
        """Find mu satisfying Eq. (2.65) n(mu) = 1 - delta via Brent root-finding."""
        if abs(self.delta) < 1e-12:
            return 0.5 * self.U

        beta = self.beta
        Sig_hf = 0.5 * self.U * self.n_target
        eps = self.eps_k[None, :, :, :]
        iwn = self.iwn[:, None, None, None]

        def n_minus_target(mu_try):
            xi_ref = self.eps_k + Sig_hf - mu_try
            f_ref = 1.0 / (1.0 + np.exp(np.clip(beta * xi_ref, -500.0, 500.0)))
            G_lat = 1.0 / (iwn + mu_try - eps - Sig_lat)
            G_ref = 1.0 / (iwn - xi_ref[None, :, :, :])
            corr = np.sum(np.real(G_lat - G_ref), axis=0) / beta
            n_lat = float(2.0 * np.mean(f_ref + corr))
            return n_lat - self.n_target

        lo, hi = self.mu - 4.0, self.mu + 4.0
        while n_minus_target(lo) > 0.0 and lo > -200.0:
            lo -= 4.0
        while n_minus_target(hi) < 0.0 and hi < 200.0:
            hi += 4.0
        if n_minus_target(lo) * n_minus_target(hi) <= 0.0:
            return float(brentq(n_minus_target, lo, hi, xtol=1e-10))
        return float(self.mu)

    # ---------- Diagnostics & Linearized 3D O_h E_g Pairing Eigenvalue ----------

    def matsubara_diagnostics(self):
        """Compute Matsubara observables, Theorem 5.1 causality bounds, and R_opt/th."""
        imp = self.last_imp
        vert = self.last_vert
        moriya = self.last_moriya
        Sig_lat = self.last_Sig_lat
        G_lat = self.last_G_lat

        w0_idx = self.Nf
        w0 = self.wn[w0_idx]

        Sig_hf = 0.5 * self.U * self.n_target
        xi_ref = self.eps_k + Sig_hf - self.mu
        f_ref = 1.0 / (1.0 + np.exp(np.clip(self.beta * xi_ref, -500.0, 500.0)))
        G_ref = 1.0 / (self.iwn[:, None, None, None] - xi_ref[None, :, :, :])
        n_lat = float(2.0 * np.mean(f_ref + np.sum(np.real(G_lat - G_ref), axis=0) / self.beta))

        Z_imp = float(1.0 / (1.0 - np.imag(imp["Sig_imp"][w0_idx]) / w0))
        Z_k = 1.0 / np.maximum(1.0 - np.imag(Sig_lat[w0_idx]) / w0, 1e-12)

        i_pi, i_half = self.L // 2, self.L // 4
        Z_hot = float(Z_k[0, i_pi, i_half])
        Z_cold = float(Z_k[i_half, i_half, i_half])

        dos_w = np.maximum(-np.imag(G_lat[w0_idx]), 0.0)
        dos_sum = float(np.sum(dos_w))
        v2_w = dos_w * self.v2
        v2_sum = float(np.sum(v2_w))
        if dos_sum > 1e-18 and v2_sum > 1e-18:
            Z_v2 = float(np.sum(v2_w * Z_k) / v2_sum)
            Zinv_dos = float(np.sum(dos_w / Z_k) / dos_sum)
            R_opt_th_mat = float(1.0 / max(Z_v2 * Zinv_dos, 1e-18))
        else:
            R_opt_th_mat = 1.0

        im_st0 = -np.imag(self.Sigma_tilde[w0_idx])
        g0_abs = float(np.abs(np.imag(imp["g"][w0_idx])))
        fs_mask = np.abs(self.eps_k - (self.mu - 0.5 * self.U * self.n_target)) <= max(1e-6, 0.35 * self.t)
        causal_fs = fs_mask & (im_st0 > 1e-10)
        if np.any(causal_fs):
            sig_vals = im_st0[causal_fs]
        else:
            causal_all = im_st0 > 1e-10
            sig_vals = im_st0[causal_all] if np.any(causal_all) else np.abs(im_st0)
        sig_max = float(np.max(sig_vals))
        sig_min = float(np.min(sig_vals))
        C0 = g0_abs * float(np.max(np.maximum(im_st0, 0.0)))
        aniso_ratio = sig_max / max(sig_min, 1e-15) if sig_max > 0.0 else 1.0
        C0_bound = 0.5 * aniso_ratio

        chi_AFM = float(moriya["chi_s_lam"][self.Nb, i_pi, i_pi, i_pi])

        return {
            "mu": float(self.mu),
            "n_lat": n_lat,
            "Z_imp": Z_imp,
            "Z_hot": Z_hot,
            "Z_cold": Z_cold,
            "R_opt_th_mat": R_opt_th_mat,
            "C0": float(C0),
            "C0_bound": float(C0_bound),
            "aniso_ratio": float(aniso_ratio),
            "D_c_irr": float(vert["D_c_irr"]),
            "lam_min_c": float(moriya["lam_min_c"]),
            "tilde_lam_c": float(moriya["tilde_lam_c"]),
            "lam_s": float(moriya["lam_s"]),
            "chi_AFM": chi_AFM,
            "D_imp": float(imp["D_imp"]),
            "Delta_B_kin": float(imp["Delta_B_kin"]),
        }

    def solve_Eg_pairing(self, max_iter=40, tol=1e-7):
        """
        O(N log N) 3D-FFT power iteration for the linearized 3D O_h E_g (d-wave)
        Eliashberg pairing eigenvalue lambda_{E_g} (Eqs. 2.51-2.52).
        """
        imp = self.last_imp
        vert = self.last_vert
        moriya = self.last_moriya
        G_lat = self.last_G_lat
        Nf, Nb, beta = self.Nf, self.Nb, self.beta
        Lam = imp["Lam_small"]

        V_sing = (
            -1.5 * moriya["Gamma_s_lam"]
            + 0.5 * moriya["Gamma_c_mat"][..., 0, 0]
            - 0.5 * (vert["gamma_c"] - vert["gamma_s"])[:, None, None, None]
        )
        V_sing_r = np.fft.ifftn(V_sing, axes=(1, 2, 3))
        G_sq = np.abs(G_lat) ** 2

        phi = (self.psi_d1[None, :, :, :] / (1.0 + self.wn[:, None, None, None] ** 2)).astype(np.complex128)
        phi = phi / np.linalg.norm(phi)
        ev = 0.0

        for _ in range(max_iter):
            W_k = Lam[:, None, None, None] * G_sq * phi
            W_r = np.fft.ifftn(W_k, axes=(1, 2, 3))
            phi_new = np.zeros_like(phi)
            for iw in range(2 * Nf):
                m_lo = max(-Nb, iw - (2 * Nf - 1))
                m_hi = min(Nb, iw)
                if m_hi >= m_lo:
                    im_idx = np.arange(m_lo + Nb, m_hi + Nb + 1)
                    iwp_idx = iw - np.arange(m_lo, m_hi + 1)
                    acc_r = np.sum(V_sing_r[im_idx] * W_r[iwp_idx], axis=0)
                    phi_new[iw] = -Lam[iw] * (np.fft.fftn(acc_r) / beta)

            phi_new = 0.5 * (phi_new - np.swapaxes(phi_new, 1, 2))
            phi_new = 0.5 * (phi_new + np.conj(phi_new[::-1]))
            ev_new = float(np.real(np.vdot(phi, phi_new)))
            nrm = float(np.linalg.norm(phi_new))
            if nrm < 1e-15:
                return 0.0
            phi = phi_new / nrm
            if abs(ev_new - ev) < tol:
                ev = ev_new
                break
            ev = ev_new
        return float(ev)

    # ---------- Step 7: Iterative O(N log N) Maki-Thompson Current Vertex Optics ----------

    def solve_maki_thompson_optics(self, m_opt_list=(0, 1, 2, 3), max_iter=25, tol=1e-6):
        """
        Solve iterative O(N log N) Maki-Thompson dual current vertex equation (Eqs. 4.23-4.27)
        and evaluate normal-state U(1) gauge diagnostic Delta_Ward.
        """
        if self.last_G_lat is None or self.last_V_ph is None:
            raise RuntimeError("Run the self-consistent solver first.")

        imp = self.last_imp
        Lam = imp["Lam_small"]
        s_Lam = np.sqrt(np.maximum(Lam, 0.0))
        g = imp["g"][:, None, None, None]
        G_lat = self.last_G_lat
        V_ph = self.last_V_ph
        Nf, Nb, beta = self.Nf, self.Nb, self.beta

        Sig_hf = 0.5 * self.U * self.n_target
        xi_ref = self.eps_k + Sig_hf - self.mu
        f_ref = 1.0 / (1.0 + np.exp(np.clip(beta * xi_ref, -500.0, 500.0)))
        G_ref = 1.0 / (self.iwn[:, None, None, None] - xi_ref[None, :, :, :])
        n_k = f_ref + np.sum(np.real(G_lat - G_ref), axis=0) / beta
        Tx = float(2.0 * np.mean((-2.0 * self.t * np.cos(self.kx)) * n_k))

        V_off = 0.5 * (V_ph[..., 0, 1] + V_ph[..., 1, 0])
        Vt_00 = V_ph[..., 0, 0] - 2.0 * V_off + V_ph[..., 1, 1]
        Vt_01 = V_off - V_ph[..., 1, 1]
        Vt_11 = V_ph[..., 1, 1]
        Vt00_nr = self._neg_r(np.fft.ifftn(Vt_00, axes=(1, 2, 3)))
        Vt01_nr = self._neg_r(np.fft.ifftn(Vt_01, axes=(1, 2, 3)))
        Vt11_nr = self._neg_r(np.fft.ifftn(Vt_11, axes=(1, 2, 3)))

        inv_pull = 1.0 / (1.0 + g * self.Sigma_tilde)
        Pi_bub = {}
        Pi_mt = {}

        for m_opt in m_opt_list:
            n_eff = 2 * Nf - m_opt
            if n_eff <= 0:
                continue
            s1, s2 = slice(0, n_eff), slice(m_opt, 2 * Nf)
            GG = G_lat[s1] * G_lat[s2]
            pull_prod = inv_pull[s1] * inv_pull[s2]
            GG_eff = GG * pull_prod

            vx_b = self.vx[None, :, :, :]
            Pi_bub[m_opt] = float(np.real(-2.0 * np.sum(vx_b * GG * vx_b) / (beta * self.N)))

            Lam_x = np.broadcast_to(vx_b, GG.shape).copy().astype(np.complex128)
            for _ in range(max_iter):
                J_k = GG_eff * Lam_x
                s1_L = s_Lam[s1][:, None, None, None]
                s2_L = s_Lam[s2][:, None, None, None]
                J0_r = np.fft.ifftn(J_k, axes=(1, 2, 3))
                J1_r = np.fft.ifftn(s1_L * J_k, axes=(1, 2, 3))
                J2_r = np.fft.ifftn(s2_L * J_k, axes=(1, 2, 3))
                J12_r = np.fft.ifftn((s1_L * s2_L) * J_k, axes=(1, 2, 3))

                dLam = np.zeros_like(Lam_x)
                for iw in range(n_eff):
                    m_lo = max(-Nb, -iw)
                    m_hi = min(Nb, n_eff - 1 - iw)
                    if m_hi >= m_lo:
                        im_sl = slice(m_lo + Nb, m_hi + Nb + 1)
                        iwp_sl = slice(iw + m_lo, iw + m_hi + 1)
                        acc_r = (
                            np.sum(Vt00_nr[im_sl] * J0_r[iwp_sl], axis=0)
                            + s_Lam[iw] * np.sum(Vt01_nr[im_sl] * J1_r[iwp_sl], axis=0)
                            + s_Lam[iw + m_opt] * np.sum(Vt01_nr[im_sl] * J2_r[iwp_sl], axis=0)
                            + (s_Lam[iw] * s_Lam[iw + m_opt]) * np.sum(Vt11_nr[im_sl] * J12_r[iwp_sl], axis=0)
                        )
                        dLam[iw] = np.fft.fftn(acc_r) / beta

                Lam_next = vx_b + pull_prod * dLam
                diff = float(np.max(np.abs(Lam_next - Lam_x)))
                Lam_x = 0.6 * Lam_x + 0.4 * Lam_next
                if diff < tol:
                    break

            Pi_mt[m_opt] = float(np.real(-2.0 * np.sum(vx_b * GG * Lam_x) / (beta * self.N)))

        Delta_Ward_bub = abs(Pi_bub[0] + Tx) / max(abs(Tx), 1e-12)
        Delta_Ward_mt = abs(Pi_mt[0] + Tx) / max(abs(Tx), 1e-12)
        return {
            "Tx": Tx,
            "Pi_bub": Pi_bub,
            "Pi_mt": Pi_mt,
            "Delta_Ward_bub": float(Delta_Ward_bub),
            "Delta_Ward_mt": float(Delta_Ward_mt),
        }

    # ---------- Real-Frequency Pade Continuation & Kubo Optics ----------

    def analytic_continuation_pade(self, omega_max=20.0, n_omega=401, eta=0.08, n_low=12, n_high=6, dps=50):
        """50-digit Vidberg-Serene Pade continuation of Sigma_dyn on the O_h IBZ."""
        if self.last_Sig_lat is None:
            raise RuntimeError("Run the self-consistent solver first.")
        omega = np.linspace(-omega_max, omega_max, int(n_omega))
        idx_nodes = select_pade_indices(self.Nf, n_low=n_low, n_high=n_high)
        z_nodes = 1j * self.wn[self.Nf + idx_nodes]
        z_eval = omega + 1j * eta
        Sigma_inf = 0.5 * self.U * self.n_target

        n_ibz = len(self.ibz_coords)
        Sig_R_ibz = np.zeros((n_ibz, len(omega)), dtype=np.complex128)
        for p, (ix, iy, iz) in enumerate(self.ibz_coords):
            sig_pos = self.last_Sig_lat[self.Nf + idx_nodes, ix, iy, iz] - Sigma_inf
            Sig_R_ibz[p] = Sigma_inf + pade_eval_mpmath(z_nodes, sig_pos, z_eval, dps=dps)

        G_R_ibz = 1.0 / (omega[None, :] + 1j * eta + self.mu - self.ibz_eps[:, None] - Sig_R_ibz)
        A_ibz = -np.imag(G_R_ibz) / np.pi
        A_loc = np.sum(self.ibz_weights[:, None] * A_ibz, axis=0)
        norm_ibz = _trapz(A_ibz, omega, axis=1)
        return {
            "omega": omega,
            "Sig_R_ibz": Sig_R_ibz,
            "A_ibz": A_ibz,
            "A_loc": A_loc,
            "norm_min": float(np.min(norm_ibz)),
            "norm_max": float(np.max(norm_ibz)),
        }

    def compute_optical_conductivity(self, ac_res, omega_opt_max=18.0, n_opt=181):
        """Compute dressed Kubo bubble optical conductivity Re sigma_xx(omega) and f-sum error Delta_f."""
        omega = ac_res["omega"]
        A_ibz = ac_res["A_ibz"]
        dw = omega[1] - omega[0]

        vx2_ibz = np.array([self.v2[c] / 3.0 for c in self.ibz_coords], dtype=float)
        w_ibz = self.ibz_weights * vx2_ibz

        f_nu = 1.0 / (1.0 + np.exp(np.clip(self.beta * omega, -500.0, 500.0)))
        omega_opt = np.linspace(0.0, omega_opt_max, int(n_opt))
        sigma_xx = np.zeros_like(omega_opt)

        for i_o, w_o in enumerate(omega_opt):
            if w_o < 1e-9:
                df = self.beta * f_nu * (1.0 - f_nu)
                integ = np.sum(w_ibz[:, None] * (A_ibz ** 2), axis=0) * df
                sigma_xx[i_o] = 2.0 * np.pi * float(_trapz(integ, omega))
            else:
                shift_f = w_o / dw
                k_sh = int(np.floor(shift_f))
                frac = shift_f - k_sh
                n_pts = len(omega) - k_sh - 1
                if n_pts <= 1:
                    continue
                nu_sl = omega[:n_pts]
                f1 = f_nu[:n_pts]
                f2 = 1.0 / (1.0 + np.exp(np.clip(self.beta * (nu_sl + w_o), -500.0, 500.0)))
                A1 = A_ibz[:, :n_pts]
                A2 = (1.0 - frac) * A_ibz[:, k_sh : k_sh + n_pts] + frac * A_ibz[:, k_sh + 1 : k_sh + 1 + n_pts]
                integ = np.sum(w_ibz[:, None] * A1 * A2, axis=0) * ((f1 - f2) / w_o)
                sigma_xx[i_o] = 2.0 * np.pi * float(_trapz(integ, nu_sl))

        Sig_hf = 0.5 * self.U * self.n_target
        xi_ref = self.eps_k + Sig_hf - self.mu
        f_ref = 1.0 / (1.0 + np.exp(np.clip(self.beta * xi_ref, -500.0, 500.0)))
        G_ref = 1.0 / (self.iwn[:, None, None, None] - xi_ref[None, :, :, :])
        n_k = f_ref + np.sum(np.real(self.last_G_lat - G_ref), axis=0) / self.beta
        Tx = float(2.0 * np.mean((-2.0 * self.t * np.cos(self.kx)) * n_k))
        I_fsum = -0.5 * np.pi * Tx
        I_opt = float(_trapz(sigma_xx, omega_opt))
        Delta_f = (I_opt - I_fsum) / max(abs(I_fsum), 1e-12)

        return {
            "omega_opt": omega_opt,
            "sigma_xx": sigma_xx,
            "sigma_dc": float(sigma_xx[0]),
            "I_opt": I_opt,
            "I_fsum": float(I_fsum),
            "Delta_f": float(Delta_f),
        }

    # ---------- Coupled Inner-Outer Level-III Self-Consistent Loop ----------

    def run(
        self,
        max_outer=60,
        max_inner=4,
        mix_outer=0.40,
        mix_inner=0.50,
        mix_mu=0.65,
        use_diis=True,
        tol=1e-5,
        verbose=True,
    ):
        diis = PulayDIIS(m_max=6) if use_diis else None
        n_del = len(self.Delta)
        sig_shape = self.Sigma_tilde.shape

        for it in range(max_outer):
            imp = self.solve_impurity(self.Delta, self.mu, self.n_target)
            vert = self.compute_local_vertex(imp)

            g = imp["g"][:, None, None, None]
            Del = self.Delta[:, None, None, None]
            eps = self.eps_k[None, :, :, :]

            G_dmft = 1.0 / (1.0 / g + Del - eps)
            G_tilde_0 = G_dmft - g
            G_tilde = 1.0 / (1.0 / G_tilde_0 - self.Sigma_tilde)

            Sig_t_target = self.Sigma_tilde.copy()
            for _ in range(max_inner):
                Sig_nl = self.Sigma_tilde / (1.0 + g * self.Sigma_tilde)
                Sig_lat = imp["Sig_imp"][:, None, None, None] + Sig_nl
                G_lat = 1.0 / (self.iwn[:, None, None, None] + self.mu - eps - Sig_lat)

                X_tilde, chi_t_L2, chi0_lat, b_lat, chiL_lat, chi_pp_tilde = self.compute_bubbles(
                    G_tilde, G_lat, imp
                )
                moriya = self.solve_moriya_and_charge(
                    vert, X_tilde, chi_t_L2, chi0_lat, b_lat, chiL_lat, self.n_target
                )
                Sig_t_new, Sig_ph, Sig_pp, V_ph = self.compute_dual_self_energy(
                    G_tilde, imp, vert, moriya, X_tilde, chi_t_L2, chi_pp_tilde
                )
                Sig_t_target = Sig_t_new
                diff_in = float(np.max(np.abs(Sig_t_new - self.Sigma_tilde)))
                if max_inner > 1:
                    self.Sigma_tilde = (1.0 - mix_inner) * self.Sigma_tilde + mix_inner * Sig_t_new
                    G_tilde = 1.0 / (1.0 / G_tilde_0 - self.Sigma_tilde)
                if diff_in < tol:
                    break

            Sig_nl = Sig_t_target / (1.0 + g * Sig_t_target)
            Sig_lat = imp["Sig_imp"][:, None, None, None] + Sig_nl
            mu_new = self.find_chemical_potential(Sig_lat)
            dmu = abs(mu_new - self.mu)
            self.mu = (1.0 - mix_mu) * self.mu + mix_mu * mu_new

            G_lat = 1.0 / (self.iwn[:, None, None, None] + self.mu - eps - Sig_lat)
            G_loc = np.mean(G_lat, axis=(1, 2, 3))
            Delta_new = self.Delta + (1.0 / imp["g"] - 1.0 / G_loc)

            diff_del = float(np.max(np.abs(Delta_new - self.Delta)))
            diff_sig = float(np.max(np.abs(Sig_t_target - self.Sigma_tilde)))
            err = max(diff_del, diff_sig, dmu)

            w0_idx = self.Nf
            C0_prop = float(
                np.max(np.abs(np.imag(Sig_t_target[w0_idx]))) * np.abs(np.imag(imp["g"][w0_idx]))
            )
            eff_mix = 0.2 * mix_outer if C0_prop > 0.85 else mix_outer

            if diis is not None and C0_prop <= 0.85:
                x_in = np.concatenate([self.Delta, self.Sigma_tilde.ravel()])
                x_out = np.concatenate([Delta_new, Sig_t_target.ravel()])
                x_next = diis.update(x_in, x_out, beta_mix=eff_mix)
                Del_cand = x_next[:n_del]
                Sig_cand = x_next[n_del:].reshape(sig_shape)
                causal_ok = (
                    np.all(np.isfinite(Del_cand))
                    and np.all(np.isfinite(Sig_cand))
                    and np.all(np.imag(Del_cand[self.Nf:]) < 0.0)
                    and np.all(np.imag(Del_cand[:self.Nf]) > 0.0)
                )
                if causal_ok:
                    self.Delta = Del_cand
                    self.Sigma_tilde = Sig_cand
                else:
                    diis.reset()
                    self.Delta = (1.0 - eff_mix) * self.Delta + eff_mix * Delta_new
                    self.Sigma_tilde = (1.0 - eff_mix) * self.Sigma_tilde + eff_mix * Sig_t_target
            else:
                if diis is not None and C0_prop > 0.85:
                    diis.reset()
                self.Delta = (1.0 - eff_mix) * self.Delta + eff_mix * Delta_new
                self.Sigma_tilde = (1.0 - eff_mix) * self.Sigma_tilde + eff_mix * Sig_t_target

            self.Sigma_tilde_ph = Sig_ph
            self.Sigma_tilde_pp = Sig_pp
            self.last_imp = imp
            self.last_vert = vert
            self.last_moriya = moriya
            self.last_V_ph = V_ph
            self.last_Sig_lat = Sig_lat
            self.last_G_tilde = G_tilde
            self.last_G_lat = G_lat

            if verbose and (it % 5 == 0 or err < tol):
                diag = self.matsubara_diagnostics()
                print(
                    f"it={it:2d} | err={err:.2e} | mu={self.mu:.4f} (n={diag['n_lat']:.4f}) "
                    f"| Z_imp={diag['Z_imp']:.3f} Z_hot={diag['Z_hot']:.3f} Z_cold={diag['Z_cold']:.3f} "
                    f"| chi_AFM={diag['chi_AFM']:.2f} | D_c={diag['D_c_irr']:+.3f} | C0={diag['C0']:.3f}"
                )
            if err < tol:
                break

        return self.matsubara_diagnostics()
