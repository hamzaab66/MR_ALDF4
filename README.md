# MR-ALDF-IV: Moriya-Renormalized Analytical Ladder Dual-Fermion Suite

A closed, UV-finite, non-singular, and fully deterministic **QMC-free** diagrammatic extension of Dynamical Mean-Field Theory (DMFT) for the three-dimensional (3D) repulsive Hubbard and multi-orbital Kanamori models at finite temperature and hole doping.

Executing in $\mathcal{O}(N \log N)$ (uniform Matsubara) or $\mathcal{O}(r N \log N)$ (sparse DLR) time via 3D FFTs, the framework preserves exact analytical reduction to single-site DMFT ($d \to \infty$), the exact atomic limit and Heisenberg superexchange scale $T_N^{\mathrm{MF}} = zJ/4$ ($t \to 0$), and weak-coupling 3D Fermi-liquid behavior through $\mathcal{O}(U^2)$ plus exact $\mathcal{O}(U^3)$ non-local particle-hole and Kanamori particle-particle skeleton diagrams.

---

## Core Physics & Capabilities

### Core Level-III / Level-IV Continuum Engine (Track IV-A)
* **QMC-Free Impurity Closure:** Exact fourth-spectral-moment interpolating perturbation theory (mIPT-4) with correlated kinetic band shift $\Delta B_{\mathrm{kin}}$, tail-subtracted Galitskii-Migdal double occupancy, and $(s, c, pp)$ local-TPSC sum rules.
* **UV-Regularized Rank-2 Vertex Closure:** Eliminates the $\mathcal{O}(\Omega_m^{+2})$ UV divergence of naive local matching, restores bare-$U$ charge asymptotics, and analytically detects charge-irreducible vertex divergence lines ($\mathcal{D}_c^{\mathrm{irr}} = 0$, crossing at $\beta U = 8$ in the atomic limit).
* **3D $O(3)$ Bosonic SCSA & Non-Singular Moriya Regularization:** Escapes spherical Berlin-Kac criticality ($\nu=1, \gamma=2, \eta=0$) via Bray's parameter-free microscopic 3D $O(3)$ SCSA ($\eta_{\mathrm{SCSA}} = 0.0790$, $\nu \approx 0.744$, $\gamma \approx 1.431$) enforced by the UV-tail-free excess Pauli sum rule.
* **Self-Consistent TU-SBE $E_g$ ($d$-Wave) Feedback:** Double-counting-free cross-channel pairing feedback on the cubic $O_h$ $E_g$ basis $\{\psi_{d_1}, \psi_{d_2}\}$ starting at $\mathcal{O}(U^4)$.
* **Smooth $C^\infty$ Dual-Inverse Herglotz Causality Control:** Eliminates Mott-insulating branch overflow ($\mathcal{C}_0 \le 0.762 < 1$) while preserving exact particle-hole symmetry and half-filling density ($n_{\mathrm{lat}} = 1.000000$).
* **Maki-Thompson + Aslamazov-Larkin Optics & 50-Digit Nevanlinna Continuation:** Captures AFM Umklapp backscattering ($\tau_{\mathrm{tr}}^{-1}/\tau_{\mathrm{sp}}^{-1} \to 2$), exact C-parity-obeying $2\times 2$ Aslamazov-Larkin optical drag ($\Pi_{xx}^{\mathrm{AL}} \sim 10^{-35}$ at $\delta=0$), and strictly causal ($\mathrm{Im}\,\Sigma^R < 0$) real-axis spectra via 50-digit Schur/Nevanlinna continuation.

### Five Production Pillars (Sparse DLR Suite)
1. **Pillar 1 — Sparse Odd-Symmetric DLR Engine:** Compresses Matsubara grids to $r_F = 22\text{--}42$ nodes, reaching $T = 0.02t$ and $L = 12$ ($N = 1728$ lattice sites) with zero tail truncation.
2. **Pillar 2 — Explicit 3D Ordered-Phase Solvers:** $2\times 2$ bipartite sublattice Néel AFM solver (capturing up to $28.8\%$ transverse-magnon suppression of $T_N^{3D}$ vs. AFM-DMFT) and $2\times 2$ Nambu-Gor'kov $O_h$ $E_g$ ($d$-wave) superconductor solver.
3. **Pillar 3 — Track IV-B Lehmann Exact Diagonalization:** Deterministic $(N_\uparrow, N_\downarrow)$ symmetry-sector ED impurity solver ($N_b = 3\text{--}5$ bath orbitals, max sector dim $\le 400$, zero sign problem) with Rank-6 SVD local vertices and multi-line $\Gamma_c^{\mathrm{irr}}$ divergence counting ($N_{\mathrm{div}} = 0 \to 10$).
4. **Pillar 4 — Deep-Mott Three-Particle $\gamma^{(6)}$ Contractions:** Analytical atomic-asymptote (Track IV-A) and source-field ED (Track IV-B) six-point cumulants capturing spin-polaron ($\mathcal{O}(\delta)$) and two-magnon feedback at $U \gg W$.
5. **Pillar 5 — Multi-Orbital Kanamori Oxide Solver:** $M = 1, 2, 3$ rotationally invariant Kanamori $(U, U', J_H, \Delta_{\mathrm{cf}})$ solver with exact $4^M$ Fock-space multiplets and $M\times M$ orbital-matrix dual ladders, benchmarked on $\mathrm{V_2O_3}$ Hund's-metal, orbital-selective Mott (OSMT), and $R\mathrm{NiO_3}$ crystal-field proxies.

---

## Quick Start

Install the required Python dependencies (Python $\ge$ 3.9):

```bash
pip install -r requirements.txt
```

Run the live five-pillar verification suite:

```bash
python run_sweeps.py
```

Or run the uniform-Matsubara Level-IV engine with Maki-Thompson + Aslamazov-Larkin optics and 50-digit Nevanlinna continuation:

```python
from mr_aldf_level4 import MRALDFLevel4Solver

solver = MRALDFLevel4Solver(U=8.0, T=0.35, delta=0.0, L=8, Nf=36, Nb=14)
diag = solver.run(max_outer=40)
mt_al = solver.solve_mt_al_optics(m_opt_list=(0, 1))
nev = solver.analytic_continuation_nevanlinna(n_nodes=10, dps=50)
```

---

## Required Modules & Dependencies

### External Python Dependencies
* **`numpy`** & **`scipy`** — 3D FFT convolutions, linear algebra, Rank-$R$ SVD, and Brent root-finding.
* **`mpmath`** — Required for 50-digit arbitrary-precision Schur/Nevanlinna and Vidberg-Serene Padé analytic continuation.

### Internal Module Map

| Workflow / Pillar | Required Suite Modules | Purpose |
| :--- | :--- | :--- |
| **Core Level-III / IV (Uniform Matsubara)** | `mr_aldf_level3.py`, `mr_aldf_level4.py` | Uniform-Matsubara Track IV-A solver: mIPT-4, Rank-2 vertices, Herglotz control, 3D $O(3)$ SCSA, TU-SBE $E_g$, MT+AL optics, and 50-digit Nevanlinna/Padé continuation. |
| **Pillar 1: Ultra-Low-$T$ Sparse DLR** | `mr_aldf_dlr.py`, `mr_aldf_level4_dlr.py`, `mr_aldf_gamma6.py` | Odd-symmetric DLR basis/interpolation engine + standalone Level-IV DLR solver ($T \ge 0.02t$, $L=12$). |
| **Pillar 2: 3D Ordered Phases** | `mr_aldf_dlr.py`, `mr_aldf_neel.py`, `mr_aldf_nambu_sc.py` | Explicit $2\times 2$ sublattice Néel AFM and $2\times 2$ Nambu $E_g$ ($d$-wave) superconducting solvers on sparse DLR grids. |
| **Pillar 3: Track IV-B Lehmann ED** | `mr_aldf_dlr.py`, `mr_aldf_ed.py`, `mr_aldf_ed_vertex.py`, `mr_aldf_track4b.py`, `mr_aldf_level4_dlr.py`, `mr_aldf_gamma6.py` | Symmetry-sector ED impurity solver, Rank-$R$ SVD local 4-point vertices, multi-line $N_{\mathrm{div}}$ counter, and Track IV-B dual-fermion driver. |
| **Pillar 4: Deep-Mott $\gamma^{(6)}$ Polarons** | `mr_aldf_dlr.py`, `mr_aldf_gamma6.py`, `mr_aldf_level4_dlr.py` *(+ `mr_aldf_ed.py`, `mr_aldf_ed_vertex.py` for Track IV-B)* | Self-consistent spin-polaron and two-magnon six-point vertex contractions coupled into the DLR loop. |
| **Pillar 5: Multi-Orbital Kanamori** | `mr_aldf_dlr.py`, `mr_aldf_kanamori.py` | $M=1,2,3$ orbital-matrix dual-fermion solver for $\mathrm{V_2O_3}$, OSMT, and $R\mathrm{NiO_3}$ proxies. |
| **Batch Verification** | `run_sweeps.py` | Executes the live five-pillar quick verification suite. |

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── run_sweeps.py
├── mr_aldf_level3.py
├── mr_aldf_level4.py
├── mr_aldf_dlr.py
├── mr_aldf_level4_dlr.py
├── mr_aldf_neel.py
├── mr_aldf_nambu_sc.py
├── mr_aldf_ed.py
├── mr_aldf_ed_vertex.py
├── mr_aldf_track4b.py
├── mr_aldf_gamma6.py
├── mr_aldf_kanamori.py
├── benchmarks/
│   ├── bench_*.json
│   └── roadmap.json
├── figures/
│   ├── fig1_mott_phase_diagram_600dpi.png
│   ├── fig2_optics_pairing_causality_600dpi.png
│   ├── fig3_level4_extension_600dpi.png
│   └── fig_level4_five_pillars.{png,pdf}
└── paper/
    ├── MR_ALDF_4.pdf
```
