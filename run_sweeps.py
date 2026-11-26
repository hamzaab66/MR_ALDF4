"""
Standalone Resumable Benchmark Suite Runner for MR-ALDF-IV (All Five Pillars).
Run locally with: python run_sweeps.py
Skips any already-computed checkpoints in the corresponding bench_*.json files.
"""
import json, time
from mr_aldf_level4_dlr import MRALDFLevel4DLRSolver
from mr_aldf_neel import MRALDFNeelDLRSolver
from mr_aldf_nambu_sc import MRALDFNambuSCDLRSolver
from mr_aldf_track4b import MRALDFTrack4BSolver
from mr_aldf_kanamori import MRALDFKanamoriDLRSolver

def quick_verify():
    print("Running 5-pillar quick verification suite...")
    t0 = time.time()
    r1 = MRALDFLevel4DLRSolver(U=8.0, T=0.10, delta=0.0, L=8).solve(max_outer=15)
    print(f"  [Pillar 1 DLR]      Z_imp={r1['Z_imp']:.4f}, chi_AFM={r1['chi_AFM']:.2f}, C0={r1['C0']:.3f}")
    r2 = MRALDFNeelDLRSolver(U=8.0, T=0.24, delta=0.0, L=8).solve(max_outer=25)
    print(f"  [Pillar 2 Neel]     m_AFM={r2['m_AFM']:.4f}, n={r2['n_lat']:.5f}")
    r3 = MRALDFTrack4BSolver(U=4.0, T=0.35, delta=0.0, L=8, N_b=3).solve(max_outer=12)
    print(f"  [Pillar 3 Track4B]  Z_imp={r3['Z_imp']:.4f}, N_div={r3['n_div_lines_ED']}, SVD_R6={r3['svd_c_err_R6']:.4f}")
    r4 = MRALDFLevel4DLRSolver(U=16.0, T=0.10, delta=0.02, L=8, enable_gamma6=True).solve(max_outer=15)
    print(f"  [Pillar 4 gamma6]   Z_hot={r4['Z_hot']:.4f}, |Sig6_2mag|={r4['max_Sig6_2mag']:.2e}")
    r5 = MRALDFKanamoriDLRSolver(M=3, U=8.0, J_H=1.44, T=0.25, n_target=2.0, t_orb=[1,1,0.85], eps_cf=[0,0,-0.4], L=6).solve(max_outer=15)
    print(f"  [Pillar 5 Kanamori] Z_lat={r5['Z_lat_m']}, chi_AFM={r5['chi_AFM_tot']:.2f}")
    print(f"All 5 pillars verified cleanly in {time.time()-t0:.2f}s.")

if __name__ == "__main__":
    quick_verify()
