# Phase-Field-FLIP

A from-scratch reproduction of

> Bernhard Braun, Jan Bender, Nils Thuerey.
> **Adaptive Phase-Field-FLIP for Very Large Scale Two-Phase Fluid Simulation.**
> ACM Trans. Graph. 44(4), SIGGRAPH 2025. [doi:10.1145/3730854](https://doi.org/10.1145/3730854)

**[`docs/REPORT.md`](docs/REPORT.md) is the full reproduction report** —
what was reproduced, every validation result, the deviations, the bugs found,
and the negative results.

`docs/ALGORITHM.md` transcribes the paper's algorithms, equations and parameter
defaults. `docs/FINDINGS.md` holds the raw measurements.

## Status

| Stage | Paper sections | State |
|---|---|---|
| **1. PF-FLIP core** on a uniform MAC grid | 3.1–3.7 | **done**, runs end to end |
| 1b. Uniform-grid Galerkin aggregation MG-PCG | 6.1, 6.3 (uniform case) | **done** |
| **2a. MSBG data structure** | 4.1–4.3 | **done**, validated |
| **2b. Multiresolution Poisson operator** | 4.2, 6.1 | **done**, validated |
| **2c. Adaptive particles + dual adaptivity** | 5.1–5.4 | **done**, validated |
| **2d. Multiresolution P2G/G2P** | 5.4 step 4 | **done**, validated |
| **3. Multiresolution multigrid** | 6.1–6.3 | **done**, validated |
| **3b. Adaptive relaxation** | 6.4 | **done**, but measured a net loss on GPU — see `docs/FINDINGS.md` |
| **4. Escaped-particle droplets + spray** | 3.4, 7 | **done** (renderer is single-scatter, not the paper's MSBG tracer) |

The adaptive path runs end to end. The section 6 multigrid keeps the iteration
count flat across resolution (8 → 10 → 12 over a 4× range at 1000:1 contrast)
where Jacobi-CG grows as ≈`h⁻¹`, for up to a 26× wall-clock speed-up.

**Constraint on spray:** `delta_phi` and `ppc` are coupled through eq. (7) —
an isolated liquid particle splats its own mass into the field it is tested
against, flooring it at `1/sqrt(alpha_phi * rho~_0)`. At the paper's defaults
escaped-particle conversion needs `ppc > 17.4`; below that the spray channel
fails silently. See `docs/FINDINGS.md`.

**Constraint on adaptive runs:** probabilistic coarsening Poisson-thins the
particle cloud, so at 8 particles per cell a bulk liquid cell can fall under the
`φ = 0.5` tagging threshold and flip its density by 1000×. Use `ppc >= 27` at
the default `alpha_phi = 1`, or `alpha_phi <= 0.25` at low `ppc`.
`AdaptiveSimulator` warns outside the measured-safe region.

See `docs/FINDINGS.md` for measured accuracy of the 2:1 coarse-fine flux,
including the regime where it does **not** converge.

The paper's own implementation is CPU-based (TBB, AVX, no GPU). This port runs
the algorithm on the GPU via [NVIDIA Warp](https://github.com/NVIDIA/warp), so
it reproduces the *method*, not the paper's CPU scalability measurements.

## What stage 1 implements

One time step, following paper section 3:

1. **P2G** — splat mass (eq. 3) and momentum (eq. 4) to MAC faces with the
   higher-order polynomial kernel of eq. (6), `w = (max(1-(d/r)², 0))³`, which
   is cubic in the *squared* distance so no square root is needed.
2. **Phase field** — eq. (7) applied to the *reused* P2G weight accumulators.
   This is PF-FLIP's central trick: the interface comes out of the mass
   transfer for free, so there is no surface reconstruction, no level-set
   reinitialisation and no VOF geometry step. Face coefficients `β = 1/ρ`
   (eq. 9) follow directly.
3. **Forces**, saving `u~*` beforehand so gravity enters the FLIP update.
4. **Pressure solve** — variable-coefficient Poisson equation (8) at a 1000:1
   density contrast.
5. **Projection** `u = u* − Δt (1/ρ) ∇p`.
6. **G2P** — eq. (12) with a per-phase `α_FLIP`, which via eq. (13) is what
   sets the phase-dependent numerical viscosity.
7. **Extrapolation** of liquid velocity a few cells into the air (3.7).
8. **Advection** — third-order Runge–Kutta.

The pressure solver is the uniform-grid specialisation of section 6:
unsmoothed algebraic aggregation Galerkin multigrid (eq. 14 / Algorithm 1 with
all blocks at resolution level 0, where the transition factor `s` is 1),
red-black weighted Gauss-Seidel smoothing with `ω = 6/7` (6.3), used as the
preconditioner of a flexible CG with Polak–Ribière updates (6.4).

Because both phases fill a closed box there is no free surface, so the system
is pure-Neumann and singular; the constant null space is projected out of both
the right-hand side and the solution.

## Usage

```bash
pip install warp-lang numpy matplotlib scipy pytest

python scripts/run.py --scene dam_break --res 64 --frames 24 --png
python scripts/run.py --scene asteroid  --res 96 --frames 40 --png
python -m pytest tests/ -q
```

Scenes: `dam_break`, `still_pool`, `asteroid`, `rayleigh_taylor`.
Frames are written as `.npz` (phase field + strided particles); `--png` adds a
quick-look phase-field slice and particle scatter.

## Validation

`tests/` asserts the properties the paper claims, not just "it runs":

* **Operator correctness** — the matrix-free 7-point stencil matches an
  explicitly assembled sparse matrix, which is verified symmetric with zero row
  sums, at both 1:1 and 1000:1 coefficient contrast.
* **Solver correctness** — MG-PCG agrees with a direct sparse solve to the
  float32 floor, modulo the constant null space.
* **Phase separation** — `φ` is *exactly* 0 throughout the air. That is eq. (7)
  working as designed: `ρ~_min` sits a factor `η_φ = log(1000)` above the air's
  raw density, so all air-phase noise is clamped away. In the liquid `φ`
  genuinely fluctuates (≈0.8–1.0 at 8 particles per cell) — the "density
  fluctuations due to uneven particle distribution" of fig. 7 — but keeps a
  wide margin above the 0.5 tagging threshold.
* **Interface thickness** — the 0.05→0.95 transition spans ~1 cell
  (`ε_φ = r_0 = Δx`), sharper than the 3+ cells classical phase-field methods
  need.
* **Exact mass conservation** — `m_p` is fixed for a particle's lifetime
  (eq. 5), so no advection scheme can lose mass.
* **Hydrostatic stability** — a pool at rest stays at rest across the 1000:1
  density jump, with velocities *decaying*; this is the test that breaks naive
  variable-density projections.
* **`ρ~_0` calibration** — the numerically calibrated normalisation of eq. (7)
  matches the closed form `n_p · 64π r³/315` to within 5%.

Observed on a 64³ dam break with 2.1M particles (H100): 12–25 PCG iterations
per step at 1000:1 contrast, ~0.2 s per step.

## Layout

```
pfflip/config.py    parameters, defaults from the paper's appendix
pfflip/kernels.py   Warp kernels for P2G, phase field, projection, G2P, advection
pfflip/poisson.py   Galerkin aggregation MG-PCG
pfflip/sim.py       time-step orchestration
pfflip/scenes.py    scene setups
pfflip/io.py        frame output and quick-look rendering
docs/ALGORITHM.md   transcribed paper reference
```

## Known gaps

* No spatial adaptivity yet — the headline contribution (MSBG, adaptive
  particles, multiresolution multigrid) is stages 2–3. Stage 1 is a uniform
  grid, so it cannot reach the paper's resolutions.
* No surface tension. The paper also neglects it deliberately (3.4), as it only
  matters below the millimetre scale.
* Escaped particles are not yet converted to Lagrangian droplets (3.4, 7), so
  thin spray is carried by FLIP particles rather than ballistic droplets.
* `c_div` volume correction is implemented but disabled by default
  (`Config.k_div = 0`).
