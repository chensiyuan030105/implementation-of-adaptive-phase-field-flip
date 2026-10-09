# Reproduction report — Adaptive Phase-Field-FLIP

A from-scratch GPU reproduction of

> Bernhard Braun, Jan Bender, Nils Thuerey.
> **Adaptive Phase-Field-FLIP for Very Large Scale Two-Phase Fluid Simulation.**
> ACM Trans. Graph. 44(4), SIGGRAPH 2025. [doi:10.1145/3730854](https://doi.org/10.1145/3730854)

6,007 lines of Python/Warp across 9 commits, 39 tests. Hardware: NVIDIA H100,
[NVIDIA Warp](https://github.com/NVIDIA/warp) 1.17.

The paper's own implementation is **CPU-based** (TBB, AVX, explicitly no GPU).
This port therefore reproduces the *method*, not the paper's CPU scalability
measurements — and as section 7 of this report shows, that distinction turns
out to matter for one of the paper's contributions.

---

## 1. What was reproduced

| Paper section | Component | State |
|---|---|---|
| 3.1–3.7 | PF-FLIP core on a uniform MAC grid | done, validated |
| 4.1–4.3 | MSBG multiresolution sparse block grid | done, validated |
| 4.2, 6.1 | Multiresolution Poisson operator | done, validated |
| 5.1–5.4 | Adaptive particles, dual adaptivity | done, validated |
| 5.4 step 4 | Multiresolution P2G / G2P | done, validated |
| 6.1–6.3 | Multiresolution multigrid preconditioner | done, validated |
| 6.4 | Adaptive relaxation (Algorithm 2) | done — **measured a net loss on GPU** |
| 3.4, 7 | Escaped particles, droplets, spray | done |
| 7 | Volumetric rendering | simplified (single-scatter, not MSBG multiple-scatter) |

Not attempted: particle-skinning surface reconstruction; the paper's
billion-particle production scenes.

---

## 2. Core PF-FLIP (section 3)

PF-FLIP's central trick is that the phase field is a **by-product of the P2G
transfer**: the splatting weight accumulators needed for the variable-coefficient
Poisson solve are reused directly by eq. (7). There is no surface
reconstruction, no level-set reinitialisation, no VOF geometry step.

Validated by asserting the paper's claims, not merely that it runs:

| Property | Result |
|---|---|
| Matrix-free operator vs assembled sparse matrix | identical, at 1:1 and 1000:1 contrast |
| Operator symmetry, zero row sums | exact |
| MG-PCG vs direct sparse solve | agrees to the float32 floor |
| `φ` in the air phase | **exactly 0** |
| Interface thickness (0.05 → 0.95) | ~1 cell (`ε_φ = r_0 = Δx`) |
| Mass conservation | exact (`m_p` fixed, eq. 5) |
| Hydrostatic equilibrium at 1000:1 | stable, velocities **decay** |
| `ρ̃_0` calibration vs `n_p·64πr³/315` | within 5% |

64³ dam break, 2.1M particles: ~0.2 s/step, 12–25 PCG iterations.

**On `φ` in the liquid.** An early test asserted `min(φ_liquid) > 0.95` and
failed at 0.785. Diagnosis showed wall-adjacent cells (0.785) and interior
cells (0.802) behave identically — so this is not a boundary artefact but the
inherent particle noise the paper draws in fig. 7. Eq. (7)'s asymmetry is
deliberate: `ρ̃_min = η_φ ρ_g ρ̃_0` with `η_φ = log(1000)` sits far above the
air's raw density, so air noise is clamped to a clean zero while liquid noise
survives. The assertion was wrong, not the code.

---

## 3. MSBG (section 4)

MSBG inverts the tree convention: **block geometric size is constant and only
resolution varies**, so every block has exactly 6 face neighbours regardless of
level. Dense block raster for O(1) access, Morton-ordered active slots,
prefix-sum cell allocation so a level-`l` block really costs `8^-l`. Addressing
is pure shift-and-mask on finest-level coordinates.

Validated: 2:1 grading holds; finest coordinates tile the cells exactly (a
level-`l` cell is hit by exactly `8^l` of them); cell counts match the level
histogram.

### The transition constants are not tuned

Writing the coupling as a plain finite volume, `T = β · Area / dist` with
`Area` the finer side's face area and `dist` the centre-to-centre distance,
reproduces all three of the paper's constants exactly:

| transition | finite volume | paper (Algorithm 1) |
|---|---|---|
| fine–fine at level `l` | `β h_l` | `s = 2^l` |
| coarse `l` ↔ fine | `β (H/2)²/(¾H) = β H/3` | `2^l σ_CF = 2^l/3` |
| fine `l` ↔ coarse | `β h²/(3h/2) = ⅔ β h` | `2^l σ_FC = 2^l ⅔` |

So `Δ_g = 3/4` is simply the centre-to-centre distance across a 2:1 jump
measured in coarse spacings (`H/2 + H/4`). Having this fall out rather than
being tabulated is what let the multigrid coarsen only `β` and get `s` from
geometry for free (section 5 below).

### Measured accuracy of the 2:1 flux

The scheme is **consistent** — a linear field is reproduced to roundoff — but
the per-volume truncation error at a transition is **O(1) and independent of
`h`**:

| field | h = 1/16 | h = 1/32 | h = 1/64 |
|---|---|---|---|
| `x` | 1.2e-10 | 2.9e-11 | 7.3e-12 |
| `x²` | 0.500 | 0.500 | 0.500 |
| `x³` | 0.781 | 0.766 | 0.758 |

For `p = x²` the error is exactly `0.25 p'' = 0.5`, matching the closed form:
the one-sided difference `(p_F − p_C)/(1.5h)` approximates the gradient at the
*cell-pair midpoint*, not at the face — an offset of `h/4`. This is the
documented behaviour of the Losasso et al. 2004 flux the paper cites, not a
defect.

The consequence is visible in the solution error:

| refinement map | 16³ | 32³ | 64³ | rate |
|---|---|---|---|---|
| uniform | 1.29e-2 | 3.22e-3 | 8.04e-4 | **2.00** |
| slab (one transition plane) | 3.63e-2 | 1.03e-2 | 4.33e-3 | 1.8 → 1.2 |
| checkerboard (~50% of cells at a transition) | 3.97e-2 | 2.46e-2 | 2.47e-2 | **0.0 — stalls** |

O(1) local error converges globally only when the transition set is a vanishing
*fraction* of the domain. That holds for the interface-following refinement
bands the method produces (paper fig. 9a) and fails for pathological maps.
**Practical rule: a checkerboard-like refinement map is not merely inefficient,
it does not converge.**

---

## 4. Adaptive particles (section 5)

Probabilistic coarsening is the elegant part: deleting each particle
independently with probability `1 − 8^-Δl` thins a uniform cloud by exactly the
volume ratio **without any neighbourhood search**, which is what keeps FLIP
neighbour-free.

Validated: coarsening keeps 1/8 of a uniform cloud and conserves mass to within
sampling noise; the `D_p ≥ 3` hysteresis delays and resets correctly; splitting
fills all 8 octants; the atomic-min refinement map and 2:1 grading produce
interface-following bands.

Per-particle `α_FLIP` (5.3) uses the **local grid spacing**, not the particle
radius — decoupling the two is what permits particles to cross resolution
boundaries within a step at the CFL of 3–4 the paper runs at.

---

## 5. Multiresolution multigrid (sections 6.1–6.3)

Multigrid levels are kept independent of MSBG resolution levels: at multigrid
level `L` a block sits at `min(l + L, log₂n₀)`. A block that has bottomed out
at one cell carries its coefficients over — Algorithm 1's copy-by-reference
case. Since `min(l + L, cap)` preserves `|l_a − l_b| ≤ 1`, every level inherits
2:1 grading for free.

Because the operator derives `T = β·Area/dist` from geometry, only `β` needs
transferring: a coarse face coefficient is the mean of the four fine faces,
i.e. Algorithm 1's `s·sum/4` with `s` supplied by geometry. Probing the 2×2
patch through `locate` makes the copy-by-reference case fall out automatically
— when the fine block has the same resolution all four probes land on the same
cell.

Measured at 1000:1 contrast, tol 1e-4:

| grid | cells | Jacobi-CG | MG-PCG | speed-up |
|---|---|---|---|---|
| 64³ uniform | 262,144 | 154 it / 1.29 s | **10 it / 0.16 s** | 8.0× |
| 128³ uniform | 2,097,152 | 379 it / 9.73 s | **12 it / 0.45 s** | 21.7× |
| 64³ adaptive | 147,456 | 196 it / 1.18 s | **9 it / 0.13 s** | 9.2× |
| 128³ adaptive | 1,179,648 | 417 it / 6.47 s | **11 it / 0.25 s** | 26.1× |

Jacobi grows as roughly `h⁻¹`; the multigrid count is flat (8 → 10 → 12 over a
4× resolution range), which is the defining property of a working multigrid.

Re-discretising rather than assembling variationally leaves the V-cycle only
approximately symmetric — which is exactly why the outer Krylov method must be
**flexible** CG with Polak–Ribière (paper 6.4) rather than standard PCG.

---

## 6. Escaped particles and spray (sections 3.4, 7)

All three transitions of 3.4 implemented: liquid → droplet, air inside liquid →
bubble → deleted (as the paper does), droplet re-entering liquid → back to a
FLIP particle. Droplets are Lagrangian point masses under gravity and sphere
drag with per-droplet adaptive sub-stepping, log-normal radii on emission, and
the `dr/dt ~ −1/r` evaporation ODE via RK2. Extra spray in steep-velocity-
gradient regions dominates escaped-particle emission by ~100:1. A 16-bit spray
density field at 2–4× the pressure resolution is rasterised only at render time.

Droplets share the FLIP particle arrays and are distinguished by a `state`
channel, so conversion either way is a flag flip and no stream compaction is
ever needed.

---

## 7. Negative result: adaptive relaxation (6.4) does not pay off on GPU

Implemented faithfully, measured, and found to be a net loss. Kept available as
`preconditioner="mg-adaptive"` but not the default.

The adaptive smoother does three full sweeps in practice against the plain
smoother's two, so the honest comparison is against plain multigrid with three:

| grid | mg pre=2 | mg pre=3 | mg-adaptive |
|---|---|---|---|
| 64³ | 9 it / 0.31 s | **6 it / 0.22 s** | 7 it / 0.40 s |
| 128³ | 11 it / 0.39 s | **8 it / 0.29 s** | 8 it / 0.73 s |
| 256³ | 15 it / 3.13 s | **9 it / 2.06 s** | 9 it / 2.50 s |

Plain multigrid with one extra sweep matches or beats the adaptive version on
both metrics. **The entire apparent gain was the third sweep.** Default
pre/post sweeps were raised from 2 to 3 as a result.

Why it does not transfer:

1. **The paper's win is explicitly CPU-cache-bound.** Section 6.4: "the
   remaining active block set often fits entirely into the CPU's last-level
   cache, eliminating the RAM bandwidth bottleneck". On an H100 a full
   red-black sweep over 9.4M cells is already bandwidth-saturated and cheap.
2. **The machinery costs more than it saves on GPU** — a per-cell `|delta|`
   pass, a per-block atomic reduction, a neighbour-activation kernel, and a
   device readback for `θ` that is a full pipeline sync.
3. **Too few blocks for the statistical threshold.** At `n₀ = 16` the counts
   here are 64/512/4096; `θ = mean + 2σ` of the log-histogram over that few
   samples often lands above the sample maximum, so the active set collapses to
   0–3% instead of the paper's ~10%.

---

## 8. Bugs found, and what caught them

| Bug | Symptom | Why the test missed it / what caught it |
|---|---|---|
| `r0 = 1.0` hardcoded in split jitter | liquid centroid jumped 0.250 → 0.377 in one step — impossible at the measured \|v\| = 0.44 | existing test allowed deviation of 4.0 in a domain of size 1; bound now expressed in units of `r0` |
| Polak–Ribière silently inert | `r_prev` snapshotted *after* the axpy updating `r`, so β ≡ 0 and flexible CG degenerated to steepest descent | found while wiring in the multigrid |
| Stale multigrid hierarchy | solver rebuilt only on cell-*count* change, but the hierarchy is built from the refinement *map* | reasoned about during integration; now keyed on the map |
| `l(i,L) ≤ L` active-set gating | 9 → 37, 11 → 81, 13 → 129 iterations | relies on the paper's shared inter-level storage; each level owns its arrays here |
| Coarsest level adaptively starved | 13 → 23 iterations at 256³ | it is a near-solve (80 sweeps), not a smoothing pass |
| Aliased CG/V-cycle buffers | — | caught before running: `self.rhs` aliased `L0.b`, which `precondition` overwrites |

Several of my own *tests* were also wrong and had to be corrected — the
`φ > 0.95` assertion, a strided particle subsample that collapsed onto one
plane, and a mass-fraction expectation that ignored the 1000:1 density ratio.

---

## 9. Two independent constraints converge on `ppc ≥ 27`

The most useful finding of the reproduction. Eight particles per cell — fine
for the uniform solver — breaks the adaptive path in two unrelated ways.

### 9.1 Phase mis-tagging under probabilistic coarsening

Coarsening replaces a lattice with a **Poisson** sample, ~35% RMS density
fluctuation at 8 ppc. A bulk liquid cell that comes up sparse falls under the
`φ = 0.5` tagging threshold, so its `β = 1/ρ` jumps by **1000×** — a genuine
density hole inside the liquid. The pressure solve responds correctly and
accelerates particles away from it, and at `α_FLIP = 0.985` nothing dissipates
the result.

Isolated by freezing parts of the loop: a *static* adaptive grid settles exactly
like a uniform one (|v| 0.048 → 0.092), so the discretisation is not implicated;
only the transition breaks (|v| → 3.08).

| `α_φ` | mis-tagged bulk cells per step | \|v\| at step 12 |
|---|---|---|
| 1.00 | 0, 0, 0, **3, 107, 88, 84** | 2.48 |
| 0.50 | 0, 0, 0, 0, 0, 6, 6, 5 | 0.81 |
| 0.25 | 0 throughout | 0.0006 |

The counts match a Poisson estimate: mis-tagging needs a 75% density deficit at
`α_φ = 1`, and `P(k ≤ 2 | λ = 8) ≈ 1.4%` over 32,768 cells predicts a few
hundred. At ppc = 27 with the paper's default `α_φ = 1`: zero mis-tagging.

### 9.2 The spray channel fails silently

`δ_φ` and `ppc` are coupled through eq. (7), because **an isolated liquid
particle splats its own mass into the field it is then tested against**:

> `φ_single = 1 / sqrt(α_φ · ρ̃_0)`,  `ρ̃_0 ≈ 0.638 · ppc`

Conversion needs `φ_single < ½ − δ_φ`, i.e.

> `ppc > 1 / (α_φ · 0.638 · (½ − δ_φ)²)`

At the paper's defaults that is **ppc > 17.4**. Measured on a 64³ asteroid
impact at ppc = 8 over 80 steps: `φ` at liquid particles bottomed out at 0.294,
exactly **one** particle out of 2.1 million crossed the threshold, and one
droplet was emitted. At ppc = 27 the same scene emitted 2,018 by step 91.
Nothing errors — there is simply no spray.

Two requirements derived from completely different directions land on the same
number. `AdaptiveSimulator` now warns outside the measured-safe region.

---

## 10. Known gaps and honest limits

* **Adaptive path velocity settling.** Fixed by `ppc ≥ 27` or `α_φ ≤ 0.25`, but
  the refinement map still oscillates between levels rather than settling into
  stable bands, because the block-resolution `d̃` shifts when the resolution
  changes. Measured only at 8 blocks/axis (a 64³ finest grid) — far below the
  regime where bands have room to be many blocks wide. Untested at scale.
* **Rendering.** A single-scattering ray marcher, not the paper's MSBG
  volumetric tracer with approximate multiple scattering, and no particle
  skinning. At 64³/7M particles the spray reads as discrete beads rather than
  mist: ~10⁴ droplets against the paper's billions of particles. **That gap is
  the paper's motivation for adaptivity and scale, not something a renderer can
  close.**
* **Scale.** The largest grid exercised here is 256³ in the solver benchmark and
  96³ in full simulation. The paper reports 6144 × 2048 × 1600 pressure grids
  with 3 billion particles.
* **No surface tension** — the paper also neglects it deliberately (3.4).
* **`c_div` volume correction** implemented but disabled by default.
* **CPU scalability claims untested.** The paper's MSBG-vs-VDB comparison and
  its single-workstation timings are properties of a CPU implementation and
  were not in scope.

---

## 11. Reproducibility

```bash
pip install warp-lang numpy matplotlib scipy pytest
python -m pytest tests/ -q                                  # 39 tests
python scripts/mms_study.py                                 # §3 convergence tables
python scripts/run.py --scene asteroid --res 96 --frames 40 --png
```

`docs/ALGORITHM.md` transcribes the paper's equations, algorithm boxes and
parameter defaults. `docs/FINDINGS.md` holds the raw measurements behind every
table above.
