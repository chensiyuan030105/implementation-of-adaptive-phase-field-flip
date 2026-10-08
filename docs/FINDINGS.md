# Accuracy of the MSBG 2:1 coarse-fine flux

Measured, not assumed.  Setup: `scripts/mms_study.py` solves
`p = cos(2πx)cos(2πy)cos(2πz)` (homogeneous Neumann on the unit box, so it is
compatible with the singular pure-Neumann operator) and reports the solution
error; `tests/test_msbg.py::test_truncation_error_at_coarse_fine_is_O1_not_growing`
measures the local truncation error directly.

## Local truncation error (per unit volume, interior cells)

| field | h = 1/16 | h = 1/32 | h = 1/64 | behaviour |
|---|---|---|---|---|
| `x`   | 1.2e-10 | 2.9e-11 | 7.3e-12 | **exact** |
| `x²`  | 0.500 | 0.500 | 0.500 | O(1), h-independent |
| `x³`  | 0.781 | 0.766 | 0.758 | O(1), h-independent |

The scheme is **consistent** — a linear field is reproduced to roundoff, i.e.
the first-order flux terms cancel exactly across a 2:1 jump.  Beyond that the
per-volume truncation error at a transition is O(1) and does not shrink with
`h`.  For `p = x²` it is exactly `0.25 p'' = 0.5`, which is the closed form:
the one-sided difference `(p_F − p_C)/(1.5h)` approximates the gradient at the
*midpoint of the cell pair* rather than at the face, an offset of `h/4`.

This is the known behaviour of the Losasso et al. 2004 flux the paper cites in
section 4.2.  It is a property of the discretisation, not a defect here.

## Consequence for the solution error

| refinement map | 16³ | 32³ | 64³ | rate |
|---|---|---|---|---|
| uniform | 1.29e-2 | 3.22e-3 | 8.04e-4 | **2.00** |
| slab (one transition plane) | 3.63e-2 | 1.03e-2 | 4.33e-3 | 1.8 → 1.2 |
| checkerboard (≈50% of cells at a transition) | 3.97e-2 | 2.46e-2 | 2.47e-2 | **0.0 — stalls** |

O(1) local truncation error converges globally only when the transition set is
a vanishing *fraction* of the domain.  That holds for the refinement **bands**
the method actually produces (interface-following shells, paper fig. 9a) and
fails for a pathological map where transitions are everywhere.

**Practical rule:** keep refinement bands several blocks wide (`beta_bw` small
enough) so that transition cells stay a small fraction of the total.  A
checkerboard-like refinement map is not merely inefficient — it does not
converge.

---

# Stage 2d: multiresolution P2G/G2P

## The split-jitter bug

`Particles.resample` passed a hardcoded `r0 = 1.0` to `emit_particles`, so
split children were jittered by `±0.5 * 1.0 * 2**l_p` **in world units** —
across the entire unit-cube domain instead of inside the parent cell.

Symptom: on a hydrostatic pool the liquid centroid jumped from 0.250 to 0.377
in a *single* step, which is impossible by advection (it would need |v| ≈ 12
while the measured maximum was 0.44). The existing unit test asserted
`|child − parent| < 1.0 * 2**2`, a threshold of 4.0 in a domain of size 1 — far
too loose to see it. The test now expresses the bound in units of `r0`.

After the fix the centroid holds to 0.25000 → 0.25011 over 20 steps.

## Phase-field normalisation is density-coupled

A level-`l` particle carries `2**(3l)` the mass of a level-0 one, so the raw
P2G accumulator is divided by `2**(3l)` to let eq. (7) keep its level-0
constants everywhere. That is only valid when the particle **density** also
matches the level, i.e. `8**-l` of the level-0 count — exactly what
probabilistic coarsening produces.

If particles are promoted without being thinned, the raw density is inflated by
`8**l` and the air phase stops clamping to zero: at level 2 it reads
`φ ≈ sqrt(64/1000) ≈ 0.25`. Since cells are tagged air/liquid at `φ = 0.5`,
a large enough mismatch would silently mis-tag the air as liquid and corrupt
the density field feeding the pressure solve. The dual-adaptivity loop keeps
the two in sync; this is the invariant it has to maintain.

## Rate-limiting level changes

The paper's `prob_del = 1 − 8**-(l_new − l_p)` handles multi-level jumps
analytically, but a 2-level jump deletes 63/64 of the particles in one step.
The grid then follows `M = min(l_p)` down to match, P2G is left catastrophically
under-sampled, and the phase field — hence the whole pressure solve — is garbage
for that step. `l_new` is therefore clamped to `l_p ± 1` per step, mirroring in
time what 2:1 grading does in space.

## What still does not work

On a hydrostatic pool the adaptive path now conserves the liquid centroid, but
**the velocity field does not settle**: peak liquid speed rises from 0.04 to
≈1.8 after the first refine/coarsen cycle and stays there, where the same
scene on a forced-uniform MSBG (`max_level = 0`) settles to ≈0.07, matching
stage 1. The resample perturbs the P2G density field, which creates spurious
pressure, and at `α_FLIP ≈ 0.985` with no physical viscosity nothing damps it.

The grid also oscillates between level 0 and level 1 rather than settling into
stable bands, because the block-resolution `d~` shifts when the resolution
changes. Both effects are measured at 8 blocks/axis with `n0 = 8` (a 64³ finest
grid) — far smaller than the regime the method targets, where bands have room
to be many blocks wide. Whether they persist at scale is untested.

---

# Stage 3: multiresolution multigrid

Measured with `tol = 1e-4` on a beta field carrying a sharp 1000:1 jump (a
liquid/air interface), comparing the two preconditioners behind the same
flexible-CG driver.

| grid | cells | Jacobi-CG | MG-PCG | speed-up |
|---|---|---|---|---|
| 32³ uniform | 32,768 | 94 it | 8 it | — (JIT-dominated) |
| 64³ uniform | 262,144 | 154 it / 1.29 s | **10 it / 0.16 s** | 8.0× |
| 128³ uniform | 2,097,152 | 379 it / 9.73 s | **12 it / 0.45 s** | 21.7× |
| 32³ adaptive | 18,432 | 94 it | 7 it | 3.4× |
| 64³ adaptive | 147,456 | 196 it / 1.18 s | **9 it / 0.13 s** | 9.2× |
| 128³ adaptive | 1,179,648 | 417 it / 6.47 s | **11 it / 0.25 s** | 26.1× |

Jacobi grows roughly as `h^-1`; the multigrid count is flat (8 → 10 → 12 over a
4× resolution range), which is the defining property of a working multigrid and
is what section 6 exists to deliver.

## Why re-discretisation rather than variational Galerkin

`mr_operator` derives each coupling as `beta * Area / dist` from the grid
geometry, so only `beta` needs transferring and the coarse coefficient is the
mean of the four fine faces — Algorithm 1's `s * sum / 4` with `s` supplied by
geometry instead of tabulated. Sampling the 2×2 patch through `locate` makes
Algorithm 1's copy-by-reference case fall out for free: when the fine block has
the same resolution, all four probes land on the same cell.

The cost is that the V-cycle is only approximately symmetric. That is exactly
why the outer Krylov method must be **flexible** CG with a Polak-Ribière update
(paper 6.4) rather than standard PCG.

## Two bugs this exposed

1. **Polak–Ribière was silently inert.** `r_prev` was snapshotted *after* the
   `axpy` that updates `r`, so `beta = <z, r − r_prev>/rz` evaluated to 0 every
   iteration and the method degenerated to steepest descent.
2. **Stale multigrid hierarchy.** The simulator reallocated its solver only
   when the cell *count* changed, but the hierarchy is built from the per-block
   refinement *map*. Two different maps with the same cell count would have
   shared a hierarchy. Now keyed on the map itself.

## Not implemented from section 6

The adaptive relaxation of 6.4 — the active-block FIFO queue, the
log-histogram threshold `theta`, and the block-local adaptive sweep count with
its `r/r0 < 1/3` stopping rule (Algorithm 2) — is absent. Those are
performance optimisations layered on this V-cycle; the convergence rates above
are without them.

---

# Why the adaptive path's velocity field did not settle

**Root cause: 8 particles per cell is too few for probabilistic coarsening at
`alpha_phi = 1`.** Not a discretisation bug.

## Isolating it

A hydrostatic pool, 15 steps, watching peak liquid speed:

| configuration | \|v\| trajectory | verdict |
|---|---|---|
| uniform grid, no resample | 0.048 → 0.127 → 0.092 | settles |
| **static** adaptive grid, no resample | 0.048 → 0.126 → 0.092 | settles |
| static adaptive grid + matched particle levels | 0.048 → 0.127 → 0.092 | settles |
| full adaptive | 0.048 → **3.077** → 2.569 | diverges |

A frozen adaptive grid is fine, so the multiresolution discretisation — P2G,
G2P, the coarse-fine projection — is not implicated. Only the *transition*
breaks.

## The mechanism

Probabilistic coarsening replaces a near-perfect lattice with a **Poisson**
sample: it keeps each particle independently with probability `8^-dl`. At 8
particles per cell the surviving density has ~35% RMS fluctuation. Splitting
then clusters 8 children around each randomly-placed survivor, which keeps the
distribution lumpy at the cell scale.

A bulk liquid cell that happens to come up sparse has `phi` fall below the 0.5
air/liquid tagging threshold of paper 3.3. That cell's `beta = 1/rho` then
jumps by a factor of **1000**, which is a real 1000:1 density hole in the middle
of the liquid. The pressure solve responds to it correctly, and the resulting
spurious gradient accelerates the particles. At `alpha_FLIP = 0.985` with no
physical viscosity, nothing dissipates the injected energy, so it persists long
after the grid has stopped changing.

Mis-tagged bulk-liquid cells appear on exactly the step the resample first
fires, and track the divergence one-for-one:

| `alpha_phi` | mis-tagged bulk cells per step | \|v\| at step 12 |
|---|---|---|
| 1.00 | 0, 0, 0, **3, 107, 88, 84** | 2.48 |
| 0.50 | 0, 0, 0, 0, 0, 6, 6, 5 | 0.81 |
| 0.25 | 0 throughout | 0.0006 |

The counts match a Poisson estimate. At `alpha_phi = 1` mis-tagging needs a 75%
density deficit; `P(k <= 2 | lambda = 8) ~ 1.4%` over 32768 cells predicts a few
hundred cells, and ~100 are observed. At `alpha_phi = 0.25` it needs a 94%
deficit, predicting ~0.

## Two fixes, both principled

| `alpha_phi` | ppc | \|v\| step 1 | \|v\| step 12 | max mis-tagged |
|---|---|---|---|---|
| 1.00 | 8 | 0.062 | 2.481 | 107 |
| 1.00 | **27** | 0.025 | 0.236 | **0** |
| 1.00 | **64** | 0.014 | 0.168 | **0** |
| **0.25** | 8 | 0.000 | 0.001 | **0** |

Either raise the particle count (noise falls as `1/sqrt(ppc)`) or lower
`alpha_phi`, which raises `phi = sqrt(rho~ / (alpha_phi rho~_0l))` and so widens
the margin above the 0.5 threshold. `alpha_phi` is exactly the knob paper 3.3
documents for this: "controls the trade-off between noisiness (higher values)
and stiffness (lower values) of the phase field". The paper's own
Rayleigh-Taylor setup uses 0.5 (8.2).

This also explains why stage 1 never showed the problem at 8 ppc: with no
resampling the seeded lattice stays a lattice and is never Poisson-thinned.

**Rule:** with adaptive particles, use `ppc >= 27` at the default
`alpha_phi = 1`, or drop `alpha_phi` to ~0.25 if particle count is the binding
constraint. `AdaptiveSimulator` warns when the combination is outside the
tested-safe region.

---

# `delta_phi` and `ppc` are coupled through eq. (7)

The escaped-particle test of paper 3.4 — convert a liquid particle where
`phi(x*) < 1/2 - delta_phi`, with `delta_phi = 0.2` — cannot fire at low
particle counts, because **an isolated liquid particle splats its own mass into
the phase field it is then tested against**.

A single particle contributes a kernel weight of at most 1 against a
normalisation of `rho~_0`, so eq. (7) floors it at

    phi_single = 1 / sqrt(alpha_phi * rho~_0),     rho~_0 ~ 0.638 * ppc

Conversion requires `phi_single < 1/2 - delta_phi`, i.e.

    ppc  >  1 / (alpha_phi * 0.638 * (1/2 - delta_phi)^2)

At the paper's defaults that is **ppc > 17.4**:

| ppc | `phi_single` | threshold `0.5 - delta_phi` | escaped particles convert? |
|---|---|---|---|
| 8 | 0.45 | 0.30 | **no** |
| 27 | 0.24 | 0.30 | yes |
| 64 | 0.16 | 0.30 | yes |

Measured on a 64³ asteroid impact at ppc = 8, over 80 steps: `phi` at liquid
particles had minimum 0.294 and 0.01-percentile 0.41 — exactly one particle out
of 2.1 million fell below the threshold, and 1 droplet was emitted. At ppc = 27
the same scene emitted 2,018 by step 91 and the merge-back path fired 24,118
times. **The spray channel fails silently at low ppc** — nothing errors, there
is simply no spray.

This is the same binding constraint that the mis-tagging analysis above
produced from a completely different direction (Poisson noise under
probabilistic coarsening). Two independent requirements converge on `ppc >= 27`.

## What is implemented from sections 3.4 and 7

* escaped-particle classification with `delta_phi = 0.2`: liquid → droplet,
  air inside liquid → bubble → **deleted** (as the paper does), droplet
  re-entering liquid → back to a FLIP particle;
* droplets as purely Lagrangian point masses under gravity + sphere drag
  `a = 3 rho_g C_d |v_rel| v_rel / (8 rho_l r)`, with per-droplet adaptive
  sub-stepping;
* log-normal radii on emission (Babinsky & Sojka 2002);
* the evaporation ODE `dr/dt ~ -1/r`, RK2 (Sirignano 2010);
* extra spray emission in steep-velocity-gradient regions (Ihmsen et al. 2012)
  — in practice this dominates escaped-particle emission by ~100:1;
* a 16-bit-float spray density field at 2–4× the pressure resolution,
  rasterised only at render time.

Droplets share the FLIP particle arrays and are distinguished by a `state`
channel, so every conversion is a flag flip and no stream compaction is needed.

**Not implemented:** the paper's MSBG volumetric ray tracer with approximate
multiple scattering, and particle-skinning surface reconstruction.
`pfflip/render.py` is a single-scattering ray marcher over the phase field and
the spray density with one shadow ray to the sun. It shows the structure, but
at 64³ / 7M particles the spray reads as discrete beads rather than mist —
there are only ~10⁴ droplets where the paper has billions of particles. That
gap is the paper's motivation for adaptivity and scale, not something a
renderer can close.
