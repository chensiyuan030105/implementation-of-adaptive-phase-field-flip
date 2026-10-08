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
