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
