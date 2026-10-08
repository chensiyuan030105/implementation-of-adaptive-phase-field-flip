#!/usr/bin/env python3
"""Grid-refinement study for the multiresolution Poisson operator.

Solves a manufactured problem ``p = cos(2 pi x) cos(2 pi y) cos(2 pi z)`` (which
satisfies homogeneous Neumann on the unit box, matching the singular operator)
on refinement maps that deliberately contain coarse-fine boundaries, and
reports the convergence rate.  A wrong transition flux shows up as a stalled
rate, so this is the acceptance criterion for paper section 4.2's
discretisation.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pfflip import mr_operator as mro  # noqa: E402
from pfflip.mr_solver import MRSolver  # noqa: E402
from pfflip.msbg import MSBG, Grid, cell_center  # noqa: E402

DEV = "cuda:0"
K = 2 * np.pi


@wp.kernel
def _centres(g: Grid, c: wp.array(dtype=wp.vec3)):
    a = wp.tid()
    c[a] = cell_center(g, a)


def levels_for(grid: MSBG, pattern: str):
    bx, by, bz = np.meshgrid(*[np.arange(n) for n in grid.nb], indexing="ij")
    if pattern == "uniform":
        lv = np.zeros_like(bx)
    elif pattern == "slab":
        lv = np.where(bx < grid.nb[0] // 2, 0, 1)
    elif pattern == "checker":
        lv = (bx + by + bz) % 2
    elif pattern == "shell":
        c = (np.array(grid.nb) - 1) / 2.0
        d = np.sqrt((bx - c[0]) ** 2 + (by - c[1]) ** 2 + (bz - c[2]) ** 2)
        lv = np.clip(np.round(d / max(grid.nb) * 3).astype(int), 0, grid.max_level)
    else:
        raise ValueError(pattern)
    return lv.ravel().astype(np.int32)


def run(nb, n0, pattern, max_level=1, tol=1e-10):
    g = MSBG((nb,) * 3, n0=n0, h0=1.0 / (nb * n0), device=DEV, max_level=max_level)
    g.build(levels_for(g, pattern))

    bface = wp.array(np.ones(g.ncells * 3, np.float32), dtype=float, device=DEV)
    vol = wp.zeros(g.ncells, dtype=float, device=DEV)
    ctr = wp.zeros(g.ncells, dtype=wp.vec3, device=DEV)
    wp.launch(mro.cell_volume, dim=g.ncells, inputs=[g.g, vol], device=DEV)
    wp.launch(_centres, dim=g.ncells, inputs=[g.g, ctr], device=DEV)
    x = ctr.numpy().astype(np.float64)
    V = vol.numpy().astype(np.float64)

    exact = np.cos(K * x[:, 0]) * np.cos(K * x[:, 1]) * np.cos(K * x[:, 2])
    # A p = -integral(laplacian p) dV  =>  rhs = 3 k^2 p V
    b = 3 * K * K * exact * V
    b -= b.mean()

    rhs = wp.array(b.astype(np.float32), dtype=float, device=DEV)
    p = wp.zeros(g.ncells, dtype=float, device=DEV)
    solver = MRSolver(g, DEV)
    it, rel = solver.solve(bface, rhs, p, tol=tol, max_iter=20000)

    sol = p.numpy().astype(np.float64)
    sol -= (sol * V).sum() / V.sum()
    ref = exact - (exact * V).sum() / V.sum()
    l2 = np.sqrt(((sol - ref) ** 2 * V).sum() / V.sum()) / np.sqrt((ref**2 * V).sum() / V.sum())
    linf = np.abs(sol - ref).max() / np.abs(ref).max()
    return g, it, rel, l2, linf


def main():
    for pattern in ("uniform", "slab", "checker", "shell"):
        print(f"\n=== refinement map: {pattern} ===")
        print(f"{'finest':>8} {'cells':>10} {'compr':>7} {'cg_it':>6} {'L2':>10} {'rate':>6} {'Linf':>10}")
        prev = None
        for nb, n0 in ((2, 8), (4, 8), (8, 8), (16, 8)):
            ml = 2 if pattern == "shell" else 1
            g, it, rel, l2, linf = run(nb, n0, pattern, max_level=ml)
            rate = "" if prev is None else f"{np.log2(prev / l2):6.2f}"
            print(
                f"{g.fine_res[0]:>6}^3 {g.ncells:>10,} {g.compression:>6.2f}x "
                f"{it:>6} {l2:>10.3e} {rate:>6} {linf:>10.3e}"
            )
            prev = l2


if __name__ == "__main__":
    main()
