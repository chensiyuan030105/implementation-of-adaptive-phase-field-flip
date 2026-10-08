"""Multiresolution multigrid on an MSBG (paper section 6).

Two hierarchies, kept independent as in paper 6.2:

* the **MSBG hierarchy**, given by each block's resolution level ``l``;
* the **multigrid hierarchy**, given by the multigrid level ``L``.

At multigrid level ``L`` a block sits at resolution ``min(l + L, log2(n0))``.
Once a block is down to a single cell it cannot coarsen further, so its
coefficients are simply carried over -- this is Algorithm 1's
``if l = level(i, L-1) then F[i,L] <- F[i,L-1]; continue  # no finer child
block``.  Because ``min(l + L, c)`` preserves ``|l_a - l_b| <= 1``, every
multigrid level inherits 2:1 grading for free.

Coarsening the coefficients
---------------------------
:mod:`pfflip.mr_operator` derives each coupling as ``beta * Area / dist`` from
the grid geometry, so only ``beta`` has to be transferred: a coarse face
coefficient is the **mean of the four fine face coefficients** it covers, which
is exactly Algorithm 1's ``s * sum / 4`` with the level-dependent factor ``s``
supplied by the geometry instead of being tabulated.

Sampling the 2x2 patch via :func:`pfflip.msbg.locate` makes the copy-by-
reference case fall out automatically: when the fine block has the same
resolution all four probes land on the same cell and the mean is that value.

Grid transfer is **unsmoothed aggregation** (paper 6.1): prolongation is
piecewise-constant injection and restriction is its transpose, i.e. summation
over the 8 fine cells of an aggregate.  Both are implemented by launching over
*fine* cells and resolving the containing coarse cell with ``locate``, so the
8:1 and 1:1 cases need no special casing.

Because coefficients are re-discretised rather than assembled variationally,
the V-cycle is only approximately symmetric -- which is precisely why the outer
Krylov method is **flexible** CG with a Polak-Ribiere update (paper 6.4).

Not implemented: the adaptive relaxation of paper 6.4 (active-block FIFO queue,
log-histogram threshold ``theta``, block-local sweep counts).  Those are
performance optimisations on top of this V-cycle, not correctness requirements.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from . import mr_operator as mro
from .msbg import MSBG, Grid, cell_origin_finest, locate


# ---------------------------------------------------------------------------
@wp.kernel
def coarsen_beta(
    cg: Grid,
    fg: Grid,
    bfine: wp.array(dtype=float),
    bcoarse: wp.array(dtype=float),
):
    """Coarse face coefficient = mean of the 2x2 fine faces it covers."""
    a = wp.tid()
    o = cell_origin_finest(cg, a)
    ext = 1 << o[3]
    half = ext >> 1  # 0 when the cell is already one finest unit across

    for axis in range(3):
        ax1 = (axis + 1) % 3
        ax2 = (axis + 2) % 3
        e1x = wp.where(ax1 == 0, 1, 0)
        e1y = wp.where(ax1 == 1, 1, 0)
        e1z = wp.where(ax1 == 2, 1, 0)
        e2x = wp.where(ax2 == 0, 1, 0)
        e2y = wp.where(ax2 == 1, 1, 0)
        e2z = wp.where(ax2 == 2, 1, 0)
        acc = float(0.0)
        cnt = float(0.0)
        for u in range(2):
            for v in range(2):
                du = u * half
                dv = v * half
                nc = locate(
                    fg,
                    o[0] + e1x * du + e2x * dv,
                    o[1] + e1y * du + e2y * dv,
                    o[2] + e1z * du + e2z * dv,
                )
                if nc.addr >= 0:
                    acc += bfine[nc.addr * 3 + axis]
                    cnt += 1.0
        bcoarse[a * 3 + axis] = wp.where(cnt > 0.0, acc / cnt, 0.0)


@wp.kernel
def restrict_sum(fg: Grid, cg: Grid, rfine: wp.array(dtype=float), bcoarse: wp.array(dtype=float)):
    """``R = P^T``: sum fine residuals into the containing coarse cell."""
    a = wp.tid()
    o = cell_origin_finest(fg, a)
    nc = locate(cg, o[0], o[1], o[2])
    if nc.addr >= 0:
        wp.atomic_add(bcoarse, nc.addr, rfine[a])


@wp.kernel
def prolong_add(fg: Grid, cg: Grid, pcoarse: wp.array(dtype=float), pfine: wp.array(dtype=float)):
    """``P``: piecewise-constant injection, accumulated onto the fine level."""
    a = wp.tid()
    o = cell_origin_finest(fg, a)
    nc = locate(cg, o[0], o[1], o[2])
    if nc.addr >= 0:
        pfine[a] += pcoarse[nc.addr]


@wp.kernel
def _sum(x: wp.array(dtype=float), out: wp.array(dtype=float)):
    i = wp.tid()
    wp.atomic_add(out, 0, x[i])


@wp.kernel
def _shift(x: wp.array(dtype=float), c: float):
    i = wp.tid()
    x[i] += c


# ---------------------------------------------------------------------------
class _MGLevel:
    def __init__(self, grid: MSBG, device):
        self.grid = grid
        n = grid.ncells
        z = lambda: wp.zeros(n, dtype=float, device=device)  # noqa: E731
        self.bface = wp.zeros(n * 3, dtype=float, device=device)
        self.diag = z()
        self.p = z()
        self.b = z()
        self.r = z()
        self.n = n


class MRMultigrid:
    """V-cycle preconditioner over the MSBG multigrid hierarchy."""

    def __init__(self, grid: MSBG, device, *, omega=6.0 / 7.0, pre=2, post=2,
                 coarsest_sweeps=80, min_cells=512):
        self.dev = device
        self.omega = omega
        self.pre = pre
        self.post = post
        self.coarsest_sweeps = coarsest_sweeps
        self._scalar = wp.zeros(1, dtype=float, device=device)

        base = np.asarray(grid.level_h, np.int32)
        cap = grid.sh0  # a block bottoms out at one cell
        self.levels: list[_MGLevel] = [_MGLevel(grid, device)]
        L = 1
        while True:
            lv = np.where(base >= 0, np.minimum(base + L, cap), -1).astype(np.int32)
            g = MSBG(grid.nb, n0=grid.n0, h0=grid.h0, device=device, max_level=cap)
            g.build(lv, grade=False)  # min(l+L, cap) preserves 2:1 grading
            if g.ncells == self.levels[-1].n:
                break  # every block already bottomed out: no further coarsening
            self.levels.append(_MGLevel(g, device))
            if g.ncells <= min_cells:
                break
            L += 1
        self.nlev = len(self.levels)

    # ----------------------------------------------------------------- utils
    def _remove_mean(self, a, n):
        self._scalar.zero_()
        wp.launch(_sum, dim=n, inputs=[a, self._scalar], device=self.dev)
        wp.launch(_shift, dim=n, inputs=[a, -float(self._scalar.numpy()[0]) / n], device=self.dev)

    # ----------------------------------------------------------------- build
    def build_hierarchy(self, bface0):
        """Install level-0 coefficients and coarsen them down the hierarchy."""
        L0 = self.levels[0]
        wp.copy(L0.bface, bface0)
        for i, lv in enumerate(self.levels):
            if i > 0:
                f = self.levels[i - 1]
                wp.launch(coarsen_beta, dim=lv.n,
                          inputs=[lv.grid.g, f.grid.g, f.bface, lv.bface], device=self.dev)
            wp.launch(mro.compute_diag, dim=lv.n,
                      inputs=[lv.grid.g, lv.bface, lv.diag], device=self.dev)

    # ----------------------------------------------------------------- cycle
    def _smooth(self, lv: _MGLevel, n_sweeps, reverse=False):
        colors = (1, 0) if reverse else (0, 1)
        for _ in range(n_sweeps):
            for c in colors:
                wp.launch(mro.rbgs_sweep, dim=lv.n,
                          inputs=[lv.grid.g, lv.p, lv.b, lv.bface, lv.diag, c, self.omega],
                          device=self.dev)

    def _vcycle(self, i=0):
        lv = self.levels[i]
        if i == self.nlev - 1:
            self._smooth(lv, self.coarsest_sweeps)
            self._remove_mean(lv.p, lv.n)
            return
        self._smooth(lv, self.pre)
        wp.launch(mro.residual, dim=lv.n,
                  inputs=[lv.grid.g, lv.p, lv.b, lv.bface, lv.r], device=self.dev)
        c = self.levels[i + 1]
        c.b.zero_()
        c.p.zero_()
        wp.launch(restrict_sum, dim=lv.n,
                  inputs=[lv.grid.g, c.grid.g, lv.r, c.b], device=self.dev)
        self._vcycle(i + 1)
        wp.launch(prolong_add, dim=lv.n,
                  inputs=[lv.grid.g, c.grid.g, c.p, lv.p], device=self.dev)
        # reversed colour order keeps the cycle close to symmetric
        self._smooth(lv, self.post, reverse=True)

    def precondition(self, r_in, z_out):
        L0 = self.levels[0]
        wp.copy(L0.b, r_in)
        L0.p.zero_()
        self._vcycle(0)
        wp.copy(z_out, L0.p)
        self._remove_mean(z_out, L0.n)

    def describe(self):
        return " -> ".join(f"{lv.n:,}" for lv in self.levels)
