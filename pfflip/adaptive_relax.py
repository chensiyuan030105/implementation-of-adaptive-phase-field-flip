"""Adaptive relaxation for the multigrid smoother (paper 6.4, Algorithm 2).

Fully adaptive multigrid (Kowarschik et al. 2006; Rüde 1993) concentrates
smoothing where the residual is still large.  Blocks are kept in an active set;
a block that is still above an error threshold after relaxation is re-activated
along with the neighbours its border error reaches.  The loop runs a fixed
``N_iter_max`` of 10-20 steps rather than to exhaustion, which the paper notes
is enough because the smoother only ever runs inside a preconditioner.

Two details are what make this work for *two-phase* flow:

**Contrast-equalised effective residual.**  The usual scaled residual
(residual over the matrix diagonal) is wrong here: the diagonal is ~1000x
larger in air than in water, which compresses the air phase's dynamic range by
the same factor and means air blocks would essentially never exceed the
threshold.  The paper's metric instead is the **maximum change over a block's
variables after relaxation, multiplied by the diagonal** -- the update
``delta = omega (b - Lu)/L_jj`` already carries the ``1/L_jj``, so multiplying
by the diagonal recovers a residual-like quantity on a common scale.

**Adaptive threshold.**  A fixed threshold cannot work when the preconditioner
is applied inside CG, where the residual's dynamic range falls exponentially
from iteration to iteration.  ``theta`` is therefore recomputed per application
from the log-histogram of the per-block effective residuals as
``mean(log r) + 2 std(log r)``.

GPU adaptation
--------------
Algorithm 2's inner loop (``repeat RelaxBlock until r/r0 < 1/3``) is
block-serial; on a GPU one thread cannot usefully sweep a 16^3 block.  The
block-local sweep count is therefore fixed while the adaptive *block selection*
-- the part that actually saves memory bandwidth -- is kept.  The paper states
this count "is not critical for overall convergence, because the adaptive
smoother functions solely within a preconditioner".

Neighbour re-activation uses the whole-block maximum rather than per-face
border maxima.  That is strictly more conservative (it activates a superset),
so it costs some efficiency but cannot cost convergence.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .msbg import Grid, block_index, cell_origin_finest
from . import mr_operator as mro


# ---------------------------------------------------------------------------
@wp.kernel
def init_active(
    g: Grid,
    mg_level: int,
    active: wp.array(dtype=wp.int32),
    base_level: wp.array(dtype=wp.int32),
):
    """Seed the active set (Algorithm 2, the parallel ``for each block`` prologue).

    The paper restricts the initial set to **relaxation-relevant** blocks,
    ``l(i, L) <= L``, plus a one-block transition zone at ``L + 1``.  That
    gating relies on the paper's *shared* storage: a block that has bottomed
    out is carried between multigrid levels by reference (Algorithm 1's
    copy-by-reference case), so relaxing it at the coarser level updates the
    very memory the finer level reads.

    This implementation gives every multigrid level its own arrays, with
    explicit restriction and prolongation between them.  Skipping coarse blocks
    at a fine level would then leave their unknowns relaxed *only* at the
    coarser level, with the correction arriving through piecewise-constant
    injection -- measured cost: iteration counts went 9 -> 37, 11 -> 81,
    13 -> 129.  So all blocks start active here, and the saving comes purely
    from residual-driven deactivation, which is where the bandwidth win is
    anyway (the active set still falls to ~10% after two sweeps, matching the
    paper's fig. 12).
    """
    s = wp.tid()
    active[s] = 1
    base_level[s] = 1


@wp.kernel
def rbgs_sweep_active(
    g: Grid,
    p: wp.array(dtype=float),
    b: wp.array(dtype=float),
    bface: wp.array(dtype=float),
    diag: wp.array(dtype=float),
    active: wp.array(dtype=wp.int32),
    color: int,
    omega: float,
    delta: wp.array(dtype=float),
):
    """Weighted red-black Gauss-Seidel restricted to active blocks.

    Records ``|delta|`` per cell so the caller can form the effective residual.
    """
    a = wp.tid()
    if active[g.cslot[a]] == 0:
        return
    o = cell_origin_finest(g, a)
    if (((o[0] >> o[3]) + (o[1] >> o[3]) + (o[2] >> o[3])) & 1) != color:
        return
    d = diag[a]
    if d <= 0.0:
        return
    upd = omega * (b[a] - mro.apply_row(g, p, bface, a, 0)) / d
    p[a] += upd
    delta[a] = wp.abs(upd)


@wp.kernel
def block_effective_residual(
    g: Grid,
    delta: wp.array(dtype=float),
    diag: wp.array(dtype=float),
    active: wp.array(dtype=wp.int32),
    out: wp.array(dtype=float),
):
    """Per-block max of ``|delta| * diagonal`` -- the contrast-equalised metric.

    ``delta`` already contains ``1/L_jj``; multiplying it back by the diagonal
    puts air and water blocks on a comparable scale, which the plain scaled
    residual does not do at a 1000:1 density ratio.
    """
    a = wp.tid()
    s = g.cslot[a]
    if active[s] == 0:
        return
    wp.atomic_max(out, s, delta[a] * diag[a])


@wp.kernel
def update_active(
    g: Grid,
    block_res: wp.array(dtype=float),
    base_level: wp.array(dtype=wp.int32),
    theta: float,
    active_next: wp.array(dtype=wp.int32),
):
    """Re-activate blocks still above ``theta``, and their face neighbours."""
    s = wp.tid()
    if block_res[s] <= theta:
        return
    active_next[s] = 1
    b = g.blk[s]
    bz = b % g.nbz
    by = (b / g.nbz) % g.nby
    bx = b / (g.nbz * g.nby)
    for d in range(6):
        ax = d / 2
        sgn = wp.where(d % 2 == 0, -1, 1)
        nx = bx + wp.where(ax == 0, sgn, 0)
        ny = by + wp.where(ax == 1, sgn, 0)
        nz = bz + wp.where(ax == 2, sgn, 0)
        if nx < 0 or ny < 0 or nz < 0 or nx >= g.nbx or ny >= g.nby or nz >= g.nbz:
            continue
        ns = g.slot[block_index(g, nx, ny, nz)]
        if ns >= 0 and base_level[ns] == 1:
            active_next[ns] = 1


# ---------------------------------------------------------------------------
class AdaptiveSmoother:
    """Per-multigrid-level state for Algorithm 2."""

    def __init__(self, grid, device, *, omega=6.0 / 7.0, n_iter_max=12,
                 inner_sweeps=1, theta_sigma=2.0, warmup_sweeps=2):
        self.g = grid
        self.dev = device
        self.omega = omega
        self.n_iter_max = n_iter_max
        self.inner_sweeps = inner_sweeps
        self.theta_sigma = theta_sigma
        # Guaranteed full sweeps before adaptive deactivation starts.  Without
        # these the smoother can do *less* work than the plain one: theta is
        # mean + 2 sigma of a log-histogram over the per-block maxima, and with
        # only 64-4096 blocks that often lands above the sample maximum, so the
        # whole active set dies after one sweep.
        self.warmup_sweeps = warmup_sweeps

        ns, nc = grid.nslots, grid.ncells
        self.active = wp.zeros(ns, dtype=wp.int32, device=device)
        self.active_next = wp.zeros(ns, dtype=wp.int32, device=device)
        self.base_level = wp.zeros(ns, dtype=wp.int32, device=device)
        self.block_res = wp.zeros(ns, dtype=float, device=device)
        self.delta = wp.zeros(nc, dtype=float, device=device)
        self.active_fraction: list[float] = []

    def _sweep(self, p, b, bface, diag, reverse=False):
        self.delta.zero_()
        colors = (1, 0) if reverse else (0, 1)
        for _ in range(self.inner_sweeps):
            for c in colors:
                wp.launch(
                    rbgs_sweep_active,
                    dim=self.g.ncells,
                    inputs=[self.g.g, p, b, bface, diag, self.active, c, self.omega, self.delta],
                    device=self.dev,
                )

    def smooth(self, p, b, bface, diag, mg_level, record=False):
        """Run the adaptive relaxation loop at one multigrid level."""
        g = self.g
        wp.launch(init_active, dim=g.nslots, inputs=[g.g, mg_level, self.active, self.base_level],
                  device=self.dev)

        for w in range(self.warmup_sweeps):
            self._sweep(p, b, bface, diag, reverse=(w % 2 == 1))

        theta = None
        for it in range(self.n_iter_max):
            if record:
                # only when profiling: this forces a device sync per iteration
                self.active_fraction.append(
                    int(self.active.numpy().sum()) / max(g.nslots, 1))

            self._sweep(p, b, bface, diag, reverse=(it % 2 == 1))

            self.block_res.zero_()
            wp.launch(block_effective_residual, dim=g.ncells,
                      inputs=[g.g, self.delta, diag, self.active, self.block_res],
                      device=self.dev)

            if theta is None:
                # adaptive threshold from the log-histogram of the *initial*
                # effective residual (paper 6.4).  Read back once per smoother
                # call, not once per iteration -- the readback is a device sync.
                r = self.block_res.numpy()
                pos = r[r > 0]
                if pos.size == 0:
                    break
                lg = np.log(pos)
                theta = float(np.exp(lg.mean() + self.theta_sigma * lg.std()))

            self.active_next.zero_()
            wp.launch(update_active, dim=g.nslots,
                      inputs=[g.g, self.block_res, self.base_level, theta, self.active_next],
                      device=self.dev)
            self.active, self.active_next = self.active_next, self.active
        return theta
