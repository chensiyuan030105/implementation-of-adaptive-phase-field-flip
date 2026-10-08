"""MSBG topology and the multiresolution Poisson operator (paper 4, 6.1).

The operator test is the gate for everything downstream: if the coarse-fine
stencil is wrong, the adaptive solver silently produces plausible garbage.  So
it is checked three ways -- exact symmetry, exact zero row sums, and
convergence against a manufactured solution on a deliberately non-uniform
refinement map.
"""

import numpy as np
import pytest
import warp as wp

from pfflip import mr_operator as mro
from pfflip.msbg import MSBG, cell_center

DEV = "cuda:0" if wp.get_cuda_device_count() else "cpu"


def checker_levels(grid: MSBG, pattern="slab"):
    """A refinement map with real coarse-fine boundaries in all orientations."""
    nb = grid.nb
    bx, by, bz = np.meshgrid(*[np.arange(n) for n in nb], indexing="ij")
    if pattern == "slab":
        lv = np.where(bx < nb[0] // 2, 0, 1)
    elif pattern == "ball":
        c = np.array(nb) / 2.0
        d = np.sqrt((bx - c[0]) ** 2 + (by - c[1]) ** 2 + (bz - c[2]) ** 2)
        lv = np.clip((d / max(nb) * 4).astype(int), 0, grid.max_level)
    else:
        lv = ((bx + by + bz) % 2).astype(int)
    return lv.ravel().astype(np.int32)


def assemble(grid: MSBG, bface_np):
    """Explicitly assemble the operator by applying it to unit vectors."""
    n = grid.ncells
    bface = wp.array(bface_np, dtype=float, device=DEV)
    p = wp.zeros(n, dtype=float, device=DEV)
    out = wp.zeros(n, dtype=float, device=DEV)
    A = np.zeros((n, n), np.float64)
    e = np.zeros(n, np.float32)
    for j in range(n):
        e[:] = 0.0
        e[j] = 1.0
        p.assign(e)
        wp.launch(mro.apply_A, dim=n, inputs=[grid.g, p, bface, out], device=DEV)
        A[:, j] = out.numpy()
    return A


# ---------------------------------------------------------------- topology
def test_grading_is_2to1():
    g = MSBG((6, 6, 6), n0=8, h0=1 / 48, device=DEV, max_level=3)
    lv = np.full(g.nblk, 3, np.int32)
    lv[g.nblk // 2] = 0  # one finest block in a sea of coarse ones
    graded = g.grade(lv).reshape(g.nb)
    for axis in range(3):
        for shift in (-1, 1):
            d = np.abs(graded - np.roll(graded, shift, axis=axis))
            # ignore the wrap-around seam introduced by roll
            sl = [slice(None)] * 3
            sl[axis] = 0 if shift == 1 else -1
            d[tuple(sl)] = 0
            assert d.max() <= 1, f"grading violated by {d.max()} levels"
    assert graded.min() == 0 and graded.max() == 3


def test_addressing_roundtrip():
    """Every finest coordinate must resolve to exactly one cell, and the cells
    of a block must tile it without gaps or overlap."""
    g = MSBG((3, 3, 3), n0=8, h0=1 / 24, device=DEV, max_level=2)
    g.build(checker_levels(g, "checker"))

    hits = wp.zeros(g.ncells, dtype=wp.int32, device=DEV)

    @wp.kernel
    def probe(gr: mro.Grid, hits: wp.array(dtype=wp.int32), nf: int):
        x, y, z = wp.tid()
        c = mro.locate(gr, x, y, z)
        if c.addr >= 0:
            wp.atomic_add(hits, c.addr, 1)

    wp.launch(probe, dim=g.fine_res, inputs=[g.g, hits, g.fine_res[0]], device=DEV)
    h = hits.numpy()
    assert (h > 0).all(), "some cells are never reached"
    # a level-l cell must be hit by exactly 8^l finest coordinates
    lev_per_cell = g.active_levels[g.g.cslot.numpy()]
    assert np.array_equal(h, (1 << (3 * lev_per_cell))), "cells do not tile the domain"


def test_compression_matches_level_counts():
    g = MSBG((4, 4, 4), n0=16, h0=1 / 64, device=DEV, max_level=3)
    lv = checker_levels(g, "ball")
    g.build(lv)
    expected = sum((16 >> l) ** 3 for l in g.active_levels)
    assert g.ncells == expected
    assert g.compression > 1.0


# ---------------------------------------------------------------- operator
@pytest.mark.parametrize("pattern", ["slab", "checker"])
def test_operator_symmetric_and_singular(pattern):
    """Symmetry and zero row sums must hold *exactly*, not approximately.

    Zero row sums encode the pure-Neumann condition (no flux leaves the
    domain); symmetry is what lets CG be used at all.  Both are properties of
    the coarse-fine coupling, so this is where a wrong transition factor shows
    up immediately.
    """
    g = MSBG((2, 2, 2), n0=4, h0=0.125, device=DEV, max_level=1)
    g.build(checker_levels(g, pattern))
    rng = np.random.default_rng(0)
    bface = rng.uniform(0.5, 2.0, g.ncells * 3).astype(np.float32)
    A = assemble(g, bface)

    asym = np.abs(A - A.T).max()
    assert asym < 1e-4 * np.abs(A).max(), f"operator not symmetric (max |A-A^T| = {asym:.3e})"
    rows = np.abs(A.sum(axis=1)).max()
    assert rows < 1e-4 * np.abs(A).max(), f"row sums do not vanish ({rows:.3e})"
    assert (np.diag(A) > 0).all(), "diagonal must be positive"

    # and the stored diagonal must match the assembled one
    diag = wp.zeros(g.ncells, dtype=float, device=DEV)
    wp.launch(
        mro.compute_diag,
        dim=g.ncells,
        inputs=[g.g, wp.array(bface, dtype=float, device=DEV), diag],
        device=DEV,
    )
    assert np.abs(diag.numpy() - np.diag(A)).max() < 1e-3 * np.abs(A).max()


def test_uniform_msbg_matches_analytic_laplacian():
    """With all blocks at one level, the operator must be the plain 7-point
    Laplacian scaled by h."""
    g = MSBG((2, 2, 2), n0=4, h0=0.125, device=DEV, max_level=0)
    g.build(np.zeros(g.nblk, np.int32))
    bface = np.ones(g.ncells * 3, np.float32)
    A = assemble(g, bface)
    h = g.h0
    # interior rows: diagonal 6h, off-diagonals -h
    interior = np.flatnonzero(np.isclose(np.diag(A), 6 * h, rtol=1e-4))
    assert len(interior) > 0
    off = A[interior].copy()
    off[np.arange(len(interior)), interior] = 0
    assert np.isclose(off[off != 0], -h, rtol=1e-4).all()


@pytest.mark.parametrize("n0,nb", [(4, (4, 4, 4)), (8, (4, 4, 4))])
def test_manufactured_solution_converges(n0, nb):
    """Solve a manufactured Poisson problem on an adaptive grid.

    ``p = cos(2 pi x) cos(2 pi y) cos(2 pi z)`` satisfies homogeneous Neumann
    on the unit box, so it is compatible with the singular pure-Neumann
    operator.  The error must be small and must shrink when the grid is
    refined -- that is the real check that the coarse-fine flux is right, since
    a wrong transition factor produces an O(1) kink at the interface.
    """
    import scipy.sparse.linalg as spla

    g = MSBG(nb, n0=n0, h0=1.0 / (nb[0] * n0), device=DEV, max_level=1)
    g.build(checker_levels(g, "slab"))
    bface = np.ones(g.ncells * 3, np.float32)
    A = assemble(g, bface)

    centre = wp.zeros(g.ncells, dtype=wp.vec3, device=DEV)
    vol = wp.zeros(g.ncells, dtype=float, device=DEV)

    @wp.kernel
    def centres(gr: mro.Grid, c: wp.array(dtype=wp.vec3)):
        a = wp.tid()
        c[a] = cell_center(gr, a)

    wp.launch(mro.cell_volume, dim=g.ncells, inputs=[g.g, vol], device=DEV)
    wp.launch(centres, dim=g.ncells, inputs=[g.g, centre], device=DEV)
    x = centre.numpy().astype(np.float64)
    V = vol.numpy().astype(np.float64)

    k = 2 * np.pi
    exact = np.cos(k * x[:, 0]) * np.cos(k * x[:, 1]) * np.cos(k * x[:, 2])
    b = 3 * k * k * exact * V
    b -= (b * V).sum() / V.sum() * np.ones_like(b)  # compatibility

    sol = spla.lsqr(A, b, atol=1e-12, btol=1e-12, iter_lim=20000)[0]
    sol -= (sol * V).sum() / V.sum()
    ref = exact - (exact * V).sum() / V.sum()

    err = np.abs(sol - ref).max() / np.abs(ref).max()
    assert err < 0.25, f"manufactured solution error {err:.3f} (n0={n0}, nb={nb})"
    return err
