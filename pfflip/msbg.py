"""Multiresolution Sparse Block Grids (paper section 4).

MSBG inverts the usual tree-based scheme: the **geometric size of a block is
constant**, only its **resolution** varies.  A block at refinement level ``l``
has ``n = n0 >> l`` cells per edge and spacing ``h_l = h0 * 2**l`` -- so level 0
is the *finest* and increasing ``l`` means coarser (paper 4.2).  Because block
extent is uniform, every block has exactly **6 face neighbours** regardless of
resolution, which is what makes stencils, grading and multi-colour schemes
cheap (Table 1).

Layout
------
* ``level``  -- dense array over the block raster, one refinement level per
  block (``INACTIVE`` for empty blocks).  This is the paper's dense linear
  block index; at ``n0 = 16`` it costs ~2% of memory (paper 4.1, curve d).
* Active blocks are compacted into *slots*, sorted by **Morton code** for
  spatially coherent traversal (paper 4.3).
* Cell data lives in one flat array; ``coff[slot]`` is the slot's base offset,
  obtained from an exclusive prefix sum over ``(n0 >> l)**3``.  Variable-size
  allocation is what makes multiresolution pay: a level-``l`` block costs
  ``8**-l`` of a finest block.

Addressing
----------
Everything is keyed on **finest-level integer coordinates** ``F``.  Resolving
``F`` to a cell is pure shift-and-mask, as the paper notes in 5.4 step 4::

    b     = F >> log2(n0)              # which block
    l     = level[b]                   # its resolution
    local = (F & (n0 - 1)) >> l        # which cell inside it

Faces are stored per cell as its three **low**-side faces, at the cell's own
resolution.  The coefficient of the face between two cells is always read from
whichever cell owns it as a low face, so both sides see the same value and the
operator stays symmetric across resolution jumps.
"""

from __future__ import annotations

import numpy as np
import warp as wp

INACTIVE = wp.constant(-1)


# ---------------------------------------------------------------------------
@wp.struct
class Grid:
    """GPU-side view of an MSBG.  Passed by value into kernels."""

    nbx: int
    nby: int
    nbz: int
    n0: int
    sh0: int  # log2(n0)
    h0: float
    nslots: int
    ncells: int

    level: wp.array(dtype=wp.int32)  # dense block raster -> refinement level
    slot: wp.array(dtype=wp.int32)  # dense block raster -> compacted slot
    blk: wp.array(dtype=wp.int32)  # slot -> dense block index
    blev: wp.array(dtype=wp.int32)  # slot -> refinement level
    coff: wp.array(dtype=wp.int32)  # slot -> first cell address
    cslot: wp.array(dtype=wp.int32)  # cell address -> slot


@wp.struct
class Cell:
    addr: int  # flat cell address, -1 if absent
    slot: int
    level: int


# --------------------------------------------------------------- device funcs
@wp.func
def block_index(g: Grid, bx: int, by: int, bz: int) -> int:
    return (bx * g.nby + by) * g.nbz + bz


@wp.func
def locate(g: Grid, fx: int, fy: int, fz: int) -> Cell:
    """Resolve a finest-level coordinate to the cell containing it."""
    c = Cell()
    c.addr = -1
    c.slot = -1
    c.level = -1
    if fx < 0 or fy < 0 or fz < 0:
        return c
    bx = fx >> g.sh0
    by = fy >> g.sh0
    bz = fz >> g.sh0
    if bx >= g.nbx or by >= g.nby or bz >= g.nbz:
        return c
    s = g.slot[block_index(g, bx, by, bz)]
    if s < 0:
        return c
    l = g.blev[s]
    n = g.n0 >> l
    m = g.n0 - 1
    i = (fx & m) >> l
    j = (fy & m) >> l
    k = (fz & m) >> l
    c.addr = g.coff[s] + (i * n + j) * n + k
    c.slot = s
    c.level = l
    return c


@wp.func
def cell_coords(g: Grid, addr: int) -> wp.vec4i:
    """Flat address -> ``(slot, i, j, k)`` local block coordinates."""
    s = g.cslot[addr]
    l = g.blev[s]
    n = g.n0 >> l
    r = addr - g.coff[s]
    i = r / (n * n)
    r = r - i * n * n
    j = r / n
    k = r - j * n
    return wp.vec4i(s, i, j, k)


@wp.func
def cell_origin_finest(g: Grid, addr: int) -> wp.vec4i:
    """Flat address -> finest-level coordinate of the cell's low corner, plus level."""
    c = cell_coords(g, addr)
    s = c[0]
    l = g.blev[s]
    b = g.blk[s]
    bz = b % g.nbz
    by = (b / g.nbz) % g.nby
    bx = b / (g.nbz * g.nby)
    return wp.vec4i(
        (bx << g.sh0) + (c[1] << l),
        (by << g.sh0) + (c[2] << l),
        (bz << g.sh0) + (c[3] << l),
        l,
    )


@wp.func
def cell_center(g: Grid, addr: int) -> wp.vec3:
    o = cell_origin_finest(g, addr)
    half = float(1 << o[3]) * 0.5
    return wp.vec3(
        (float(o[0]) + half) * g.h0,
        (float(o[1]) + half) * g.h0,
        (float(o[2]) + half) * g.h0,
    )


@wp.func
def locate_at(g: Grid, x: wp.vec3) -> Cell:
    """Resolve a world position to its cell."""
    return locate(
        g,
        int(wp.floor(x[0] / g.h0)),
        int(wp.floor(x[1] / g.h0)),
        int(wp.floor(x[2] / g.h0)),
    )


# ---------------------------------------------------------------------------
def _morton3(x, y, z):
    """Interleave three 21-bit integers (vectorised over numpy arrays)."""

    def part1by2(v):
        v = v.astype(np.uint64) & np.uint64(0x1FFFFF)
        v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
        v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
        v = (v | (v << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
        v = (v | (v << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
        v = (v | (v << np.uint64(2))) & np.uint64(0x1249249249249249)
        return v

    return part1by2(x) | (part1by2(y) << np.uint64(1)) | (part1by2(z) << np.uint64(2))


class MSBG:
    """Host-side MSBG topology manager.

    One instance owns the block raster and rebuilds the cell allocation
    whenever the refinement map changes (paper 5.4 steps 1-3).
    """

    def __init__(self, nblocks, n0=16, h0=1.0, device="cuda:0", max_level=None):
        self.nb = tuple(int(v) for v in nblocks)
        self.n0 = int(n0)
        self.sh0 = int(np.log2(n0))
        assert 1 << self.sh0 == self.n0, "n0 must be a power of two"
        self.h0 = float(h0)
        self.device = device
        # L_max = log2(n0); the paper uses 3-4 levels in all results (4.2)
        self.max_level = self.sh0 if max_level is None else int(max_level)

        self.nblk = self.nb[0] * self.nb[1] * self.nb[2]
        self.fine_res = tuple(n * self.n0 for n in self.nb)
        self._precompute_morton_order()

        self.level_h = np.full(self.nblk, -1, np.int32)
        self.g = Grid()
        self.g.nbx, self.g.nby, self.g.nbz = self.nb
        self.g.n0, self.g.sh0, self.g.h0 = self.n0, self.sh0, self.h0

    def _precompute_morton_order(self):
        bx, by, bz = np.meshgrid(
            np.arange(self.nb[0]), np.arange(self.nb[1]), np.arange(self.nb[2]), indexing="ij"
        )
        self._morton = _morton3(bx.ravel(), by.ravel(), bz.ravel())

    # ------------------------------------------------------------------ grading
    def grade(self, level_h: np.ndarray) -> np.ndarray:
        """Enforce the 2:1 grading constraint (paper 4.2, 5.4 step 2).

        Neighbouring active blocks may differ by at most one refinement level.
        Iterated until stable; with ``max_level`` small this converges in a
        handful of passes.
        """
        lv = level_h.reshape(self.nb).copy()
        active = lv >= 0
        work = np.where(active, lv, self.max_level + 1).astype(np.int16)
        for _ in range(self.max_level * 3 + 2):
            nb = work.copy()
            for axis in range(3):
                for shift in (-1, 1):
                    nb = np.minimum(nb, np.roll(work, shift, axis=axis) + 1)
            if np.array_equal(nb, work):
                break
            work = nb
        out = np.where(active, np.minimum(work, self.max_level), -1).astype(np.int32)
        return out.ravel()

    # ------------------------------------------------------------------ build
    def build(self, level_h: np.ndarray, *, grade=True):
        """Install a refinement map and (re)allocate cell storage.

        Returns the number of cells.  Active blocks are emitted in Morton order
        so that neighbouring blocks land near each other in memory (paper 4.3).
        """
        level_h = np.asarray(level_h, np.int32).ravel()
        assert level_h.size == self.nblk
        if grade:
            level_h = self.grade(level_h)
        self.level_h = level_h

        act = np.flatnonzero(level_h >= 0)
        act = act[np.argsort(self._morton[act], kind="stable")]  # Morton traversal
        lev = level_h[act]
        per_block = (self.n0 >> lev).astype(np.int64) ** 3
        coff = np.concatenate(([0], np.cumsum(per_block)))
        ncells = int(coff[-1])

        slot = np.full(self.nblk, -1, np.int32)
        slot[act] = np.arange(len(act), dtype=np.int32)
        cslot = np.repeat(np.arange(len(act), dtype=np.int32), per_block)

        d = self.device
        self.g.level = wp.array(level_h, dtype=wp.int32, device=d)
        self.g.slot = wp.array(slot, dtype=wp.int32, device=d)
        self.g.blk = wp.array(act.astype(np.int32), dtype=wp.int32, device=d)
        self.g.blev = wp.array(lev, dtype=wp.int32, device=d)
        self.g.coff = wp.array(coff[:-1].astype(np.int32), dtype=wp.int32, device=d)
        self.g.cslot = wp.array(cslot, dtype=wp.int32, device=d)
        self.g.nslots = len(act)
        self.g.ncells = ncells

        self.nslots = len(act)
        self.ncells = ncells
        self.active_blocks = act
        self.active_levels = lev
        return ncells

    # ------------------------------------------------------------------ helpers
    def zeros(self, dtype=float):
        return wp.zeros(self.ncells, dtype=dtype, device=self.device)

    @property
    def dense_cells(self) -> int:
        """Cell count an equivalent uniform finest grid would need."""
        return self.nblk * self.n0**3

    @property
    def compression(self) -> float:
        return self.dense_cells / max(self.ncells, 1)

    def level_histogram(self) -> dict[int, int]:
        lv, cnt = np.unique(self.active_levels, return_counts=True)
        return {int(a): int(b) for a, b in zip(lv, cnt)}

    def __repr__(self):
        return (
            f"MSBG(blocks={self.nb}, n0={self.n0}, fine_res={self.fine_res}, "
            f"active={getattr(self, 'nslots', 0)}, cells={getattr(self, 'ncells', 0):,}, "
            f"compression={self.compression:.2f}x)"
        )
