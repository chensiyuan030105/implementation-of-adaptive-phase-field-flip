"""Adaptive particles and dual particle-grid adaptivity (paper section 5).

Each particle carries a **level** ``l_p`` setting its radius ``r_p = r_0 2**l_p``
and hence its volume ``2**(3 l_p)`` and mass.  Level 0 is finest, matching the
MSBG convention.

The two transitions are deliberately asymmetric:

* **Refine** (``l_new < l_p``): split into 8 particles of level ``l_p - 1``,
  jittered within the octants of the containing cell (Adams et al. 2007).
* **Coarsen** (``l_new > l_p``): *probabilistic coarsening* (paper 5.1) --
  delete each marked particle with probability
  ``prob_del = 1 - 1/(2**(l_new - l_p))**3`` and promote the survivors.
  Coarsening a uniform distribution by 2x per axis means 8x fewer particles,
  which random deletion achieves **without any neighbourhood search**.  Keeping
  FLIP neighbour-free is the whole point: merging schemes would reintroduce the
  search that FLIP exists to avoid.

A deferred-coarsening counter ``D_p`` provides hysteresis: coarsening only
fires once ``D_p >= 3``, and resets whenever the criterion stops asking for it
(paper 5.4 step 6).  Without it particles oscillate across the band boundary.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .msbg import Grid, locate_at

AIR = wp.constant(0)
LIQUID = wp.constant(1)


# ---------------------------------------------------------------------------
@wp.func
def level_radius(r0: float, level: int) -> float:
    return r0 * float(1 << level)


@wp.func
def level_volume(level: int) -> float:
    return float(1 << (3 * level))


# ---------------------------------------------------------------------------
# 1. block refinement map from particles (paper 5.4 step 1)
# ---------------------------------------------------------------------------
@wp.kernel
def atomic_min_refinement_map(
    px: wp.array(dtype=wp.vec3),
    plev: wp.array(dtype=wp.int32),
    level_map: wp.array(dtype=wp.int32),
    nbx: int,
    nby: int,
    nbz: int,
    block_extent: float,
    cfl_eps: float,
):
    """``M[i] <- min(l_p, M[i])`` over the blocks a particle's (enlarged)
    bounding box touches.

    The box is grown by a CFL-dependent epsilon so that a block is already
    refined by the time a fast particle arrives in it -- at the CFL of 3-4 the
    paper uses, a particle can cross a whole resolution boundary in one step.
    """
    p = wp.tid()
    x = px[p]
    l = plev[p]
    rad = cfl_eps
    for a in range(3):
        pass
    lo0 = int(wp.floor((x[0] - rad) / block_extent))
    hi0 = int(wp.floor((x[0] + rad) / block_extent))
    lo1 = int(wp.floor((x[1] - rad) / block_extent))
    hi1 = int(wp.floor((x[1] + rad) / block_extent))
    lo2 = int(wp.floor((x[2] - rad) / block_extent))
    hi2 = int(wp.floor((x[2] + rad) / block_extent))
    for i in range(wp.max(lo0, 0), wp.min(hi0, nbx - 1) + 1):
        for j in range(wp.max(lo1, 0), wp.min(hi1, nby - 1) + 1):
            for k in range(wp.max(lo2, 0), wp.min(hi2, nbz - 1) + 1):
                wp.atomic_min(level_map, (i * nby + j) * nbz + k, l)


# ---------------------------------------------------------------------------
# 2. per-particle level update (paper 5.4 step 6)
# ---------------------------------------------------------------------------
@wp.kernel
def classify_particles(
    px: wp.array(dtype=wp.vec3),
    plev: wp.array(dtype=wp.int32),
    pdefer: wp.array(dtype=wp.int32),
    dist_blocks: wp.array3d(dtype=float),
    lnew_out: wp.array(dtype=wp.int32),
    nout: wp.array(dtype=wp.int32),
    block_extent: float,
    beta_bw: float,
    max_level: int,
    defer_threshold: int,
    rand_seed: int,
):
    """Evaluate the refinement criterion and decide split / keep / coarsen.

    ``l_new = floor(|beta_bw * d~(x_p)| / m_0)`` with ``m_0`` the MSBG base
    block resolution (paper 5.4 step 6).  Since ``d~`` is carried here in
    *block* units, the division by ``m_0`` is already folded in and
    ``beta_bw`` directly sets how many blocks wide each refinement band is.

    Writes the per-particle output count so the caller can prefix-sum it into
    a compaction: 0 = deleted, 1 = kept, 8 = split.
    """
    p = wp.tid()
    x = px[p]
    lp = plev[p]

    # trilinear lookup in the reduced-resolution distance field
    gx = x[0] / block_extent - 0.5
    gy = x[1] / block_extent - 0.5
    gz = x[2] / block_extent - 0.5
    i0 = int(wp.floor(gx))
    j0 = int(wp.floor(gy))
    k0 = int(wp.floor(gz))
    tx = gx - float(i0)
    ty = gy - float(j0)
    tz = gz - float(k0)
    nx = dist_blocks.shape[0]
    ny = dist_blocks.shape[1]
    nz = dist_blocks.shape[2]
    d = float(0.0)
    for a in range(2):
        wa = wp.where(a == 0, 1.0 - tx, tx)
        ii = wp.clamp(i0 + a, 0, nx - 1)
        for b in range(2):
            wb = wp.where(b == 0, 1.0 - ty, ty)
            jj = wp.clamp(j0 + b, 0, ny - 1)
            for c in range(2):
                wc = wp.where(c == 0, 1.0 - tz, tz)
                kk = wp.clamp(k0 + c, 0, nz - 1)
                d += wa * wb * wc * dist_blocks[ii, jj, kk]

    lnew = wp.clamp(int(wp.floor(wp.abs(beta_bw * d))), 0, max_level)
    lnew_out[p] = lnew

    if lnew < lp:
        pdefer[p] = 0
        nout[p] = 8  # split into 8 children of level lp - 1
        return
    if lnew > lp:
        dp = pdefer[p] + 1
        pdefer[p] = dp
        if dp >= defer_threshold:
            # probabilistic coarsening, paper 5.1
            state = wp.rand_init(rand_seed, p)
            prob_del = 1.0 - 1.0 / wp.pow(2.0, 3.0 * float(lnew - lp))
            if wp.randf(state) < prob_del:
                nout[p] = 0
                return
        nout[p] = 1
        return
    pdefer[p] = 0  # l_new <= l_p resets the hysteresis counter
    nout[p] = 1


@wp.kernel
def emit_particles(
    px: wp.array(dtype=wp.vec3),
    pv: wp.array(dtype=wp.vec3),
    ptype: wp.array(dtype=wp.int32),
    plev: wp.array(dtype=wp.int32),
    pdefer: wp.array(dtype=wp.int32),
    lnew: wp.array(dtype=wp.int32),
    nout: wp.array(dtype=wp.int32),
    offset: wp.array(dtype=wp.int32),
    defer_threshold: int,
    ox: wp.array(dtype=wp.vec3),
    ov: wp.array(dtype=wp.vec3),
    otype: wp.array(dtype=wp.int32),
    olev: wp.array(dtype=wp.int32),
    odefer: wp.array(dtype=wp.int32),
    r0: float,
    rand_seed: int,
):
    """Scatter survivors and split children into the compacted output arrays."""
    p = wp.tid()
    n = nout[p]
    if n == 0:
        return
    o = offset[p]
    x = px[p]
    v = pv[p]
    t = ptype[p]
    lp = plev[p]
    ln = lnew[p]

    if n == 1:
        ox[o] = x
        ov[o] = v
        otype[o] = t
        # promote only when the hysteresis counter has actually fired
        promote = ln > lp and pdefer[p] >= defer_threshold
        olev[o] = wp.where(promote, ln, lp)
        odefer[o] = wp.where(promote, 0, pdefer[p])
        return

    # split: 8 children one level finer, jittered inside the octants of the
    # cell that contained the parent (paper 5.4 step 6)
    child = lp - 1
    hp = level_radius(r0, lp)  # parent cell extent
    half = 0.5 * hp
    state = wp.rand_init(rand_seed, p)
    for c in range(8):
        sx = float(c & 1) - 0.5
        sy = float((c >> 1) & 1) - 0.5
        sz = float((c >> 2) & 1) - 0.5
        jitter = wp.vec3(
            (wp.randf(state) - 0.5) * half,
            (wp.randf(state) - 0.5) * half,
            (wp.randf(state) - 0.5) * half,
        )
        ox[o + c] = x + wp.vec3(sx * half, sy * half, sz * half) + jitter
        ov[o + c] = v
        otype[o + c] = t
        olev[o + c] = child
        odefer[o + c] = 0


@wp.kernel
def derive_mass_radius(
    ptype: wp.array(dtype=wp.int32),
    plev: wp.array(dtype=wp.int32),
    pmass: wp.array(dtype=float),
    prad: wp.array(dtype=float),
    r0: float,
    rho_l: float,
    rho_g: float,
):
    """``m_p = rho_phase * V_p`` with ``V_p = 2**(3 l_p)`` and ``r_p = r_0 2**l_p``."""
    p = wp.tid()
    lv = plev[p]
    prad[p] = level_radius(r0, lv)
    pmass[p] = wp.where(ptype[p] == LIQUID, rho_l, rho_g) * level_volume(lv)


@wp.kernel
def per_particle_alpha_flip(
    g: Grid,
    px: wp.array(dtype=wp.vec3),
    ptype: wp.array(dtype=wp.int32),
    alpha_out: wp.array(dtype=float),
    nu_liquid: float,
    nu_air: float,
    dt: float,
    alpha_cap: float,
):
    """Equalise numerical viscosity across resolution levels (paper 5.3).

    Inverting eq. (13), ``alpha_FLIP = 1 - 6 nu dt / dx^2`` with ``dx`` the
    **local grid spacing** -- crucially *not* the particle radius.  Decoupling
    the two is what lets particles cross resolution boundaries within a single
    step at the CFL of 3-4 the paper runs at.
    """
    p = wp.tid()
    c = locate_at(g, px[p])
    lv = wp.where(c.addr >= 0, c.level, 0)
    dx = g.h0 * float(1 << lv)
    nu = wp.where(ptype[p] == LIQUID, nu_liquid, nu_air)
    a = 1.0 - 6.0 * nu * dt / (dx * dx)
    alpha_out[p] = wp.clamp(a, 0.0, alpha_cap)


# ---------------------------------------------------------------------------
# host-side container
# ---------------------------------------------------------------------------
class Particles:
    """Growable particle set with the dual-adaptivity resample step."""

    def __init__(self, device, capacity: int):
        self.dev = device
        self.capacity = int(capacity)
        self.n = 0
        self._alloc(self.capacity)

    def _alloc(self, cap):
        d = self.dev
        mk = lambda dt: wp.zeros(cap, dtype=dt, device=d)  # noqa: E731
        self.x, self.v = mk(wp.vec3), mk(wp.vec3)
        self.type, self.level, self.defer = mk(wp.int32), mk(wp.int32), mk(wp.int32)
        self.mass, self.rad, self.alpha = mk(float), mk(float), mk(float)
        self._ox, self._ov = mk(wp.vec3), mk(wp.vec3)
        self._otype, self._olev, self._odefer = mk(wp.int32), mk(wp.int32), mk(wp.int32)
        self._nout, self._off, self._lnew = mk(wp.int32), mk(wp.int32), mk(wp.int32)
        self.capacity = cap

    def ensure_capacity(self, need):
        if need <= self.capacity:
            return
        new_cap = max(int(need * 1.3), self.capacity * 2)
        old = {k: getattr(self, k) for k in ("x", "v", "type", "level", "defer")}
        n = self.n
        self._alloc(new_cap)
        for k, a in old.items():
            wp.copy(getattr(self, k), a, count=n)

    def set(self, pos, vel, ptype, level=None):
        n = len(pos)
        self.ensure_capacity(n)
        self.n = n
        self.x.assign(np.ascontiguousarray(pos, np.float32))
        self.v.assign(np.ascontiguousarray(vel, np.float32))
        self.type.assign(np.ascontiguousarray(ptype, np.int32))
        lv = np.zeros(n, np.int32) if level is None else np.ascontiguousarray(level, np.int32)
        self.level.assign(lv)
        self.defer.assign(np.zeros(n, np.int32))

    def update_mass_radius(self, r0, rho_l, rho_g):
        wp.launch(
            derive_mass_radius,
            dim=self.n,
            inputs=[self.type, self.level, self.mass, self.rad, r0, rho_l, rho_g],
            device=self.dev,
        )

    # ------------------------------------------------------------------ resample
    def resample(self, dist_blocks, block_extent, *, beta_bw, max_level,
                 defer_threshold=3, seed=0):
        """One dual-adaptivity pass: classify, then compact with splits.

        Returns ``(n_before, n_after)``.
        """
        n = self.n
        wp.launch(
            classify_particles,
            dim=n,
            inputs=[self.x, self.level, self.defer, dist_blocks, self._lnew, self._nout,
                    block_extent, beta_bw, max_level, defer_threshold, seed],
            device=self.dev,
        )
        counts = self._nout.numpy()[:n]
        offs = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
        total = int(offs[-1])
        if total > self.capacity:
            # grow the output side before scattering
            self._grow_outputs(total)
        self._off.assign(np.ascontiguousarray(offs[:-1], np.int32))

        wp.launch(
            emit_particles,
            dim=n,
            inputs=[self.x, self.v, self.type, self.level, self.defer, self._lnew,
                    self._nout, self._off, defer_threshold,
                    self._ox, self._ov, self._otype, self._olev, self._odefer,
                    1.0, seed + 7919],
            device=self.dev,
        )
        self.x, self._ox = self._ox, self.x
        self.v, self._ov = self._ov, self.v
        self.type, self._otype = self._otype, self.type
        self.level, self._olev = self._olev, self.level
        self.defer, self._odefer = self._odefer, self.defer
        self.n = total
        return n, total

    def _grow_outputs(self, need):
        cap = max(int(need * 1.3), self.capacity * 2)
        keep = {k: getattr(self, k) for k in ("x", "v", "type", "level", "defer")}
        n = self.n
        self._alloc(cap)
        for k, a in keep.items():
            wp.copy(getattr(self, k), a, count=n)


# ---------------------------------------------------------------------------
# auxiliary reduced-resolution distance field (paper 5.2)
# ---------------------------------------------------------------------------
def block_distance_field(interface_mask: np.ndarray, max_dist: float = 64.0) -> np.ndarray:
    """Distance (in block units) to the nearest interface block.

    Paper 5.2: PF-FLIP keeps no distance field, so an auxiliary ``d~`` is
    generated from the **downsampled and thresholded phase field** with a
    simple **first-order accurate marching algorithm**.  Block resolution is
    deliberate -- ``d~`` is used *only* to pick particle levels, so first-order
    accuracy at block granularity is sufficient and costs almost nothing.
    """
    if not interface_mask.any():
        return np.full(interface_mask.shape, max_dist, np.float32)
    try:
        from scipy.ndimage import distance_transform_edt

        # exact Euclidean rather than the paper's first-order marching: at block
        # resolution the raster is tiny, so the better field is free
        d = distance_transform_edt(~interface_mask).astype(np.float32)
    except ImportError:
        # fallback: iterated city-block sweep.  np.roll is periodic, so the
        # seam is reset every pass to stop distance wrapping around the domain.
        d = np.where(interface_mask, 0.0, max_dist).astype(np.float32)
        for _ in range(int(max_dist)):
            prev = d
            for axis in range(3):
                for shift in (1, -1):
                    r = np.roll(d, shift, axis=axis) + 1.0
                    sl = [slice(None)] * 3
                    sl[axis] = 0 if shift == 1 else -1
                    r[tuple(sl)] = max_dist
                    d = np.minimum(d, r)
            if np.array_equal(d, prev):
                break
    return np.minimum(d, max_dist).astype(np.float32)
