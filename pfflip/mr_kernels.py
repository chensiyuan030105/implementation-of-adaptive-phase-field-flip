"""Multiresolution P2G / G2P and field kernels on an MSBG (paper 5.4, 3.2-3.7).

Face storage convention matches :mod:`pfflip.mr_operator`: every cell owns its
three **low**-side faces at ``face[addr*3 + axis]``, positioned at
``cell_centre - 0.5 * h_l * e_a``.

P2G loops over the grid **blocks** the particle's support sphere touches and
then over the cells inside each block (paper 5.4 step 4), rather than over
finest-level coordinates.  Each block carries its own resolution, so each cell
and each face is visited exactly once with no deduplication needed.  When the
particle level matches the grid level this reduces to the paper's usual 27-cell
neighbourhood; a particle one level coarser than the grid (the ``l_p = l + 1``
case the paper explicitly allows) simply covers twice as many cells per axis.

Normalisation across levels
---------------------------
At level ``l`` a matched particle has ``m_p = rho * 2**(3l)`` and
``r_p = h_l``, so a uniform lattice splats ``rho~_ai = rho~_0 * rho * 2**(3l)``.
Dividing the accumulator by ``2**(3l)`` therefore makes eq. (7) usable with the
*level-0* constants everywhere.  Face velocity ``P_ai / m_ai`` is a ratio, so
the normalisation cancels there and does not need applying.
"""

from __future__ import annotations

import warp as wp

from .msbg import Cell, Grid, block_index, cell_origin_finest, locate, locate_at

AIR = wp.constant(0)
LIQUID = wp.constant(1)


# ---------------------------------------------------------------------------
@wp.func
def splat_weight(d2: float, r2: float) -> float:
    """Paper eq. (6), cubic in the squared distance so no sqrt is needed."""
    t = wp.max(1.0 - d2 / r2, 0.0)
    return t * t * t


@wp.func
def cell_size(g: Grid, level: int) -> float:
    return g.h0 * float(1 << level)


# ---------------------------------------------------------------------------
# P2G
# ---------------------------------------------------------------------------
@wp.kernel
def p2g(
    g: Grid,
    px: wp.array(dtype=wp.vec3),
    pv: wp.array(dtype=wp.vec3),
    pmass: wp.array(dtype=float),
    prad: wp.array(dtype=float),
    fmass: wp.array(dtype=float),
    fmom: wp.array(dtype=float),
    cmass: wp.array(dtype=float),
    cwsum: wp.array(dtype=float),
):
    """Splat mass (eq. 3) and momentum (eq. 4) to the MAC faces and centres."""
    p = wp.tid()
    x = px[p]
    v = pv[p]
    mp = pmass[p]
    rp = prad[p]
    r2 = rp * rp

    bext = float(g.n0) * g.h0  # geometric block size, constant for all levels
    b0 = wp.vec3i(
        int(wp.floor((x[0] - rp) / bext)),
        int(wp.floor((x[1] - rp) / bext)),
        int(wp.floor((x[2] - rp) / bext)),
    )
    b1 = wp.vec3i(
        int(wp.floor((x[0] + rp) / bext)),
        int(wp.floor((x[1] + rp) / bext)),
        int(wp.floor((x[2] + rp) / bext)),
    )

    for bi in range(wp.max(b0[0], 0), wp.min(b1[0], g.nbx - 1) + 1):
        for bj in range(wp.max(b0[1], 0), wp.min(b1[1], g.nby - 1) + 1):
            for bk in range(wp.max(b0[2], 0), wp.min(b1[2], g.nbz - 1) + 1):
                s = g.slot[block_index(g, bi, bj, bk)]
                if s < 0:
                    continue
                l = g.blev[s]
                n = g.n0 >> l
                h = cell_size(g, l)
                ox = float(bi) * bext
                oy = float(bj) * bext
                oz = float(bk) * bext

                # local cell index range overlapping the support, widened by one
                # cell on each side so that every low face in range is reached
                i0 = wp.max(int(wp.floor((x[0] - rp - ox) / h)) - 1, 0)
                i1 = wp.min(int(wp.floor((x[0] + rp - ox) / h)) + 1, n - 1)
                j0 = wp.max(int(wp.floor((x[1] - rp - oy) / h)) - 1, 0)
                j1 = wp.min(int(wp.floor((x[1] + rp - oy) / h)) + 1, n - 1)
                k0 = wp.max(int(wp.floor((x[2] - rp - oz) / h)) - 1, 0)
                k1 = wp.min(int(wp.floor((x[2] + rp - oz) / h)) + 1, n - 1)

                base = g.coff[s]
                for i in range(i0, i1 + 1):
                    cx = ox + (float(i) + 0.5) * h
                    for j in range(j0, j1 + 1):
                        cy = oy + (float(j) + 0.5) * h
                        for k in range(k0, k1 + 1):
                            cz = oz + (float(k) + 0.5) * h
                            addr = base + (i * n + j) * n + k

                            # cell centre: raw mass density and bare weight sum
                            dx0 = cx - x[0]
                            dy0 = cy - x[1]
                            dz0 = cz - x[2]
                            wc = splat_weight(dx0 * dx0 + dy0 * dy0 + dz0 * dz0, r2)
                            if wc > 0.0:
                                wp.atomic_add(cmass, addr, wc * mp)
                                wp.atomic_add(cwsum, addr, wc)

                            # the cell's three low faces
                            half = 0.5 * h
                            for a in range(3):
                                fxp = cx - wp.where(a == 0, half, 0.0)
                                fyp = cy - wp.where(a == 1, half, 0.0)
                                fzp = cz - wp.where(a == 2, half, 0.0)
                                dx1 = fxp - x[0]
                                dy1 = fyp - x[1]
                                dz1 = fzp - x[2]
                                wf = splat_weight(dx1 * dx1 + dy1 * dy1 + dz1 * dz1, r2)
                                if wf > 0.0:
                                    wp.atomic_add(fmass, addr * 3 + a, wf * mp)
                                    wp.atomic_add(fmom, addr * 3 + a, wf * mp * v[a])


@wp.kernel
def face_velocity(
    fmass: wp.array(dtype=float),
    fmom: wp.array(dtype=float),
    fvel: wp.array(dtype=float),
    eps: float,
):
    """``u~*_ai = P_ai / m_ai``; the level normalisation cancels in the ratio."""
    i = wp.tid()
    m = fmass[i]
    fvel[i] = wp.where(m > eps, fmom[i] / m, 0.0)


# ---------------------------------------------------------------------------
# phase field (paper 3.3, eq. 7) with per-level normalisation
# ---------------------------------------------------------------------------
@wp.func
def phase_field(rho: float, rho_min: float, alpha_phi: float, rho_0l: float) -> float:
    if rho < rho_min:
        return 0.0
    return wp.min(wp.sqrt(wp.max(rho - rho_min, 0.0) / (alpha_phi * rho_0l)), 1.0)


@wp.kernel
def cell_phase(
    g: Grid,
    cmass: wp.array(dtype=float),
    phi: wp.array(dtype=float),
    is_liquid: wp.array(dtype=wp.int32),
    rho_min: float,
    alpha_phi: float,
    rho_0l: float,
):
    a = wp.tid()
    l = cell_origin_finest(g, a)[3]
    norm = float(1 << (3 * l))  # a level-l particle carries 2^(3l) the mass
    f = phase_field(cmass[a] / norm, rho_min, alpha_phi, rho_0l)
    phi[a] = f
    is_liquid[a] = wp.where(f >= 0.5, 1, 0)


@wp.kernel
def face_beta(
    g: Grid,
    fmass: wp.array(dtype=float),
    is_liquid: wp.array(dtype=wp.int32),
    fbeta: wp.array(dtype=float),
    rho_min: float,
    alpha_phi: float,
    rho_0l: float,
    rho_l: float,
    rho_g: float,
):
    """Face phase field -> density -> ``beta = 1/rho`` (eqs. 7, 9).

    ``phi = 1`` on liquid-liquid faces and ``0`` on air-air faces, so only
    interface-cutting faces use eq. (7) and the residual phase-field noise
    away from the interface is removed (paper 3.3).  Domain-boundary faces get
    ``beta = 0``, which is the homogeneous Neumann condition for pressure.
    """
    a = wp.tid()
    o = cell_origin_finest(g, a)
    l = o[3]
    ext = 1 << l
    norm = float(1 << (3 * l))
    hi = is_liquid[a]

    for axis in range(3):
        dx = wp.where(axis == 0, 1, 0)
        dy = wp.where(axis == 1, 1, 0)
        dz = wp.where(axis == 2, 1, 0)
        nb = locate(g, o[0] - dx, o[1] - dy, o[2] - dz)
        if nb.addr < 0:
            fbeta[a * 3 + axis] = 0.0  # solid wall
            continue
        lo = is_liquid[nb.addr]
        if lo == 1 and hi == 1:
            f = 1.0
        elif lo == 0 and hi == 0:
            f = 0.0
        else:
            f = phase_field(fmass[a * 3 + axis] / norm, rho_min, alpha_phi, rho_0l)
        fbeta[a * 3 + axis] = 1.0 / ((1.0 - f) * rho_g + f * rho_l)


@wp.kernel
def zero_wall_velocity(g: Grid, fbeta: wp.array(dtype=float), fvel: wp.array(dtype=float)):
    a = wp.tid()
    for axis in range(3):
        if fbeta[a * 3 + axis] == 0.0:
            fvel[a * 3 + axis] = 0.0


# ---------------------------------------------------------------------------
# forces, divergence, projection
# ---------------------------------------------------------------------------
@wp.kernel
def add_gravity(g: Grid, fvel: wp.array(dtype=float), fbeta: wp.array(dtype=float),
                gx: float, gy: float, gz: float, dt: float):
    a = wp.tid()
    for axis in range(3):
        if fbeta[a * 3 + axis] != 0.0:
            gc = wp.where(axis == 0, gx, wp.where(axis == 1, gy, gz))
            fvel[a * 3 + axis] += gc * dt


@wp.func
def _face_velocity_high(g: Grid, fvel: wp.array(dtype=float), addr: int,
                        o: wp.vec4i, axis: int) -> float:
    """Flux-weighted velocity leaving cell ``addr`` through its high face.

    If the high neighbour is finer, the single coarse face is backed by a 2x2
    patch of fine faces; the outgoing flux is their area-weighted mean.
    """
    la = o[3]
    ext = 1 << la
    dx = wp.where(axis == 0, 1, 0)
    dy = wp.where(axis == 1, 1, 0)
    dz = wp.where(axis == 2, 1, 0)
    fx = o[0] + dx * ext
    fy = o[1] + dy * ext
    fz = o[2] + dz * ext
    nb = locate(g, fx, fy, fz)
    if nb.addr < 0:
        return 0.0  # wall
    if nb.level >= la:
        return fvel[nb.addr * 3 + axis]
    sub = 1 << nb.level
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
            du = u * sub
            dv = v * sub
            nc = locate(g, fx + e1x * du + e2x * dv, fy + e1y * du + e2y * dv,
                        fz + e1z * du + e2z * dv)
            if nc.addr >= 0:
                acc += fvel[nc.addr * 3 + axis]
                cnt += 1.0
    return wp.where(cnt > 0.0, acc / cnt, 0.0)


@wp.kernel
def build_rhs(
    g: Grid,
    fvel: wp.array(dtype=float),
    cwsum: wp.array(dtype=float),
    rhs: wp.array(dtype=float),
    dt: float,
    rho_0: float,
    k_div: float,
):
    """``b = -(1/dt) * sum_faces u* . A_f``, matching the finite-volume operator.

    The operator of :mod:`pfflip.mr_operator` is ``sum_f beta A/d (p_A - p_B)``,
    i.e. an integrated flux, so the right-hand side must be the integrated
    divergence -- not the pointwise one.
    """
    a = wp.tid()
    o = cell_origin_finest(g, a)
    h = cell_size(g, o[3])
    area = h * h
    flux = float(0.0)
    for axis in range(3):
        u_lo = fvel[a * 3 + axis]
        u_hi = _face_velocity_high(g, fvel, a, o, axis)
        flux += (u_hi - u_lo) * area
    r = -flux / dt
    if k_div > 0.0:
        excess = (cwsum[a] / float(1 << (3 * o[3])) - rho_0) / rho_0
        r -= k_div * wp.max(excess, 0.0) * h * h * h / dt
    rhs[a] = r


@wp.kernel
def project(
    g: Grid,
    p: wp.array(dtype=float),
    fbeta: wp.array(dtype=float),
    fvel: wp.array(dtype=float),
    dt: float,
):
    """``u = u* - dt * beta * (p_hi - p_lo) / dist`` on each owned low face."""
    a = wp.tid()
    o = cell_origin_finest(g, a)
    la = o[3]
    hA = cell_size(g, la)
    pa = p[a]
    for axis in range(3):
        beta = fbeta[a * 3 + axis]
        if beta == 0.0:
            continue
        dx = wp.where(axis == 0, 1, 0)
        dy = wp.where(axis == 1, 1, 0)
        dz = wp.where(axis == 2, 1, 0)
        nb = locate(g, o[0] - dx, o[1] - dy, o[2] - dz)
        if nb.addr < 0:
            continue
        hB = cell_size(g, nb.level)
        dist = 0.5 * (hA + hB)
        fvel[a * 3 + axis] -= dt * beta * (pa - p[nb.addr]) / dist


@wp.kernel
def max_divergence(g: Grid, fvel: wp.array(dtype=float), out: wp.array(dtype=float)):
    a = wp.tid()
    o = cell_origin_finest(g, a)
    h = cell_size(g, o[3])
    d = float(0.0)
    for axis in range(3):
        d += _face_velocity_high(g, fvel, a, o, axis) - fvel[a * 3 + axis]
    wp.atomic_max(out, 0, wp.abs(d / h))


# ---------------------------------------------------------------------------
# G2P and advection
# ---------------------------------------------------------------------------
@wp.func
def sample_face(g: Grid, f: wp.array(dtype=float), x: wp.vec3, axis: int) -> float:
    """Trilinear interpolation of a face field, at the resolution of the cell
    containing ``x``.

    Samples are addressed in level-``l`` index space; each is resolved back
    through :func:`locate`, so a neighbour at a coarser level simply
    contributes its own (coarser) face value.  The result is continuous within
    a resolution level and merely consistent across jumps -- enough for FLIP,
    which only needs a velocity to advect with.
    """
    c = locate_at(g, x)
    if c.addr < 0:
        return 0.0
    l = c.level
    ext = 1 << l
    h = cell_size(g, l)

    ga = wp.vec3(x[0] / h, x[1] / h, x[2] / h)
    # the sampled quantity sits at index m along `axis` and m+0.5 on the others
    sx = ga[0] - wp.where(axis == 0, 0.0, 0.5)
    sy = ga[1] - wp.where(axis == 1, 0.0, 0.5)
    sz = ga[2] - wp.where(axis == 2, 0.0, 0.5)
    i0 = int(wp.floor(sx))
    j0 = int(wp.floor(sy))
    k0 = int(wp.floor(sz))
    tx = sx - float(i0)
    ty = sy - float(j0)
    tz = sz - float(k0)

    acc = float(0.0)
    wsum = float(0.0)
    for di in range(2):
        wi = wp.where(di == 0, 1.0 - tx, tx)
        for dj in range(2):
            wj = wp.where(dj == 0, 1.0 - ty, ty)
            for dk in range(2):
                wk = wp.where(dk == 0, 1.0 - tz, tz)
                w = wi * wj * wk
                if w <= 0.0:
                    continue
                nc = locate(g, (i0 + di) * ext, (j0 + dj) * ext, (k0 + dk) * ext)
                if nc.addr < 0:
                    continue
                acc += w * f[nc.addr * 3 + axis]
                wsum += w
    return wp.where(wsum > 0.0, acc / wsum, 0.0)


@wp.func
def sample_velocity(g: Grid, f: wp.array(dtype=float), x: wp.vec3) -> wp.vec3:
    return wp.vec3(sample_face(g, f, x, 0), sample_face(g, f, x, 1), sample_face(g, f, x, 2))


@wp.kernel
def g2p(
    g: Grid,
    px: wp.array(dtype=wp.vec3),
    pv: wp.array(dtype=wp.vec3),
    palpha: wp.array(dtype=float),
    fvel: wp.array(dtype=float),
    fdvel: wp.array(dtype=float),
):
    """``u_p = alpha (u_p + I(du)) + (1 - alpha) I(u)`` -- eq. (12).

    ``alpha`` is per particle, set by :func:`pfflip.particles.per_particle_alpha_flip`
    from the *local grid spacing* so that numerical viscosity is equalised
    across resolution levels (paper 5.3).
    """
    p = wp.tid()
    x = px[p]
    a = palpha[p]
    delta = sample_velocity(g, fdvel, x)
    pic = sample_velocity(g, fvel, x)
    pv[p] = a * (pv[p] + delta) + (1.0 - a) * pic


@wp.kernel
def field_difference(a: wp.array(dtype=float), b: wp.array(dtype=float), out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = a[i] - b[i]


@wp.kernel
def advect_rk3(
    g: Grid,
    px: wp.array(dtype=wp.vec3),
    fvel: wp.array(dtype=float),
    dt: float,
    lo: wp.vec3,
    hi: wp.vec3,
):
    """RK3 (Ralston) trace using MSBG velocity interpolation (paper 5.4 step 7)."""
    p = wp.tid()
    x = px[p]
    k1 = sample_velocity(g, fvel, x)
    k2 = sample_velocity(g, fvel, x + 0.5 * dt * k1)
    k3 = sample_velocity(g, fvel, x + 0.75 * dt * k2)
    x = x + dt * ((2.0 / 9.0) * k1 + (3.0 / 9.0) * k2 + (4.0 / 9.0) * k3)
    px[p] = wp.vec3(
        wp.clamp(x[0], lo[0], hi[0]),
        wp.clamp(x[1], lo[1], hi[1]),
        wp.clamp(x[2], lo[2], hi[2]),
    )


@wp.kernel
def reflect_at_walls(px: wp.array(dtype=wp.vec3), pv: wp.array(dtype=wp.vec3),
                     lo: wp.vec3, hi: wp.vec3):
    p = wp.tid()
    x = px[p]
    v = pv[p]
    xo = wp.vec3(0.0, 0.0, 0.0)
    vo = wp.vec3(0.0, 0.0, 0.0)
    for a in range(3):
        xa = x[a]
        va = v[a]
        if xa <= lo[a]:
            xa = lo[a]
            va = wp.max(va, 0.0)
        elif xa >= hi[a]:
            xa = hi[a]
            va = wp.min(va, 0.0)
        xo[a] = xa
        vo[a] = va
    px[p] = xo
    pv[p] = vo


@wp.kernel
def interface_flag_per_block(
    g: Grid,
    is_liquid: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.int32),
):
    """Mark blocks whose cells are not all one phase -- the interface band.

    Feeds the reduced-resolution auxiliary distance field of paper 5.2.
    """
    a = wp.tid()
    s = g.cslot[a]
    wp.atomic_max(out, g.blk[s] * 2 + 0, is_liquid[a])
    wp.atomic_max(out, g.blk[s] * 2 + 1, 1 - is_liquid[a])


@wp.kernel
def particle_max_speed(pv: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    p = wp.tid()
    wp.atomic_max(out, 0, wp.length(pv[p]))
