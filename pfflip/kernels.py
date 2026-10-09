"""Warp kernels for the PF-FLIP core algorithm (paper section 3).

Grid layout
-----------
Uniform staggered MAC grid.  Cell (i,j,k) has its centre at
``(i + .5, j + .5, k + .5) * dx``.  Face-normal arrays are stored separately per
axis; the face with index ``idx`` on axis ``a`` sits at ``(idx + o_a) * dx``
where ``o_a = (.5,.5,.5) - .5 * e_a``.  Shapes are therefore
``(nx+1, ny, nz)``, ``(nx, ny+1, nz)`` and ``(nx, ny, nz+1)``.

Particle types: ``0 = air``, ``1 = liquid``.
"""

from __future__ import annotations

import warp as wp

AIR = wp.constant(0)
LIQUID = wp.constant(1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
@wp.func
def face_offset(axis: int) -> wp.vec3:
    """``o_a`` such that face centre = ``(index + o_a) * dx``."""
    return wp.vec3(
        wp.where(axis == 0, 0.0, 0.5),
        wp.where(axis == 1, 0.0, 0.5),
        wp.where(axis == 2, 0.0, 0.5),
    )


@wp.func
def vcomp(v: wp.vec3, axis: int) -> float:
    return wp.where(axis == 0, v[0], wp.where(axis == 1, v[1], v[2]))


@wp.func
def splat_weight(d2: float, r2: float) -> float:
    """Paper eq. (6): ``w = (max(1 - (d/r)^2, 0))^3``.

    Cubic in the *squared* distance, so no square root is needed and the
    support is a sphere of radius ``r``.
    """
    t = 1.0 - d2 / r2
    t = wp.max(t, 0.0)
    return t * t * t


@wp.func
def sample_face(f: wp.array3d(dtype=float), x: wp.vec3, dx: float, axis: int) -> float:
    """Trilinear interpolation of a face-sampled field at world position ``x``."""
    o = face_offset(axis)
    nx = f.shape[0]
    ny = f.shape[1]
    nz = f.shape[2]

    g = wp.vec3(x[0] / dx - o[0], x[1] / dx - o[1], x[2] / dx - o[2])
    i0 = int(wp.floor(g[0]))
    j0 = int(wp.floor(g[1]))
    k0 = int(wp.floor(g[2]))
    tx = g[0] - float(i0)
    ty = g[1] - float(j0)
    tz = g[2] - float(k0)

    acc = float(0.0)
    for di in range(2):
        wi = wp.where(di == 0, 1.0 - tx, tx)
        ii = wp.clamp(i0 + di, 0, nx - 1)
        for dj in range(2):
            wj = wp.where(dj == 0, 1.0 - ty, ty)
            jj = wp.clamp(j0 + dj, 0, ny - 1)
            for dk in range(2):
                wk = wp.where(dk == 0, 1.0 - tz, tz)
                kk = wp.clamp(k0 + dk, 0, nz - 1)
                acc += wi * wj * wk * f[ii, jj, kk]
    return acc


@wp.func
def sample_velocity(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    x: wp.vec3,
    dx: float,
) -> wp.vec3:
    return wp.vec3(
        sample_face(u, x, dx, 0),
        sample_face(v, x, dx, 1),
        sample_face(w, x, dx, 2),
    )


# ---------------------------------------------------------------------------
# particle -> grid  (paper 3.2, eqs. 3-6)
# ---------------------------------------------------------------------------
@wp.kernel
def p2g_face(
    px: wp.array(dtype=wp.vec3),
    pv: wp.array(dtype=wp.vec3),
    ptype: wp.array(dtype=wp.int32),
    pmass: wp.array(dtype=float),
    prad: wp.array(dtype=float),
    pstate: wp.array(dtype=wp.int32),
    mass: wp.array3d(dtype=float),
    mom: wp.array3d(dtype=float),
    axis: int,
    dx: float,
):
    """Splat mass (eq. 3) and momentum (eq. 4) onto the faces of one axis.

    The accumulated ``mass`` doubles as the raw mass density ``rho~_ai`` used
    for the phase field (paper 3.3) -- that reuse is the reason PF-FLIP gets
    its interface representation "for free".
    """
    p = wp.tid()
    # droplets are purely Lagrangian and bubbles are gone: neither contributes
    # mass or momentum to the grid (paper 3.4, 7)
    if pstate[p] != 0:
        return
    xp = px[p]
    mp = pmass[p]
    rp = prad[p]
    up = vcomp(pv[p], axis)

    o = face_offset(axis)
    r2 = rp * rp
    R = rp / dx

    nx = mass.shape[0]
    ny = mass.shape[1]
    nz = mass.shape[2]

    # integer face indices whose centre can lie inside the support sphere
    cx = xp[0] / dx - o[0]
    cy = xp[1] / dx - o[1]
    cz = xp[2] / dx - o[2]
    ilo = wp.max(int(wp.floor(cx - R)) + 1, 0)
    ihi = wp.min(int(wp.floor(cx + R)), nx - 1)
    jlo = wp.max(int(wp.floor(cy - R)) + 1, 0)
    jhi = wp.min(int(wp.floor(cy + R)), ny - 1)
    klo = wp.max(int(wp.floor(cz - R)) + 1, 0)
    khi = wp.min(int(wp.floor(cz + R)), nz - 1)

    for i in range(ilo, ihi + 1):
        fx = (float(i) + o[0]) * dx - xp[0]
        for j in range(jlo, jhi + 1):
            fy = (float(j) + o[1]) * dx - xp[1]
            for k in range(klo, khi + 1):
                fz = (float(k) + o[2]) * dx - xp[2]
                d2 = fx * fx + fy * fy + fz * fz
                wgt = splat_weight(d2, r2)
                if wgt > 0.0:
                    wp.atomic_add(mass, i, j, k, wgt * mp)
                    wp.atomic_add(mom, i, j, k, wgt * mp * up)


@wp.kernel
def p2g_cell(
    px: wp.array(dtype=wp.vec3),
    pmass: wp.array(dtype=float),
    prad: wp.array(dtype=float),
    pstate: wp.array(dtype=wp.int32),
    cell_mass: wp.array3d(dtype=float),
    cell_wsum: wp.array3d(dtype=float),
    dx: float,
):
    """Cell-centred raw mass density and bare kernel-weight sum.

    ``cell_mass`` feeds eq. (7) for the per-*cell* air/liquid tag, ``cell_wsum``
    is the particle-density estimate used by the ``c_div`` volume correction.
    """
    p = wp.tid()
    if pstate[p] != 0:
        return
    xp = px[p]
    mp = pmass[p]
    rp = prad[p]
    r2 = rp * rp
    R = rp / dx

    nx = cell_mass.shape[0]
    ny = cell_mass.shape[1]
    nz = cell_mass.shape[2]

    cx = xp[0] / dx - 0.5
    cy = xp[1] / dx - 0.5
    cz = xp[2] / dx - 0.5
    ilo = wp.max(int(wp.floor(cx - R)) + 1, 0)
    ihi = wp.min(int(wp.floor(cx + R)), nx - 1)
    jlo = wp.max(int(wp.floor(cy - R)) + 1, 0)
    jhi = wp.min(int(wp.floor(cy + R)), ny - 1)
    klo = wp.max(int(wp.floor(cz - R)) + 1, 0)
    khi = wp.min(int(wp.floor(cz + R)), nz - 1)

    for i in range(ilo, ihi + 1):
        fx = (float(i) + 0.5) * dx - xp[0]
        for j in range(jlo, jhi + 1):
            fy = (float(j) + 0.5) * dx - xp[1]
            for k in range(klo, khi + 1):
                fz = (float(k) + 0.5) * dx - xp[2]
                d2 = fx * fx + fy * fy + fz * fz
                wgt = splat_weight(d2, r2)
                if wgt > 0.0:
                    wp.atomic_add(cell_mass, i, j, k, wgt * mp)
                    wp.atomic_add(cell_wsum, i, j, k, wgt)


@wp.kernel
def face_velocity_from_momentum(
    mass: wp.array3d(dtype=float),
    mom: wp.array3d(dtype=float),
    vel: wp.array3d(dtype=float),
    valid: wp.array3d(dtype=wp.int32),
    mass_eps: float,
):
    """``u~*_ai = P_ai / m_ai`` (paper 3.2)."""
    i, j, k = wp.tid()
    m = mass[i, j, k]
    if m > mass_eps:
        vel[i, j, k] = mom[i, j, k] / m
        valid[i, j, k] = 1
    else:
        vel[i, j, k] = 0.0
        valid[i, j, k] = 0


# ---------------------------------------------------------------------------
# phase field  (paper 3.3, eq. 7)
# ---------------------------------------------------------------------------
@wp.func
def phase_field(rho_raw: float, rho_min: float, alpha_phi: float, rho_0l: float) -> float:
    """Eq. (7).  ``rho_0l`` is the raw density of a perfect liquid lattice."""
    if rho_raw < rho_min:
        return 0.0
    return wp.min(wp.sqrt(wp.max(rho_raw - rho_min, 0.0) / (alpha_phi * rho_0l)), 1.0)


@wp.kernel
def cell_phase(
    cell_mass: wp.array3d(dtype=float),
    phi_cell: wp.array3d(dtype=float),
    is_liquid: wp.array3d(dtype=wp.int32),
    rho_min: float,
    alpha_phi: float,
    rho_0l: float,
):
    i, j, k = wp.tid()
    f = phase_field(cell_mass[i, j, k], rho_min, alpha_phi, rho_0l)
    phi_cell[i, j, k] = f
    # "We tag cells with phase field values below 0.5 as air" (paper 3.3)
    is_liquid[i, j, k] = wp.where(f >= 0.5, 1, 0)


@wp.kernel
def face_beta(
    mass: wp.array3d(dtype=float),
    is_liquid: wp.array3d(dtype=wp.int32),
    phi_face: wp.array3d(dtype=float),
    beta: wp.array3d(dtype=float),
    axis: int,
    rho_min: float,
    alpha_phi: float,
    rho_0l: float,
    rho_l: float,
    rho_g: float,
):
    """Face phase field, density and Poisson coefficient ``beta = 1/rho``.

    Paper 3.3: ``phi = 1`` for liquid-liquid faces and ``phi = 0`` for air-air
    faces, which removes the residual phase-field noise away from the
    interface; only interface-cutting faces use eq. (7) directly.
    Paper eq. (9): ``beta(x) = 1 / rho(x)``.
    """
    i, j, k = wp.tid()
    nc0 = is_liquid.shape[0]
    nc1 = is_liquid.shape[1]
    nc2 = is_liquid.shape[2]

    # the two cells adjacent to this face along `axis`
    im = i - wp.where(axis == 0, 1, 0)
    jm = j - wp.where(axis == 1, 1, 0)
    km = k - wp.where(axis == 2, 1, 0)

    lo_outside = im < 0 or jm < 0 or km < 0
    hi_outside = i >= nc0 or j >= nc1 or k >= nc2

    a = int(0)
    b = int(0)
    if not lo_outside:
        a = is_liquid[im, jm, km]
    if not hi_outside:
        b = is_liquid[i, j, k]
    if lo_outside:
        a = b
    if hi_outside:
        b = a

    if a == 1 and b == 1:
        f = 1.0
    elif a == 0 and b == 0:
        f = 0.0
    else:
        f = phase_field(mass[i, j, k], rho_min, alpha_phi, rho_0l)

    phi_face[i, j, k] = f
    rho = (1.0 - f) * rho_g + f * rho_l
    beta[i, j, k] = 1.0 / rho


@wp.kernel
def zero_solid_faces(beta: wp.array3d(dtype=float), vel: wp.array3d(dtype=float), axis: int):
    """No-flux domain walls: ``beta = 0`` and ``u = 0`` on boundary faces.

    A zero face coefficient removes the neighbour term from the 7-point
    stencil, which is exactly the homogeneous Neumann condition for pressure.
    """
    i, j, k = wp.tid()
    n = wp.where(axis == 0, i, wp.where(axis == 1, j, k))
    last = wp.where(axis == 0, beta.shape[0], wp.where(axis == 1, beta.shape[1], beta.shape[2])) - 1
    if n == 0 or n == last:
        beta[i, j, k] = 0.0
        vel[i, j, k] = 0.0


# ---------------------------------------------------------------------------
# forces, divergence, projection  (paper 3.5)
# ---------------------------------------------------------------------------
@wp.kernel
def add_gravity(vel: wp.array3d(dtype=float), g: float, dt: float):
    i, j, k = wp.tid()
    vel[i, j, k] += g * dt


@wp.kernel
def build_rhs(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    cell_wsum: wp.array3d(dtype=float),
    b: wp.array3d(dtype=float),
    dx: float,
    dt: float,
    rho_0: float,
    k_div: float,
):
    """``b_i = -(dx/dt) * sum_a (u*_{a,i+1} - u*_{a,i})  -  dx^2 * c_div``.

    This is eq. (8)/(10) scaled by ``dx^2`` so that the operator reads
    ``A p = sum_f beta_f (p_i - p_nbr)``, which is symmetric positive
    semi-definite.  ``c_div`` is the volume correction that keeps the particle
    density near its target ``rho~_0`` (paper eq. 8, Losasso et al. 2008).
    """
    i, j, k = wp.tid()
    du = (u[i + 1, j, k] - u[i, j, k]) + (v[i, j + 1, k] - v[i, j, k]) + (w[i, j, k + 1] - w[i, j, k])
    rhs = -(dx / dt) * du
    if k_div > 0.0:
        excess = (cell_wsum[i, j, k] - rho_0) / rho_0
        rhs -= dx * dx * k_div * wp.max(excess, 0.0) / dt
    b[i, j, k] = rhs


@wp.kernel
def apply_pressure_gradient(
    p: wp.array3d(dtype=float),
    beta: wp.array3d(dtype=float),
    vel: wp.array3d(dtype=float),
    axis: int,
    dx: float,
    dt: float,
):
    """``u = u* - dt * (1/rho) * grad p`` (paper 3.5)."""
    i, j, k = wp.tid()
    bcoef = beta[i, j, k]
    if bcoef == 0.0:
        return
    im = i - wp.where(axis == 0, 1, 0)
    jm = j - wp.where(axis == 1, 1, 0)
    km = k - wp.where(axis == 2, 1, 0)
    if im < 0 or jm < 0 or km < 0:
        return
    if i >= p.shape[0] or j >= p.shape[1] or k >= p.shape[2]:
        return
    vel[i, j, k] -= dt * bcoef * (p[i, j, k] - p[im, jm, km]) / dx


@wp.kernel
def divergence_norm(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    out: wp.array(dtype=float),
    dx: float,
):
    i, j, k = wp.tid()
    d = (u[i + 1, j, k] - u[i, j, k]) + (v[i, j + 1, k] - v[i, j, k]) + (w[i, j, k + 1] - w[i, j, k])
    wp.atomic_max(out, 0, wp.abs(d / dx))


# ---------------------------------------------------------------------------
# velocity extrapolation into the air phase  (paper 3.7)
# ---------------------------------------------------------------------------
@wp.kernel
def seed_liquid_velocity(
    vel: wp.array3d(dtype=float),
    is_liquid: wp.array3d(dtype=wp.int32),
    vel_l: wp.array3d(dtype=float),
    known: wp.array3d(dtype=wp.int32),
    axis: int,
):
    """Copy velocities on faces touching a liquid cell; mark the rest unknown."""
    i, j, k = wp.tid()
    im = i - wp.where(axis == 0, 1, 0)
    jm = j - wp.where(axis == 1, 1, 0)
    km = k - wp.where(axis == 2, 1, 0)
    liq = int(0)
    if i < is_liquid.shape[0] and j < is_liquid.shape[1] and k < is_liquid.shape[2]:
        liq = wp.max(liq, is_liquid[i, j, k])
    if im >= 0 and jm >= 0 and km >= 0:
        liq = wp.max(liq, is_liquid[im, jm, km])
    known[i, j, k] = liq
    vel_l[i, j, k] = wp.where(liq == 1, vel[i, j, k], 0.0)


@wp.kernel
def extrapolate_sweep(
    vel_l: wp.array3d(dtype=float),
    known: wp.array3d(dtype=wp.int32),
    vel_out: wp.array3d(dtype=float),
    known_out: wp.array3d(dtype=wp.int32),
):
    """One Jacobi sweep of nearest-neighbour extrapolation."""
    i, j, k = wp.tid()
    if known[i, j, k] == 1:
        vel_out[i, j, k] = vel_l[i, j, k]
        known_out[i, j, k] = 1
        return
    nx = vel_l.shape[0]
    ny = vel_l.shape[1]
    nz = vel_l.shape[2]
    acc = float(0.0)
    cnt = int(0)
    for d in range(6):
        ax = d // 2
        s = wp.where(d % 2 == 0, -1, 1)
        ii = i + wp.where(ax == 0, s, 0)
        jj = j + wp.where(ax == 1, s, 0)
        kk = k + wp.where(ax == 2, s, 0)
        if ii < 0 or jj < 0 or kk < 0 or ii >= nx or jj >= ny or kk >= nz:
            continue
        if known[ii, jj, kk] == 1:
            acc += vel_l[ii, jj, kk]
            cnt += 1
    if cnt > 0:
        vel_out[i, j, k] = acc / float(cnt)
        known_out[i, j, k] = 1
    else:
        vel_out[i, j, k] = vel_l[i, j, k]
        known_out[i, j, k] = 0


# ---------------------------------------------------------------------------
# grid -> particle  (paper 3.6, eqs. 11-12)
# ---------------------------------------------------------------------------
@wp.kernel
def g2p(
    px: wp.array(dtype=wp.vec3),
    pv: wp.array(dtype=wp.vec3),
    ptype: wp.array(dtype=wp.int32),
    pstate: wp.array(dtype=wp.int32),
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    du: wp.array3d(dtype=float),
    dv: wp.array3d(dtype=float),
    dw: wp.array3d(dtype=float),
    dx: float,
    alpha_liquid: float,
    alpha_air: float,
):
    """``u_p = alpha (u_p + I(du)) + (1 - alpha) I(u)`` -- eq. (12).

    ``alpha_FLIP`` is chosen per phase; via eq. (13) this is what sets the
    phase-dependent numerical viscosity (paper 3.7).
    """
    p = wp.tid()
    if pstate[p] != 0:
        return
    xp = px[p]
    alpha = wp.where(ptype[p] == LIQUID, alpha_liquid, alpha_air)
    delta = sample_velocity(du, dv, dw, xp, dx)
    pic = sample_velocity(u, v, w, xp, dx)
    pv[p] = alpha * (pv[p] + delta) + (1.0 - alpha) * pic


@wp.kernel
def field_difference(a: wp.array3d(dtype=float), b: wp.array3d(dtype=float), out: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    out[i, j, k] = a[i, j, k] - b[i, j, k]


# ---------------------------------------------------------------------------
# advection  (paper 3.7: third-order Runge-Kutta)
# ---------------------------------------------------------------------------
@wp.kernel
def advect_rk3(
    px: wp.array(dtype=wp.vec3),
    ptype: wp.array(dtype=wp.int32),
    pstate: wp.array(dtype=wp.int32),
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    ul: wp.array3d(dtype=float),
    vl: wp.array3d(dtype=float),
    wl: wp.array3d(dtype=float),
    dt: float,
    dx: float,
    lo: wp.vec3,
    hi: wp.vec3,
):
    """RK3 (Ralston) trace of particle positions.

    Liquid particles are advected with the velocity field extrapolated a few
    cells into the air phase (paper 3.7), air particles with the raw field.
    """
    p = wp.tid()
    if pstate[p] != 0:
        return  # droplets are integrated by pfflip.droplets instead
    x = px[p]
    liquid = ptype[p] == LIQUID

    if liquid:
        k1 = sample_velocity(ul, vl, wl, x, dx)
        k2 = sample_velocity(ul, vl, wl, x + 0.5 * dt * k1, dx)
        k3 = sample_velocity(ul, vl, wl, x + 0.75 * dt * k2, dx)
    else:
        k1 = sample_velocity(u, v, w, x, dx)
        k2 = sample_velocity(u, v, w, x + 0.5 * dt * k1, dx)
        k3 = sample_velocity(u, v, w, x + 0.75 * dt * k2, dx)

    x = x + dt * ((2.0 / 9.0) * k1 + (3.0 / 9.0) * k2 + (4.0 / 9.0) * k3)
    px[p] = wp.vec3(
        wp.clamp(x[0], lo[0], hi[0]),
        wp.clamp(x[1], lo[1], hi[1]),
        wp.clamp(x[2], lo[2], hi[2]),
    )


@wp.kernel
def particle_max_speed(pv: wp.array(dtype=wp.vec3), pstate: wp.array(dtype=wp.int32),
                       out: wp.array(dtype=float)):
    """CFL speed over the *FLIP* particles only.

    Droplets carry their own locally adaptive sub-stepping (paper 7), which is
    exactly so that a handful of fast ballistic drops cannot throttle the grid
    time step.  Counting them collapsed dt by more than 10x in the surge scene.
    """
    p = wp.tid()
    if pstate[p] != 0:
        return
    wp.atomic_max(out, 0, wp.length(pv[p]))


@wp.kernel
def reflect_at_walls(
    px: wp.array(dtype=wp.vec3),
    pv: wp.array(dtype=wp.vec3),
    pstate: wp.array(dtype=wp.int32),
    lo: wp.vec3,
    hi: wp.vec3,
):
    """Clamp particles into the box and kill the inward-pointing component."""
    p = wp.tid()
    if pstate[p] == 2:
        return
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


# ---------------------------------------------------------------------------
# solid obstacles
# ---------------------------------------------------------------------------
@wp.kernel
def zero_solid_obstacle_faces(
    solid: wp.array3d(dtype=wp.int32),
    beta: wp.array3d(dtype=float),
    vel: wp.array3d(dtype=float),
    axis: int,
):
    """No-flux on any face touching a solid cell.

    Zeroing the face coefficient removes that neighbour from the 7-point
    stencil, which is exactly the homogeneous Neumann condition the pressure
    solve needs at a solid boundary (paper 4.2 follows Losasso et al. 2004 /
    Weber et al. 2015 for obstacle handling).
    """
    i, j, k = wp.tid()
    nx, ny, nz = solid.shape[0], solid.shape[1], solid.shape[2]
    im = i - wp.where(axis == 0, 1, 0)
    jm = j - wp.where(axis == 1, 1, 0)
    km = k - wp.where(axis == 2, 1, 0)
    blocked = int(0)
    if i < nx and j < ny and k < nz:
        blocked = wp.max(blocked, solid[i, j, k])
    if im >= 0 and jm >= 0 and km >= 0:
        blocked = wp.max(blocked, solid[im, jm, km])
    if blocked == 1:
        beta[i, j, k] = 0.0
        vel[i, j, k] = 0.0


@wp.kernel
def push_out_of_solids(
    px: wp.array(dtype=wp.vec3),
    pv: wp.array(dtype=wp.vec3),
    pstate: wp.array(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    dx: float,
):
    """Eject particles that ended up inside an obstacle, killing the inward
    velocity component along the shallowest escape direction."""
    p = wp.tid()
    if pstate[p] == 2:
        return
    x = px[p]
    nx, ny, nz = solid.shape[0], solid.shape[1], solid.shape[2]
    i = wp.clamp(int(x[0] / dx), 0, nx - 1)
    j = wp.clamp(int(x[1] / dx), 0, ny - 1)
    k = wp.clamp(int(x[2] / dx), 0, nz - 1)
    if solid[i, j, k] == 0:
        return
    # search the 6 axial directions for the nearest free cell
    best = float(1.0e9)
    bax = int(0)
    bsg = int(1)
    for d in range(6):
        ax = d // 2
        sg = wp.where(d % 2 == 0, -1, 1)
        for t in range(1, 8):
            ii = i + wp.where(ax == 0, sg * t, 0)
            jj = j + wp.where(ax == 1, sg * t, 0)
            kk = k + wp.where(ax == 2, sg * t, 0)
            if ii < 0 or jj < 0 or kk < 0 or ii >= nx or jj >= ny or kk >= nz:
                break
            if solid[ii, jj, kk] == 0:
                dist = float(t)
                if dist < best:
                    best = dist
                    bax = ax
                    bsg = sg
                break
    if best > 1.0e8:
        return
    off = (best + 0.5) * dx * float(bsg)
    xo = wp.vec3(x[0], x[1], x[2])
    vo = wp.vec3(pv[p][0], pv[p][1], pv[p][2])
    xo[bax] = x[bax] + off
    if float(bsg) * vo[bax] < 0.0:
        vo[bax] = 0.0
    px[p] = xo
    pv[p] = vo
