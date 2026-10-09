"""Escaped particles, Lagrangian droplets and spray (paper 3.4 and 7).

PF-FLIP does **not** use the "particle bounce-back" heuristic of earlier
two-phase FLIP work: its velocity field resolves the no-slip condition between
phases, so a particle found on the wrong side of the interface is not an error
to be corrected but a physically meaningful **subgrid-scale droplet or bubble**
(paper 3.4).

Classification, with ``delta_phi = 0.2``:

* a **liquid** particle where ``phi(x) < 1/2 - delta_phi`` becomes a *droplet*;
* an **air** particle where ``phi(x) > 1/2 + delta_phi`` is a *bubble* and is
  **deleted** -- the paper does the same, having found their visual
  contribution negligible at large scale;
* a droplet that finds itself back inside the liquid is **re-converted to a
  FLIP particle**.

Droplets are then purely Lagrangian point masses (paper 7) under gravity and
air drag, with radii drawn from a log-normal distribution on emission and
evolved by the evaporation ODE ``dr/dt ~ -1/r``, integrated with RK2.

Implementation note: droplets share the FLIP particle arrays and are
distinguished by a per-particle ``state`` channel, so conversion in either
direction is a flag flip and no stream compaction is ever needed.
"""

from __future__ import annotations

import numpy as np
import warp as wp

STATE_FLIP = wp.constant(0)
STATE_DROPLET = wp.constant(1)
STATE_DEAD = wp.constant(2)

AIR = wp.constant(0)
LIQUID = wp.constant(1)


# ---------------------------------------------------------------------------
@wp.func
def sample_cell(f: wp.array3d(dtype=float), x: wp.vec3, dx: float) -> float:
    """Trilinear interpolation of a cell-centred field."""
    nx, ny, nz = f.shape[0], f.shape[1], f.shape[2]
    g = wp.vec3(x[0] / dx - 0.5, x[1] / dx - 0.5, x[2] / dx - 0.5)
    i0 = int(wp.floor(g[0]))
    j0 = int(wp.floor(g[1]))
    k0 = int(wp.floor(g[2]))
    tx, ty, tz = g[0] - float(i0), g[1] - float(j0), g[2] - float(k0)
    acc = float(0.0)
    for a in range(2):
        wa = wp.where(a == 0, 1.0 - tx, tx)
        ii = wp.clamp(i0 + a, 0, nx - 1)
        for b in range(2):
            wb = wp.where(b == 0, 1.0 - ty, ty)
            jj = wp.clamp(j0 + b, 0, ny - 1)
            for c in range(2):
                wc = wp.where(c == 0, 1.0 - tz, tz)
                kk = wp.clamp(k0 + c, 0, nz - 1)
                acc += wa * wb * wc * f[ii, jj, kk]
    return acc


# ---------------------------------------------------------------------------
# classification (paper 3.4)
# ---------------------------------------------------------------------------
@wp.kernel
def classify_escaped(
    px: wp.array(dtype=wp.vec3),
    ptype: wp.array(dtype=wp.int32),
    pstate: wp.array(dtype=wp.int32),
    pdrop_r: wp.array(dtype=float),
    phi: wp.array3d(dtype=float),
    dx: float,
    delta_phi: float,
    r_mu: float,
    r_sigma: float,
    r_min: float,
    r_max: float,
    seed: int,
    counters: wp.array(dtype=wp.int32),
):
    """Convert escaped particles, delete bubbles, re-absorb returning droplets."""
    p = wp.tid()
    st = pstate[p]
    if st == STATE_DEAD:
        return
    f = sample_cell(phi, px[p], dx)

    if st == STATE_FLIP:
        if ptype[p] == LIQUID and f < 0.5 - delta_phi:
            pstate[p] = STATE_DROPLET
            # log-normal droplet radii on emission (Babinsky & Sojka 2002)
            rng = wp.rand_init(seed, p)
            r = wp.exp(r_mu + r_sigma * wp.randn(rng))
            pdrop_r[p] = wp.clamp(r, r_min, r_max)
            wp.atomic_add(counters, 0, 1)
        elif ptype[p] == AIR and f > 0.5 + delta_phi:
            # bubble: deleted, as in the paper's implementation
            pstate[p] = STATE_DEAD
            wp.atomic_add(counters, 1, 1)
    elif st == STATE_DROPLET:
        if f > 0.5 + delta_phi:
            # merged back into the liquid body -> back to FLIP
            pstate[p] = STATE_FLIP
            wp.atomic_add(counters, 2, 1)


# ---------------------------------------------------------------------------
# spray emission in steep velocity gradients (paper 7, Ihmsen et al. 2012)
# ---------------------------------------------------------------------------
@wp.kernel
def emit_spray_in_shear(
    px: wp.array(dtype=wp.vec3),
    ptype: wp.array(dtype=wp.int32),
    pstate: wp.array(dtype=wp.int32),
    pdrop_r: wp.array(dtype=float),
    phi: wp.array3d(dtype=float),
    shear: wp.array3d(dtype=float),
    dx: float,
    threshold: float,
    rate: float,
    r_mu: float,
    r_sigma: float,
    r_min: float,
    r_max: float,
    seed: int,
    counters: wp.array(dtype=wp.int32),
):
    """Emit extra droplets from liquid near the surface in high-shear regions.

    Paper 7 adds spray where the velocity gradient is steep; this is where
    aerodynamic break-up tears liquid into droplets, and the escaped-particle
    criterion alone under-produces it.
    """
    p = wp.tid()
    if pstate[p] != STATE_FLIP or ptype[p] != LIQUID:
        return
    x = px[p]
    f = sample_cell(phi, x, dx)
    if f > 0.9 or f < 0.1:
        return  # only the interfacial band breaks up
    if sample_cell(shear, x, dx) < threshold:
        return
    rng = wp.rand_init(seed + 104729, p)
    if wp.randf(rng) > rate:
        return
    pstate[p] = STATE_DROPLET
    pdrop_r[p] = wp.clamp(wp.exp(r_mu + r_sigma * wp.randn(rng)), r_min, r_max)
    wp.atomic_add(counters, 3, 1)


@wp.kernel
def velocity_gradient_magnitude(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    out: wp.array3d(dtype=float),
    dx: float,
):
    """Frobenius norm of grad(u) at cell centres, from the MAC faces."""
    i, j, k = wp.tid()
    nx, ny, nz = out.shape[0], out.shape[1], out.shape[2]
    acc = float(0.0)
    # diagonal terms straight off the staggered faces
    dudx = (u[i + 1, j, k] - u[i, j, k]) / dx
    dvdy = (v[i, j + 1, k] - v[i, j, k]) / dx
    dwdz = (w[i, j, k + 1] - w[i, j, k]) / dx
    acc = dudx * dudx + dvdy * dvdy + dwdz * dwdz
    # off-diagonals via centred differences of face averages
    jm = wp.max(j - 1, 0)
    jp = wp.min(j + 1, ny - 1)
    km = wp.max(k - 1, 0)
    kp = wp.min(k + 1, nz - 1)
    im = wp.max(i - 1, 0)
    ip = wp.min(i + 1, nx - 1)
    dudy = (
        0.5 * (u[i, jp, k] + u[i + 1, jp, k]) - 0.5 * (u[i, jm, k] + u[i + 1, jm, k])
    ) / (2.0 * dx)
    dudz = (
        0.5 * (u[i, j, kp] + u[i + 1, j, kp]) - 0.5 * (u[i, j, km] + u[i + 1, j, km])
    ) / (2.0 * dx)
    dvdx = (
        0.5 * (v[ip, j, k] + v[ip, j + 1, k]) - 0.5 * (v[im, j, k] + v[im, j + 1, k])
    ) / (2.0 * dx)
    dwdx = (
        0.5 * (w[ip, j, k] + w[ip, j, k + 1]) - 0.5 * (w[im, j, k] + w[im, j, k + 1])
    ) / (2.0 * dx)
    acc += dudy * dudy + dudz * dudz + dvdx * dvdx + dwdx * dwdx
    out[i, j, k] = wp.sqrt(acc)


# ---------------------------------------------------------------------------
# droplet dynamics (paper 7)
# ---------------------------------------------------------------------------
@wp.func
def drag_acceleration(vrel: wp.vec3, r: float, rho_g: float, rho_l: float, cd: float) -> wp.vec3:
    """Sphere drag per unit mass.

    ``F = 1/2 rho_g C_d pi r^2 |v_rel| v_rel`` over ``m = rho_l 4/3 pi r^3``
    gives ``a = 3 rho_g C_d |v_rel| v_rel / (8 rho_l r)`` -- so small droplets
    couple to the air far more strongly than large ones, which is what makes
    fine spray follow the turbulent air while large drops stay ballistic.
    """
    s = wp.length(vrel)
    if s <= 0.0 or r <= 0.0:
        return wp.vec3(0.0, 0.0, 0.0)
    return (3.0 * rho_g * cd * s / (8.0 * rho_l * r)) * vrel


@wp.kernel
def advance_droplets(
    px: wp.array(dtype=wp.vec3),
    pv: wp.array(dtype=wp.vec3),
    pstate: wp.array(dtype=wp.int32),
    pdrop_r: wp.array(dtype=float),
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    dt: float,
    dx: float,
    gravity: wp.vec3,
    rho_g: float,
    rho_l: float,
    cd: float,
    evap_k: float,
    r_min: float,
    lo: wp.vec3,
    hi: wp.vec3,
    max_substeps: int,
):
    """Integrate droplets under gravity + air drag, with locally adaptive steps.

    Sub-stepping is per droplet (paper 7, "locally adaptive time stepping"):
    fast droplets in strong drag need small steps, slow ones do not, and the
    cost is paid only where it is needed.
    """
    p = wp.tid()
    if pstate[p] != STATE_DROPLET:
        return
    x = px[p]
    vel = pv[p]
    r = pdrop_r[p]

    # choose the sub-step count from the CFL of this droplet alone
    speed = wp.length(vel) + 1e-8
    nsub = wp.clamp(int(speed * dt / (0.5 * dx)) + 1, 1, max_substeps)
    h = dt / float(nsub)

    for _ in range(nsub):
        ua = wp.vec3(
            sample_face_u(u, x, dx, 0),
            sample_face_u(v, x, dx, 1),
            sample_face_u(w, x, dx, 2),
        )
        a = gravity + drag_acceleration(ua - vel, r, rho_g, rho_l, cd)
        vel = vel + h * a
        x = x + h * vel

        # evaporation dr/dt = -k/r, RK2 (Sirignano 2010)
        if evap_k > 0.0 and r > 0.0:
            k1 = -evap_k / r
            rm = wp.max(r + 0.5 * h * k1, 1e-9)
            r = r + h * (-evap_k / rm)
            if r <= r_min:
                pstate[p] = STATE_DEAD
                return

    # A droplet that leaves the domain is gone, not stuck to the wall.
    # Clamping instead piles spray up against the boundary and renders as a
    # hard rectangular slab of mist at the ceiling.
    for a in range(3):
        if x[a] <= lo[a] or x[a] >= hi[a]:
            pstate[p] = STATE_DEAD
            return
    pdrop_r[p] = r
    pv[p] = vel
    px[p] = x


@wp.func
def sample_face_u(f: wp.array3d(dtype=float), x: wp.vec3, dx: float, axis: int) -> float:
    nx, ny, nz = f.shape[0], f.shape[1], f.shape[2]
    ox = wp.where(axis == 0, 0.0, 0.5)
    oy = wp.where(axis == 1, 0.0, 0.5)
    oz = wp.where(axis == 2, 0.0, 0.5)
    g = wp.vec3(x[0] / dx - ox, x[1] / dx - oy, x[2] / dx - oz)
    i0 = int(wp.floor(g[0]))
    j0 = int(wp.floor(g[1]))
    k0 = int(wp.floor(g[2]))
    tx, ty, tz = g[0] - float(i0), g[1] - float(j0), g[2] - float(k0)
    acc = float(0.0)
    for a in range(2):
        wa = wp.where(a == 0, 1.0 - tx, tx)
        ii = wp.clamp(i0 + a, 0, nx - 1)
        for b in range(2):
            wb = wp.where(b == 0, 1.0 - ty, ty)
            jj = wp.clamp(j0 + b, 0, ny - 1)
            for c in range(2):
                wc = wp.where(c == 0, 1.0 - tz, tz)
                kk = wp.clamp(k0 + c, 0, nz - 1)
                acc += wa * wb * wc * f[ii, jj, kk]
    return acc


# ---------------------------------------------------------------------------
# spray density field (paper 7)
# ---------------------------------------------------------------------------
@wp.kernel
def rasterize_spray(
    px: wp.array(dtype=wp.vec3),
    pstate: wp.array(dtype=wp.int32),
    pdrop_r: wp.array(dtype=float),
    density: wp.array3d(dtype=wp.float16),
    dx_spray: float,
):
    """Splat droplet volume into the high-resolution spray density field.

    Paper 7 keeps this field at 2-4x the pressure/velocity resolution and in
    16-bit float, and only rasterises it at render time (every 5-10 steps),
    because thin spray is the visually dominant feature and is far below the
    pressure grid's resolution.
    """
    p = wp.tid()
    if pstate[p] != STATE_DROPLET:
        return
    x = px[p]
    r = pdrop_r[p]
    vol = (4.0 / 3.0) * 3.14159265 * r * r * r
    nx, ny, nz = density.shape[0], density.shape[1], density.shape[2]
    g = wp.vec3(x[0] / dx_spray - 0.5, x[1] / dx_spray - 0.5, x[2] / dx_spray - 0.5)
    i0 = int(wp.floor(g[0]))
    j0 = int(wp.floor(g[1]))
    k0 = int(wp.floor(g[2]))
    tx, ty, tz = g[0] - float(i0), g[1] - float(j0), g[2] - float(k0)
    cell_vol = dx_spray * dx_spray * dx_spray
    for a in range(2):
        wa = wp.where(a == 0, 1.0 - tx, tx)
        ii = i0 + a
        if ii < 0 or ii >= nx:
            continue
        for b in range(2):
            wb = wp.where(b == 0, 1.0 - ty, ty)
            jj = j0 + b
            if jj < 0 or jj >= ny:
                continue
            for c in range(2):
                wc = wp.where(c == 0, 1.0 - tz, tz)
                kk = k0 + c
                if kk < 0 or kk >= nz:
                    continue
                wp.atomic_add(density, ii, jj, kk, wp.float16(wa * wb * wc * vol / cell_vol))


# ---------------------------------------------------------------------------
def lognormal_params(mean_radius: float, spread: float = 0.5):
    """``(mu, sigma)`` of a log-normal with the requested mean radius."""
    sigma = spread
    mu = np.log(mean_radius) - 0.5 * sigma * sigma
    return float(mu), float(sigma)
