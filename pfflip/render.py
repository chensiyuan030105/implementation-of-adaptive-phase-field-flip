"""Volumetric renderer for the phase field and the spray density (paper 7).

The paper uses an MSBG volumetric ray tracer with approximate multiple
scattering, and reconstructs the water surface by particle skinning.  This is a
single-scattering ray marcher, but one built for the two things that actually
carry the look of a large-scale water shot:

* **the water reads as a surface, not a fog.**  The march detects the
  ``phi = 1/2`` crossing, takes a normal from the phase-field gradient, and
  shades it with a Fresnel mix of sky reflection and an absorbing water body
  plus a sun specular.  Paper 7 gets the same effect from particle skinning.
* **the spray reads as backlit mist.**  Droplet density scatters with a
  Henyey-Greenstein phase function, so spray lit from behind glows along the
  sun direction the way real spray sheets do, instead of looking like uniform
  grey smoke.

High-curvature interface regions are tinted toward a foam response, which paper
7 also does ("foam-like scattering coefficients").

Rendered every 5-10 steps, as the paper does: spray rasterisation and surface
shading are render-time operations.
"""

from __future__ import annotations

import os

import numpy as np
import warp as wp


# ---------------------------------------------------------------------------
@wp.func
def _sample(f: wp.array3d(dtype=float), g: wp.vec3) -> float:
    nx, ny, nz = f.shape[0], f.shape[1], f.shape[2]
    if g[0] < 0.0 or g[1] < 0.0 or g[2] < 0.0:
        return 0.0
    if g[0] > float(nx - 1) or g[1] > float(ny - 1) or g[2] > float(nz - 1):
        return 0.0
    i0, j0, k0 = int(g[0]), int(g[1]), int(g[2])
    tx, ty, tz = g[0] - float(i0), g[1] - float(j0), g[2] - float(k0)
    acc = float(0.0)
    for a in range(2):
        wa = wp.where(a == 0, 1.0 - tx, tx)
        ii = wp.min(i0 + a, nx - 1)
        for b in range(2):
            wb = wp.where(b == 0, 1.0 - ty, ty)
            jj = wp.min(j0 + b, ny - 1)
            for c in range(2):
                wc = wp.where(c == 0, 1.0 - tz, tz)
                kk = wp.min(k0 + c, nz - 1)
                acc += wa * wb * wc * f[ii, jj, kk]
    return acc


@wp.func
def _phi_at(phi: wp.array3d(dtype=float), x: wp.vec3, inv_dx: float) -> float:
    return _sample(phi, wp.vec3(x[0] * inv_dx - 0.5, x[1] * inv_dx - 0.5, x[2] * inv_dx - 0.5))


@wp.func
def _spray_at(sp: wp.array3d(dtype=float), x: wp.vec3, inv_dxs: float) -> float:
    return _sample(sp, wp.vec3(x[0] * inv_dxs - 0.5, x[1] * inv_dxs - 0.5, x[2] * inv_dxs - 0.5))


@wp.func
def _sky(rd: wp.vec3, sun_dir: wp.vec3, horizon: wp.vec3, zenith: wp.vec3,
         sun_col: wp.vec3) -> wp.vec3:
    """Gradient sky with a sun disc and a broad glow around it."""
    t = wp.clamp(0.5 * (rd[1] + 1.0), 0.0, 1.0)
    c = horizon + (zenith - horizon) * wp.pow(t, 0.8)
    d = wp.clamp(wp.dot(rd, sun_dir), 0.0, 1.0)
    c = c + sun_col * 0.30 * wp.pow(d, 12.0)      # glow
    c = c + sun_col * 2.2 * wp.pow(d, 2200.0)     # disc
    return c


@wp.func
def _hg(cos_t: float, g: float) -> float:
    """Henyey-Greenstein phase function; g > 0 is forward scattering."""
    gg = g * g
    d = 1.0 + gg - 2.0 * g * cos_t
    return (1.0 - gg) / (4.0 * 3.14159265 * wp.max(d, 1e-4) * wp.sqrt(wp.max(d, 1e-4)))


@wp.func
def _transmittance(
    phi: wp.array3d(dtype=float),
    sp: wp.array3d(dtype=float),
    x: wp.vec3,
    dir: wp.vec3,
    inv_dx: float,
    inv_dxs: float,
    k_liquid: float,
    k_spray: float,
    dt: float,
    n: int,
) -> float:
    """Shadow ray: optical depth from ``x`` along ``dir``."""
    tau = float(0.0)
    for i in range(n):
        p = x + dir * ((float(i) + 0.5) * dt)
        tau += (k_liquid * _phi_at(phi, p, inv_dx) + k_spray * _spray_at(sp, p, inv_dxs)) * dt
        if tau > 6.0:
            return 0.0
    return wp.exp(-tau)


# ---------------------------------------------------------------------------
@wp.kernel
def raymarch(
    phi: wp.array3d(dtype=float),
    spray: wp.array3d(dtype=float),
    img: wp.array3d(dtype=float),
    cam_pos: wp.vec3,
    cam_fwd: wp.vec3,
    cam_right: wp.vec3,
    cam_up: wp.vec3,
    tan_half_fov: float,
    domain: wp.vec3,
    inv_dx: float,
    inv_dxs: float,
    k_spray: float,
    k_shadow_liquid: float,
    steps: int,
    shadow_steps: int,
    spp: int,
    sun_dir: wp.vec3,
    sun_col: wp.vec3,
    horizon: wp.vec3,
    zenith: wp.vec3,
    water_col: wp.vec3,
    absorb: wp.vec3,
    hg_g: float,
    seed: int,
):
    px, py = wp.tid()
    h = img.shape[0]
    w = img.shape[1]
    aspect = float(w) / float(h)
    out = wp.vec3(0.0, 0.0, 0.0)
    rng = wp.rand_init(seed, px * 7919 + py)

    for s in range(spp):
        jx = wp.where(spp == 1, 0.5, wp.randf(rng))
        jy = wp.where(spp == 1, 0.5, wp.randf(rng))
        u = (2.0 * (float(px) + jx) / float(w) - 1.0) * tan_half_fov * aspect
        v = (1.0 - 2.0 * (float(py) + jy) / float(h)) * tan_half_fov
        rd = wp.normalize(cam_fwd + u * cam_right + v * cam_up)

        # slab intersection with the domain box
        t0 = float(0.0)
        t1 = float(1.0e9)
        for a in range(3):
            o = cam_pos[a]
            d = rd[a]
            if wp.abs(d) < 1.0e-9:
                if o < 0.0 or o > domain[a]:
                    t1 = -1.0
            else:
                ta = (0.0 - o) / d
                tb = (domain[a] - o) / d
                t0 = wp.max(t0, wp.min(ta, tb))
                t1 = wp.min(t1, wp.max(ta, tb))

        bg = _sky(rd, sun_dir, horizon, zenith, sun_col)
        col = bg
        if t1 > t0:
            dt = (t1 - t0) / float(steps)
            sdt = dt * 2.5
            trans = float(1.0)
            acc = wp.vec3(0.0, 0.0, 0.0)
            cos_t = wp.dot(rd, sun_dir)
            phase = _hg(cos_t, hg_g)
            hit = int(0)
            prev_phi = float(0.0)

            for i in range(steps):
                if trans < 0.004:
                    break
                tcur = t0 + (float(i) + 0.5) * dt
                x = cam_pos + rd * tcur
                f = _phi_at(phi, x, inv_dx)

                # ---- water surface: the phi = 1/2 crossing -------------------
                if f >= 0.5 and hit == 0:
                    hit = 1
                    # refine the crossing by one bisection against the previous
                    # sample so the surface is not quantised to the step size
                    if prev_phi < 0.5:
                        frac = (0.5 - prev_phi) / wp.max(f - prev_phi, 1e-6)
                        x = cam_pos + rd * (tcur - dt * (1.0 - frac))
                    e = 0.75 / inv_dx
                    gx = _phi_at(phi, x + wp.vec3(e, 0.0, 0.0), inv_dx) - _phi_at(phi, x - wp.vec3(e, 0.0, 0.0), inv_dx)
                    gy = _phi_at(phi, x + wp.vec3(0.0, e, 0.0), inv_dx) - _phi_at(phi, x - wp.vec3(0.0, e, 0.0), inv_dx)
                    gz = _phi_at(phi, x + wp.vec3(0.0, 0.0, e), inv_dx) - _phi_at(phi, x - wp.vec3(0.0, 0.0, e), inv_dx)
                    gl = wp.length(wp.vec3(gx, gy, gz))
                    nrm = wp.vec3(0.0, 1.0, 0.0)
                    if gl > 1e-7:
                        nrm = wp.vec3(-gx / gl, -gy / gl, -gz / gl)  # points out of liquid

                    ndl = wp.max(wp.dot(nrm, sun_dir), 0.0)
                    sh = _transmittance(phi, spray, x + nrm * (1.5 / inv_dx), sun_dir,
                                        inv_dx, inv_dxs, k_shadow_liquid, k_spray,
                                        sdt, shadow_steps)
                    refl = rd - nrm * (2.0 * wp.dot(rd, nrm))
                    sky_r = _sky(wp.normalize(refl), sun_dir, horizon, zenith, sun_col)

                    # Schlick Fresnel for water, F0 = 0.02
                    ct = wp.clamp(-wp.dot(rd, nrm), 0.0, 1.0)
                    fres = 0.02 + 0.98 * wp.pow(1.0 - ct, 5.0)

                    # body: diffuse plus Beer-Lambert tint for the light that
                    # travelled through water before coming back out
                    depth = wp.max(0.35 / inv_dx, 0.0)
                    atten = wp.vec3(wp.exp(-absorb[0] * depth), wp.exp(-absorb[1] * depth),
                                    wp.exp(-absorb[2] * depth))
                    body = wp.cw_mul(water_col, atten) * (0.25 + 0.75 * ndl * sh)
                    body = body + wp.cw_mul(sun_col, water_col) * 0.10 * sh

                    spec = wp.pow(wp.max(wp.dot(wp.normalize(refl), sun_dir), 0.0), 900.0)

                    # foam where the interface is sharply curved (paper 7)
                    foam = wp.clamp((gl * (1.0 / inv_dx) - 0.55) * 2.2, 0.0, 1.0)
                    surf = body * (1.0 - fres) + sky_r * fres
                    surf = surf + sun_col * spec * 1.6 * sh
                    surf = surf + wp.vec3(1.0, 1.0, 1.0) * foam * 0.30 * (0.35 + 0.65 * sh)

                    acc = acc + surf * trans
                    trans = 0.0
                    break
                prev_phi = f

                # ---- spray ---------------------------------------------------
                sd = _spray_at(spray, x, inv_dxs)
                if sd > 0.0:
                    sig = k_spray * sd
                    a_i = 1.0 - wp.exp(-sig * dt)
                    sh = _transmittance(phi, spray, x, sun_dir, inv_dx, inv_dxs,
                                        k_shadow_liquid, k_spray, sdt, shadow_steps)
                    lit = sun_col * (sh * phase * 4.0) + bg * 0.55
                    acc = acc + lit * (trans * a_i * 0.92)
                    trans *= 1.0 - a_i

            col = acc + bg * trans
        out = out + col

    out = out / float(spp)
    img[py, px, 0] = out[0]
    img[py, px, 1] = out[1]
    img[py, px, 2] = out[2]


# ---------------------------------------------------------------------------
def orbit_camera(domain, azimuth, elevation, distance, target_frac=(0.5, 0.33, 0.5)):
    d = np.asarray(domain, float)
    tgt = d * np.asarray(target_frac, float)
    a, e = np.radians(azimuth), np.radians(elevation)
    off = np.array([np.cos(e) * np.cos(a), np.sin(e), np.cos(e) * np.sin(a)])
    pos = tgt + off * distance * float(np.linalg.norm(d))
    fwd = tgt - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return pos, fwd, right, up


_PRESETS = {
    "overcast": dict(sun_col=(1.15, 1.16, 1.20), horizon=(0.62, 0.66, 0.72),
                     zenith=(0.34, 0.40, 0.50), water_col=(0.10, 0.26, 0.32),
                     absorb=(1.2, 0.35, 0.18)),
    "golden": dict(sun_col=(1.70, 1.26, 0.82), horizon=(0.72, 0.60, 0.52),
                   zenith=(0.18, 0.28, 0.46), water_col=(0.08, 0.22, 0.30),
                   absorb=(1.6, 0.45, 0.22)),
    "storm": dict(sun_col=(1.05, 1.05, 1.10), horizon=(0.40, 0.44, 0.50),
                  zenith=(0.14, 0.17, 0.22), water_col=(0.07, 0.17, 0.21),
                  absorb=(1.5, 0.50, 0.30)),
}


def render(
    sim,
    path: str,
    *,
    width=1600,
    height=900,
    azimuth=38.0,
    elevation=11.0,
    distance=1.05,
    fov=36.0,
    steps=640,
    shadow_steps=20,
    spp=2,
    k_spray=130.0,
    k_shadow_liquid=90.0,
    hg_g=0.45,
    sun_dir=(-0.62, 0.42, -0.66),
    preset="overcast",
    exposure=1.0,
    seed=0,
    spray=None,
):
    """Render the current simulation state to a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = sim.cfg
    dev = sim.dev
    if spray is None:
        spray = sim.rasterize_spray()
    spray_f = wp.zeros(spray.shape, dtype=float, device=dev)

    @wp.kernel
    def _to_f32(a: wp.array3d(dtype=wp.float16), b: wp.array3d(dtype=float)):
        i, j, k = wp.tid()
        b[i, j, k] = float(a[i, j, k])

    wp.launch(_to_f32, dim=spray.shape, inputs=[spray, spray_f], device=dev)

    pos, fwd, right, up = orbit_camera(cfg.domain_size, azimuth, elevation, distance)
    sd = np.asarray(sun_dir, float)
    sd /= np.linalg.norm(sd)
    pal = _PRESETS[preset]

    img = wp.zeros((height, width, 3), dtype=float, device=dev)
    wp.launch(
        raymarch,
        dim=(width, height),
        inputs=[
            sim.phi_cell, spray_f, img,
            wp.vec3(*pos), wp.vec3(*fwd), wp.vec3(*right), wp.vec3(*up),
            float(np.tan(np.radians(fov) * 0.5)),
            wp.vec3(*cfg.domain_size),
            1.0 / cfg.dx, 1.0 / sim.dx_spray,
            k_spray, k_shadow_liquid, steps, shadow_steps, spp,
            wp.vec3(*sd), wp.vec3(*pal["sun_col"]),
            wp.vec3(*pal["horizon"]), wp.vec3(*pal["zenith"]),
            wp.vec3(*pal["water_col"]), wp.vec3(*pal["absorb"]),
            hg_g, seed,
        ],
        device=dev,
    )
    a = img.numpy() * exposure
    a = a / (1.0 + a)                        # Reinhard tonemap
    a = np.clip(a, 0.0, 1.0) ** (1.0 / 2.2)  # gamma
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.imsave(path, a)
    return path
