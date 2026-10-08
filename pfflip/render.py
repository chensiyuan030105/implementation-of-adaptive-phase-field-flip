"""Volumetric renderer for the phase field and the spray density (paper 7).

The paper renders with an MSBG volumetric ray tracer using approximate multiple
scattering, and reconstructs the water surface by particle skinning.  This is a
simpler single-scattering ray marcher over two volumes -- the liquid phase field
and the high-resolution spray density -- with a shadow ray to the sun.  It
reproduces the *look* that matters here (dense water body, wispy backlit spray)
without the surface reconstruction or multiple-scattering machinery.

Rendered every 5-10 steps, as the paper does: spray rasterisation and surface
reconstruction are render-time operations, not simulation ones.
"""

from __future__ import annotations

import numpy as np
import warp as wp


@wp.func
def _sample(f: wp.array3d(dtype=float), g: wp.vec3) -> float:
    """Trilinear sample in index space; zero outside."""
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
def _extinction(
    phi: wp.array3d(dtype=float),
    spray: wp.array3d(dtype=float),
    x: wp.vec3,
    inv_dx: float,
    inv_dxs: float,
    k_liquid: float,
    k_spray: float,
) -> float:
    gl = wp.vec3(x[0] * inv_dx - 0.5, x[1] * inv_dx - 0.5, x[2] * inv_dx - 0.5)
    gs = wp.vec3(x[0] * inv_dxs - 0.5, x[1] * inv_dxs - 0.5, x[2] * inv_dxs - 0.5)
    f = wp.clamp(_sample(phi, gl), 0.0, 1.0)
    s = wp.max(_sample(spray, gs), 0.0)
    return k_liquid * f + k_spray * s


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
    k_liquid: float,
    k_spray: float,
    steps: int,
    shadow_steps: int,
    sun_dir: wp.vec3,
    sun_col: wp.vec3,
    sky_lo: wp.vec3,
    sky_hi: wp.vec3,
    water_col: wp.vec3,
):
    px, py = wp.tid()
    h = img.shape[0]
    w = img.shape[1]
    u = (2.0 * (float(px) + 0.5) / float(w) - 1.0) * tan_half_fov * float(w) / float(h)
    v = (1.0 - 2.0 * (float(py) + 0.5) / float(h)) * tan_half_fov
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

    # background: a simple vertical sky gradient
    sky_t = wp.clamp(0.5 * (rd[1] + 1.0), 0.0, 1.0)
    bg = sky_lo + (sky_hi - sky_lo) * sky_t
    col = bg
    if t1 > t0:
        dt = (t1 - t0) / float(steps)
        trans = float(1.0)
        acc = wp.vec3(0.0, 0.0, 0.0)
        for i in range(steps):
            if trans < 0.003:
                break
            x = cam_pos + rd * (t0 + (float(i) + 0.5) * dt)
            sig = _extinction(phi, spray, x, inv_dx, inv_dxs, k_liquid, k_spray)
            if sig <= 0.0:
                continue
            a_i = 1.0 - wp.exp(-sig * dt)

            # single-scattering shadow ray towards the sun
            sh = float(1.0)
            sdt = dt * 2.0
            for j in range(shadow_steps):
                xs = x + sun_dir * (float(j) + 0.5) * sdt
                ss = _extinction(phi, spray, xs, inv_dx, inv_dxs, k_liquid, k_spray)
                sh *= wp.exp(-ss * sdt)
                if sh < 0.01:
                    break

            lit = sun_col * sh + bg * 0.45
            # water reads as its own colour, spray stays near-white
            gl = wp.vec3(x[0] * inv_dx - 0.5, x[1] * inv_dx - 0.5, x[2] * inv_dx - 0.5)
            fl = wp.clamp(_sample(phi, gl), 0.0, 1.0)
            albedo = wp.vec3(1.0, 1.0, 1.0) * (1.0 - fl) + water_col * fl
            acc = acc + trans * a_i * wp.cw_mul(lit, albedo)
            trans *= 1.0 - a_i
        col = acc + bg * trans

    img[py, px, 0] = col[0]
    img[py, px, 1] = col[1]
    img[py, px, 2] = col[2]


def _orbit_camera(domain, azimuth, elevation, distance, target=None):
    d = np.asarray(domain, float)
    tgt = d * np.array([0.5, 0.35, 0.5]) if target is None else np.asarray(target, float)
    a, e = np.radians(azimuth), np.radians(elevation)
    off = np.array([np.cos(e) * np.cos(a), np.sin(e), np.cos(e) * np.sin(a)])
    pos = tgt + off * distance * float(np.linalg.norm(d))
    fwd = tgt - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return pos, fwd, right, up


def render(
    sim,
    path: str,
    *,
    width=960,
    height=540,
    azimuth=35.0,
    elevation=12.0,
    distance=1.15,
    fov=38.0,
    steps=320,
    shadow_steps=12,
    k_liquid=220.0,
    k_spray=55.0,
    sun_dir=(-0.45, 0.72, -0.52),
    exposure=1.0,
):
    """Render the current state to a PNG."""
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = sim.cfg
    dev = sim.dev
    spray = sim.rasterize_spray()
    spray_f = wp.zeros(spray.shape, dtype=float, device=dev)

    @wp.kernel
    def _to_f32(a: wp.array3d(dtype=wp.float16), b: wp.array3d(dtype=float)):
        i, j, k = wp.tid()
        b[i, j, k] = float(a[i, j, k])

    wp.launch(_to_f32, dim=spray.shape, inputs=[spray, spray_f], device=dev)

    pos, fwd, right, up = _orbit_camera(cfg.domain_size, azimuth, elevation, distance)
    sd = np.asarray(sun_dir, float)
    sd /= np.linalg.norm(sd)

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
            k_liquid, k_spray, steps, shadow_steps,
            wp.vec3(*sd),
            wp.vec3(1.25, 1.18, 1.05),   # sun
            wp.vec3(0.42, 0.50, 0.62),   # sky low
            wp.vec3(0.14, 0.22, 0.38),   # sky high
            wp.vec3(0.18, 0.42, 0.55),   # water
        ],
        device=dev,
    )
    a = img.numpy() * exposure
    a = a / (1.0 + a)                      # Reinhard tonemap
    a = np.clip(a, 0.0, 1.0) ** (1.0 / 2.2)  # gamma
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.imsave(path, a)
    return path
