"""Scene setups.

Both phases are seeded everywhere -- that is the point of a two-phase solver:
there is no free surface, the air is a real fluid with its own particles
(paper section 1, "Necessity of Two-Phase Simulation").
"""

from __future__ import annotations

import numpy as np

from .config import Config

AIR, LIQUID = 0, 1


def _lattice(res, dx, ppc, rng):
    """Jittered ``ppc``-per-cell lattice covering the whole domain."""
    per_axis = round(ppc ** (1.0 / 3.0))
    assert per_axis**3 == ppc, "ppc must be a perfect cube (8, 27, ...)"
    nx, ny, nz = res
    step = dx / per_axis
    idx = np.stack(
        np.meshgrid(
            np.arange(nx * per_axis), np.arange(ny * per_axis), np.arange(nz * per_axis), indexing="ij"
        ),
        axis=-1,
    ).reshape(-1, 3)
    pos = (idx + 0.5) * step
    pos += (rng.random(pos.shape) - 0.5) * step * 0.8
    return pos


def seed(cfg: Config, liquid_mask_fn, velocity_fn=None):
    """Build particles from a predicate on world position.

    ``liquid_mask_fn(pos) -> bool array`` marks the liquid region;
    ``velocity_fn(pos, is_liquid) -> (n,3)`` optionally sets initial velocity.
    """
    rng = np.random.default_rng(cfg.seed)
    pos = _lattice(cfg.res, cfg.dx, cfg.ppc, rng)
    liq = np.asarray(liquid_mask_fn(pos), dtype=bool)
    ptype = np.where(liq, LIQUID, AIR).astype(np.int32)
    vel = np.zeros_like(pos) if velocity_fn is None else np.asarray(velocity_fn(pos, liq), float)
    return pos.astype(np.float32), vel.astype(np.float32), ptype


# ---------------------------------------------------------------------------
def dam_break(cfg: Config, frac=(0.4, 0.6, 1.0)):
    """Classic dam break: a liquid column collapsing in a closed box.

    Paper 8.4 runs a dam-break scene; here it is the primary stage-1
    correctness and stability test.
    """
    sx, sy, sz = cfg.domain_size

    def mask(p):
        return (p[:, 0] < frac[0] * sx) & (p[:, 1] < frac[1] * sy) & (p[:, 2] < frac[2] * sz)

    return seed(cfg, mask)


def still_pool(cfg: Config, fill=0.5):
    """Hydrostatic rest state -- should stay at rest (validation scene)."""
    sy = cfg.domain_size[1]

    def mask(p):
        return p[:, 1] < fill * sy

    return seed(cfg, mask)


def asteroid_impact(cfg: Config, *, water_level=0.45, radius_frac=0.08, speed=None):
    """A dense sphere fired into a deep pool.

    This is the motivating example of paper figure 2: a single-phase solver
    makes the splash rim rise unrealistically because the air exerts no drag,
    whereas the two-phase solve produces the spray streamers.
    """
    sx, sy, sz = cfg.domain_size
    r = radius_frac * sx
    centre = np.array([0.5 * sx, water_level * sy + 0.35 * sy, 0.5 * sz])
    v_impact = speed if speed is not None else -0.9 * np.sqrt(2 * 9.81 * 0.35 * sy)

    def mask(p):
        in_pool = p[:, 1] < water_level * sy
        in_ball = np.linalg.norm(p - centre, axis=1) < r
        return in_pool | in_ball

    def vel(p, liq):
        v = np.zeros_like(p)
        in_ball = np.linalg.norm(p - centre, axis=1) < r
        v[in_ball, 1] = v_impact
        return v

    return seed(cfg, mask, vel)


def rayleigh_taylor(cfg: Config, *, ratio=None):
    """Heavy phase on top of light phase with a perturbed interface.

    Paper 8.2 uses this as a basic phase-field test with ``alpha_phi = 0.5``
    and ``CFL = 0.5``.
    """
    sx, sy, sz = cfg.domain_size

    def mask(p):
        h = 0.5 * sy + 0.02 * sy * (
            np.cos(2 * np.pi * p[:, 0] / sx) + np.cos(2 * np.pi * p[:, 2] / sz)
        )
        return p[:, 1] > h  # heavy phase on top

    return seed(cfg, mask)


SCENES = {
    "dam_break": dam_break,
    "still_pool": still_pool,
    "asteroid": asteroid_impact,
    "rayleigh_taylor": rayleigh_taylor,
}
