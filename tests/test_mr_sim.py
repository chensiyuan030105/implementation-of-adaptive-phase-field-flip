"""Multiresolution P2G/G2P and the adaptive simulator (paper 5.4).

The load-bearing test is :func:`test_msbg_uniform_matches_stage1_hydrostatics`.
Forcing ``max_level = 0`` makes MSBG a plain uniform grid, so the whole
multiresolution path -- block-wise P2G, per-level phase-field normalisation,
finite-volume rhs, projection on owned faces, MSBG velocity interpolation --
must reproduce the stage-1 result exactly in character.  If it does not, the
machinery is wrong independently of any adaptivity.
"""

import numpy as np
import pytest
import warp as wp

from pfflip import Config
from pfflip.mr_sim import AdaptiveSimulator

DEV = "cuda:0" if wp.get_cuda_device_count() else "cpu"


def _seed_pool(sim, cfg, fill=0.5, per=2):
    rng = np.random.default_rng(0)
    nb, n0 = sim.grid.nb, sim.grid.n0
    L = nb[0] * n0 * per
    step = cfg.dx / per
    idx = np.stack(np.meshgrid(*[np.arange(L)] * 3, indexing="ij"), -1).reshape(-1, 3)
    pos = ((idx + 0.5) * step + (rng.random((len(idx), 3)) - 0.5) * step * 0.8).astype(np.float32)
    liq = pos[:, 1] < fill * sim.domain[1]
    sim.set_particles(pos, np.zeros_like(pos), np.where(liq, 1, 0).astype(np.int32))
    return liq


def _liquid_stats(sim):
    n = sim.particles.n
    t = sim.particles.type.numpy()[:n] == 1
    v = np.linalg.norm(sim.particles.v.numpy()[:n], axis=1)
    y = sim.particles.x.numpy()[:n, 1]
    return v[t].max(), y[t].mean()


def test_msbg_uniform_matches_stage1_hydrostatics():
    """With max_level = 0 the MSBG path must hold a pool at rest."""
    cfg = Config(res=(32,) * 3, dx=1 / 32, cfl=1.0, device=DEV, pcg_max_iter=1500)
    sim = AdaptiveSimulator(cfg, (4, 4, 4), n0=8, max_level=0)
    _seed_pool(sim, cfg)

    speeds, y0 = [], None
    for _ in range(10):
        sim.step()
        v, y = _liquid_stats(sim)
        speeds.append(v)
        if y0 is None:
            y0 = y
    v_end, y_end = _liquid_stats(sim)

    g_dt = 9.81 * sim.stats[-1]["dt"]
    assert max(speeds) < 0.6 * g_dt, f"spurious motion {max(speeds):.3f} vs g*dt {g_dt:.3f}"
    assert abs(y_end - y0) < 1e-3, f"liquid centroid drifted {y_end - y0:+.2e}"
    assert sim.grid.compression == pytest.approx(1.0), "max_level=0 must give a uniform grid"


def test_p2g_recovers_uniform_density_at_every_level():
    """The per-level normalisation of eq. (7) must make phi level-independent.

    A level-l particle carries 2**(3l) the mass of a level-0 one, so the raw
    accumulator scales the same way; dividing it out is what lets the *same*
    eq. (7) constants tag phases correctly on a coarse block and a fine one.

    The normalisation assumes the particle *density* matches the level -- i.e.
    8**-l of the level-0 count, which is exactly what probabilistic coarsening
    produces.  Promoting particles without thinning them inflates the raw
    density by 8**l and the air phase stops clamping to zero (at level 2 it
    reads phi ~ sqrt(64/1000) ~ 0.25).  So the particle cloud is thinned here
    the same way the resample would thin it.
    """
    for max_level in (0, 1, 2):
        cfg = Config(res=(32,) * 3, dx=1 / 32, cfl=1.0, device=DEV, pcg_max_iter=50)
        sim = AdaptiveSimulator(cfg, (4, 4, 4), n0=8, max_level=max_level)
        _seed_pool(sim, cfg)
        # thin to the density a level-`max_level` cloud must have, then promote
        full = sim.particles.n
        rng = np.random.default_rng(1)
        # random thinning, not a strided slice: the lattice is stored with k
        # fastest, so a stride of 8**l collapses onto a single plane.  Random
        # selection is also what probabilistic coarsening actually produces.
        sel = rng.choice(full, size=full // 8**max_level, replace=False)
        pos = sim.particles.x.numpy()[:full][sel]
        typ = sim.particles.type.numpy()[:full][sel]
        sim.set_particles(pos, np.zeros_like(pos), typ,
                          np.full(len(pos), max_level, np.int32))
        n = sim.particles.n
        sim._rebuild_grid(1e-3)
        sim.particles.update_mass_radius(cfg.r_p, cfg.rho_l, cfg.rho_g)
        for a in (sim.fmass, sim.fmom, sim.cmass, sim.cwsum):
            a.zero_()
        from pfflip import mr_kernels as MK

        wp.launch(MK.p2g, dim=n,
                  inputs=[sim.grid.g, sim.particles.x, sim.particles.v,
                          sim.particles.mass, sim.particles.rad,
                          sim.fmass, sim.fmom, sim.cmass, sim.cwsum], device=DEV)
        wp.launch(MK.cell_phase, dim=sim.grid.ncells,
                  inputs=[sim.grid.g, sim.cmass, sim.phi, sim.is_liquid,
                          sim.rho_min, cfg.alpha_phi, sim.rho_0l], device=DEV)
        phi = sim.phi.numpy()
        assert phi.max() > 0.9, f"level {max_level}: liquid never saturates (max {phi.max():.3f})"
        assert phi.min() == 0.0, f"level {max_level}: air not clamped to zero"
        frac = (sim.is_liquid.numpy() > 0).mean()
        assert 0.4 < frac < 0.6, f"level {max_level}: liquid fraction {frac:.3f}, expected ~0.5"


def test_adaptive_run_conserves_liquid_centroid():
    """A pool at rest must keep its centroid through refine/coarsen cycles.

    This is the test that caught the split-jitter bug: children were being
    scattered using r0 = 1.0 (world units) instead of the finest grid spacing,
    which teleported the liquid centroid by 0.13 in a single step.
    """
    cfg = Config(res=(32,) * 3, dx=1 / 32, cfl=1.0, device=DEV, pcg_max_iter=800)
    sim = AdaptiveSimulator(cfg, (4, 4, 4), n0=8, max_level=1, beta_bw=0.34)
    _seed_pool(sim, cfg)

    ys = []
    for _ in range(8):
        sim.step()
        ys.append(_liquid_stats(sim)[1])
    drift = max(ys) - min(ys)
    assert drift < 0.02 * sim.domain[1], f"liquid centroid wandered by {drift:.4f}"
    assert any(r["compression"] > 1.5 for r in sim.stats), "grid never coarsened at all"
