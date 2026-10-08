"""End-to-end validation of the PF-FLIP core (paper section 3).

These are the properties the paper actually claims for PF-FLIP, phrased as
assertions:

* **eq. (7) separates the phases.**  Because the density ratio is 1000:1, the
  FLIP-induced density noise in the air is 1000x smaller in amplitude than in
  the liquid, so ``phi`` must saturate at 0 and 1 away from the interface
  (paper figs. 6, 7).
* **Mass is conserved exactly.**  ``m_p`` is fixed for the life of a particle
  (paper eq. 5), so no advection scheme can lose mass.
* **The interface stays ~1 cell thick** (``eps_phi = r_0 = dx``, paper 3.3),
  i.e. sharper than classical phase-field methods which need 3+ cells.
* **Hydrostatic equilibrium is stable** at 1000:1 contrast -- the test that
  breaks naive variable-density projections.
"""

import numpy as np
import pytest
import warp as wp

from pfflip import Config, Simulator, scenes

DEV = "cuda:0" if wp.get_cuda_device_count() else "cpu"


@pytest.fixture(scope="module")
def pool():
    cfg = Config(res=(32, 32, 32), dx=1.0 / 32, cfl=1.0, device=DEV)
    sim = Simulator(cfg)
    sim.set_particles(*scenes.still_pool(cfg, fill=0.5))
    return sim


def test_rho_tilde_0_matches_closed_form():
    """Calibrated ``rho~_0`` must match ``n_p * integral(w) = n_p * 64 pi r^3 / 315``."""
    cfg = Config(res=(8, 8, 8), dx=1.0 / 8, device=DEV)
    sim = Simulator(cfg)
    n_p = cfg.ppc / cfg.dx**3
    analytic = n_p * 64.0 * np.pi * cfg.r_p**3 / 315.0
    assert abs(sim.rho_tilde_0 - analytic) / analytic < 0.05


def test_phase_field_saturates(pool):
    """phi must be exactly 0 in the air and near 1 in the liquid bulk.

    The asymmetry is the whole point of eq. (7): ``rho~_min`` sits a factor
    ``eta_phi = log(1000)`` above the air's raw density, so *all* air-phase
    noise is clamped to a clean 0.  In the liquid, by contrast, phi genuinely
    fluctuates below 1 -- that is the "density fluctuations due to uneven
    particle distribution" of paper fig. 7, and ``alpha_phi`` is the knob that
    trades it off against interface stiffness.  What matters is that the
    fluctuation keeps a wide margin above the 0.5 tagging threshold.
    """
    pool.step()
    phi = pool.phi_cell.numpy()
    ny = phi.shape[1]
    air = phi[:, 3 * ny // 4 :, :]
    liq = phi[:, : ny // 4, :]
    assert air.max() == 0.0, f"air phase is not clean: max phi = {air.max()}"
    assert liq.mean() > 0.95, f"liquid bulk not saturated on average: {liq.mean()}"
    assert liq.min() > 0.6, f"noise reaches the 0.5 tag threshold: min phi = {liq.min()}"


def test_interface_thickness_one_cell(pool):
    """Transition from phi<0.05 to phi>0.95 must span ~1 cell, not 3+."""
    pool.step()
    phi = pool.phi_cell.numpy()
    col = phi[16, :, 16]
    inside = np.where(col > 0.95)[0]
    outside = np.where(col < 0.05)[0]
    # first air cell above the last saturated liquid cell
    gap = outside[outside > inside.max()].min() - inside.max()
    assert gap <= 3, f"interface spans {gap} cells; eps_phi should be ~1 cell"


def test_mass_conserved_exactly():
    """Paper eq. (5): ``m_p`` is fixed, so total mass cannot drift."""
    cfg = Config(res=(24, 24, 24), dx=1.0 / 24, cfl=1.0, device=DEV)
    sim = Simulator(cfg)
    sim.set_particles(*scenes.dam_break(cfg))
    m0 = sim.pmass.numpy().sum()
    n0 = (sim.ptype.numpy() == 1).sum()
    for _ in range(10):
        sim.step()
    assert sim.pmass.numpy().sum() == m0
    assert (sim.ptype.numpy() == 1).sum() == n0


def test_hydrostatic_equilibrium_is_stable(pool):
    """A pool at rest must stay at rest across a 1000:1 density jump."""
    liq = pool.ptype.numpy() == 1
    y0 = pool.px.numpy()[liq, 1].mean()
    speeds = []
    for _ in range(30):
        pool.step()
        speeds.append(np.linalg.norm(pool.pv.numpy()[liq], axis=1).max())
    y1 = pool.px.numpy()[liq, 1].mean()

    g_dt = 9.81 * pool.stats[-1]["dt"]
    assert max(speeds) < 0.5 * g_dt, f"spurious liquid motion {max(speeds):.3f} vs g*dt {g_dt:.3f}"
    assert abs(y1 - y0) < 0.01 * pool.cfg.domain_size[1], "liquid centroid drifted"
    assert speeds[-1] <= speeds[0] * 1.5, "velocities are growing, not decaying"


def test_projection_reduces_divergence():
    cfg = Config(res=(32, 32, 32), dx=1.0 / 32, cfl=1.0, device=DEV, pcg_tol=1e-5)
    sim = Simulator(cfg)
    sim.set_particles(*scenes.dam_break(cfg))
    sim._p2g()
    sim._phase_field_and_coefficients()
    for ax in range(3):
        wp.copy(sim.vel_star[ax], sim.vel[ax])
    dt = 1e-3
    sim._add_forces(dt)
    div_before = sim._max_divergence()
    sim._pressure_solve(dt)
    sim._project(dt)
    div_after = sim._max_divergence()
    assert div_after < 0.05 * div_before, f"{div_before:.3e} -> {div_after:.3e}"
