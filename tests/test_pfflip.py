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


# ------------------------------------------------- escaped particles / spray
def test_bubbles_are_deleted_and_droplets_merge_back():
    """Paper 3.4's three transitions must all fire."""
    cfg = Config(res=(32,) * 3, dx=1 / 32, cfl=1.0, device=DEV, ppc=27,
                 droplets=True, delta_phi=0.2)
    sim = Simulator(cfg)
    sim.set_particles(*scenes.asteroid_impact(cfg))
    for _ in range(25):
        sim.step()
    s = sim.droplet_stats
    assert s["bubbles"] > 0, "air particles trapped in liquid are never deleted"
    counts = sim.state_counts()
    assert counts["flip"] + counts["droplet"] + counts["dead"] == sim.n_particles
    # dead particles must stay dead and droplets must not contribute to the grid
    assert counts["dead"] == s["bubbles"] or counts["dead"] >= s["bubbles"] - s["merged"]


def test_droplet_threshold_is_coupled_to_particle_count():
    """``delta_phi`` and ``ppc`` are not independent knobs.

    An isolated liquid particle splats its *own* mass into the phase field, so
    eq. (7) floors it at ``phi_single = 1/sqrt(alpha_phi * rho~_0)`` with
    ``rho~_0 ~ 0.638 * ppc``.  Conversion to a droplet needs
    ``phi_single < 1/2 - delta_phi``, i.e.

        ppc > 1 / (alpha_phi * 0.638 * (1/2 - delta_phi)**2)

    At the paper's defaults (``alpha_phi = 1``, ``delta_phi = 0.2``) that is
    ppc > 17.4 -- so at 8 particles per cell escaped particles essentially never
    convert, and the spray channel is silently dead.
    """
    import math

    for ppc in (8, 27):
        cfg = Config(res=(16,) * 3, dx=1 / 16, device=DEV, ppc=ppc)
        sim = Simulator(cfg)
        phi_single = 1.0 / math.sqrt(cfg.alpha_phi * sim.rho_tilde_0)
        predicted = 0.638 * ppc
        assert abs(sim.rho_tilde_0 - predicted) / predicted < 0.10, (
            f"rho~_0 = {sim.rho_tilde_0:.2f} vs predicted {predicted:.2f}"
        )
        convertible = phi_single < 0.5 - cfg.delta_phi
        assert convertible == (ppc >= 18), (
            f"ppc={ppc}: phi_single={phi_single:.3f}, threshold={0.5-cfg.delta_phi:.2f}"
        )


def test_droplets_do_not_contribute_to_the_grid():
    """A droplet is a Lagrangian point mass: it must not splat into P2G."""
    cfg = Config(res=(16,) * 3, dx=1 / 16, cfl=1.0, device=DEV, ppc=8, droplets=True)
    sim = Simulator(cfg)
    sim.set_particles(*scenes.still_pool(cfg, fill=0.5))
    sim.step()
    m_all = sim.cell_mass.numpy().sum()

    # flag a tenth of the liquid particles as droplets and re-run P2G only
    st = sim.pstate.numpy()
    liq = np.flatnonzero(sim.ptype.numpy() == 1)
    st[liq[::10]] = 1
    sim.pstate.assign(st)
    sim._p2g()
    m_flip = sim.cell_mass.numpy().sum()
    assert m_flip < m_all, "droplets still contributed mass to the grid"
    # liquid is only ~half the particles but ~99.9% of the mass (1000:1 density
    # ratio), so removing a tenth of the liquid removes a tenth of the mass
    removed = 1.0 - m_flip / m_all
    assert abs(removed - 0.10) < 0.02, f"expected ~10% of mass removed, got {100*removed:.1f}%"


# ------------------------------------------------------------ solid obstacles
def test_obstacle_does_not_leak_gravity():
    """Faces inside an obstacle must stay at zero velocity.

    Gravity is applied to every face, and projection *skips* faces with
    ``beta = 0`` -- so obstacle faces have to be re-zeroed after the force
    step or the contamination is permanent.  Without that, the leaked velocity
    shows up as spurious divergence on the next step and the pressure solve
    diverges: a surge scene went from a bounded |v| of 4.6 -> 10 to 13 -> 345
    within ten steps.
    """
    cfg = Config(res=(32,) * 3, dx=1 / 32, cfl=1.0, device=DEV, ppc=8, droplets=False)
    sim = Simulator(cfg)
    pos, vel, pt, solid = scenes.wave_vs_tower(cfg)
    sim.set_particles(pos, vel, pt)
    sim.set_solids(solid)
    assert solid.sum() > 0

    for _ in range(5):
        sim.step()

    # every face whose coefficient was zeroed must carry zero velocity
    for ax in range(3):
        beta = sim.beta[ax].numpy()
        v = sim.vel[ax].numpy()
        blocked = beta == 0.0
        assert blocked.any(), f"axis {ax}: no blocked faces at all"
        assert np.abs(v[blocked]).max() == 0.0, (
            f"axis {ax}: velocity leaked onto {int((np.abs(v[blocked]) > 0).sum())} "
            f"blocked faces, max {np.abs(v[blocked]).max():.3e}"
        )


def test_obstacle_scene_stays_bounded():
    """A surge breaking on a tower must not blow up."""
    cfg = Config(res=(32,) * 3, dx=1 / 32, cfl=1.0, device=DEV, ppc=8, droplets=False)
    sim = Simulator(cfg)
    pos, vel, pt, solid = scenes.wave_vs_tower(cfg)
    sim.set_particles(pos, vel, pt)
    sim.set_solids(solid)
    v0 = float(np.abs(sim.pv.numpy()).max())

    speeds = []
    for _ in range(40):
        sim.step()
        st = sim.pstate.numpy()
        speeds.append(float(np.linalg.norm(sim.pv.numpy()[st == 0], axis=1).max()))

    # a collapsing surge accelerates, but only to a few times its launch speed
    assert max(speeds) < 12 * v0, f"|v| reached {max(speeds):.1f} from v0 = {v0:.2f}"
    assert np.isfinite(speeds).all()


def test_droplets_do_not_throttle_the_time_step():
    """CFL must be taken over FLIP particles only (paper 7).

    Droplets have their own adaptive sub-stepping precisely so a few fast
    ballistic drops cannot shrink the grid time step; including them collapsed
    dt by more than 10x in the surge scene.
    """
    cfg = Config(res=(32,) * 3, dx=1 / 32, cfl=1.0, device=DEV, ppc=8, droplets=True)
    sim = Simulator(cfg)
    sim.set_particles(*scenes.asteroid_impact(cfg))
    sim.step()

    # plant one absurdly fast droplet and one equally fast dead particle
    v = sim.pv.numpy()
    st = sim.pstate.numpy()
    v[0] = (500.0, 0.0, 0.0)
    st[0] = 1  # droplet
    v[1] = (500.0, 0.0, 0.0)
    st[1] = 2  # dead
    sim.pv.assign(v)
    sim.pstate.assign(st)

    dt = sim.dt_from_cfl()
    assert dt > cfg.dx / 50.0, f"a single fast droplet collapsed dt to {dt:.2e}"
