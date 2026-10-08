"""Adaptive particles and dual adaptivity (paper section 5).

The statistical claim of probabilistic coarsening is the one worth testing
properly: deleting each marked particle independently with probability
``1 - 8**-(l_new - l_p)`` must thin a uniform distribution by exactly the
volume ratio, *without* any neighbourhood search.  If that expectation is off,
mass drifts every time the interface moves.
"""

import numpy as np
import pytest
import warp as wp

from pfflip.msbg import MSBG
from pfflip.particles import Particles, block_distance_field

DEV = "cuda:0" if wp.get_cuda_device_count() else "cpu"


def uniform_particles(n_per_axis, extent, level=1):
    g = np.stack(
        np.meshgrid(*[(np.arange(n_per_axis) + 0.5) / n_per_axis * extent] * 3, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    return g.astype(np.float32)


def make_set(pos, level):
    ps = Particles(DEV, capacity=max(64, len(pos) * 12))
    ps.set(pos, np.zeros_like(pos), np.ones(len(pos), np.int32), np.full(len(pos), level, np.int32))
    return ps


# ------------------------------------------------------------------ distance
def test_block_distance_is_a_band():
    mask = np.zeros((16, 16, 16), bool)
    mask[8] = True  # an interface plane
    d = block_distance_field(mask)
    assert d[8].max() == 0.0
    assert np.allclose(d[:, 0, 0], np.abs(np.arange(16) - 8))


# ------------------------------------------------------------------ coarsening
def test_probabilistic_coarsening_thins_by_eight():
    """One level of coarsening must keep ~1/8 of the particles (paper 5.1).

    This is the whole scheme: a uniform cloud coarsened by 2x per axis is a
    uniform cloud with 8x lower density, which random deletion reproduces in
    expectation without inspecting any neighbours.
    """
    pos = uniform_particles(24, 1.0)  # 13824 particles
    ps = make_set(pos, level=0)
    # d~ = 1.5 blocks with beta_bw = 1 -> l_new = floor(1.5) = 1, a single
    # level of coarsening.  (A larger d~ would demand a multi-level jump, which
    # the same formula handles by deleting 1 - 8**-(dl) instead.)
    dist = wp.array(np.full((4, 4, 4), 1.5, np.float32), dtype=float, device=DEV)

    kept = None
    for step in range(5):  # hysteresis must delay the first three attempts
        before, after = ps.resample(dist, 0.25, beta_bw=1.0, max_level=3, seed=step)
        if step < 2:
            assert after == before, f"coarsening fired early at step {step} (D_p < 3)"
        else:
            kept = after / before
            break
    assert kept is not None
    assert 0.11 < kept < 0.14, f"survival fraction {kept:.3f}, expected ~1/8"
    assert (ps.level.numpy()[: ps.n] == 1).all(), "survivors must be promoted"


def test_coarsening_hysteresis_resets():
    """``D_p`` must reset as soon as the criterion stops asking to coarsen."""
    pos = uniform_particles(12, 1.0)
    ps = make_set(pos, level=0)
    far = wp.array(np.full((4, 4, 4), 1.5, np.float32), dtype=float, device=DEV)
    near = wp.array(np.zeros((4, 4, 4), np.float32), dtype=float, device=DEV)

    for _ in range(2):
        ps.resample(far, 0.25, beta_bw=1.0, max_level=3)  # D_p -> 2
    assert ps.defer.numpy()[: ps.n].max() == 2
    ps.resample(near, 0.25, beta_bw=1.0, max_level=3)  # criterion says stay fine
    assert ps.defer.numpy()[: ps.n].max() == 0, "hysteresis counter did not reset"


# ------------------------------------------------------------------ refinement
def test_splitting_produces_eight_children_in_place():
    """Refinement splits 1 -> 8 one level finer, inside the parent's cell."""
    pos = np.array([[0.5, 0.5, 0.5]], np.float32)
    ps = make_set(pos, level=2)
    near = wp.array(np.zeros((4, 4, 4), np.float32), dtype=float, device=DEV)
    before, after = ps.resample(near, 0.25, beta_bw=1.0, max_level=3)
    assert (before, after) == (1, 8)
    assert (ps.level.numpy()[:8] == 1).all(), "children must be one level finer"
    child = ps.x.numpy()[:8]
    # all children stay within the parent's cell (half-extent r0 * 2**l_p)
    assert np.abs(child - pos).max() < 1.0 * 2**2, "children escaped the parent cell"
    # and they must be spread across the 8 octants, not stacked
    octants = {tuple(np.sign(c - pos[0]).astype(int)) for c in child}
    assert len(octants) == 8, f"children occupy only {len(octants)} octants"


def test_mass_scales_with_level():
    """``m_p = rho * 2**(3 l_p)``: a level-1 particle carries 8x a level-0 one."""
    pos = uniform_particles(4, 1.0)
    ps = make_set(pos, level=0)
    ps.update_mass_radius(r0=1.0, rho_l=1000.0, rho_g=1.0)
    m0 = ps.mass.numpy()[: ps.n].copy()
    r0 = ps.rad.numpy()[: ps.n].copy()
    ps.level.assign(np.ones(ps.capacity, np.int32))
    ps.update_mass_radius(r0=1.0, rho_l=1000.0, rho_g=1.0)
    assert np.allclose(ps.mass.numpy()[: ps.n], 8 * m0)
    assert np.allclose(ps.rad.numpy()[: ps.n], 2 * r0)


def test_coarsening_conserves_mass_in_expectation():
    """Deleting 7/8 and octupling the survivors' mass must preserve total mass."""
    pos = uniform_particles(28, 1.0)
    ps = make_set(pos, level=0)
    ps.update_mass_radius(1.0, 1000.0, 1.0)
    m_before = ps.mass.numpy()[: ps.n].sum()

    dist = wp.array(np.full((4, 4, 4), 1.5, np.float32), dtype=float, device=DEV)
    for step in range(3):
        ps.resample(dist, 0.25, beta_bw=1.0, max_level=3, seed=step)
    ps.update_mass_radius(1.0, 1000.0, 1.0)
    m_after = ps.mass.numpy()[: ps.n].sum()
    rel = abs(m_after - m_before) / m_before
    assert rel < 0.05, f"mass drifted {100*rel:.1f}% through one coarsening"


# ------------------------------------------------------------------ dual loop
def test_refinement_map_from_particles_is_an_atomic_min():
    """``M[i] = min(l_p)`` over the blocks each particle touches (paper 5.4.1)."""
    from pfflip.particles import atomic_min_refinement_map

    nb = (4, 4, 4)
    extent = 0.25
    pos = np.array([[0.1, 0.1, 0.1], [0.6, 0.6, 0.6]], np.float32)
    lev = np.array([0, 2], np.int32)
    px = wp.array(pos, dtype=wp.vec3, device=DEV)
    pl = wp.array(lev, dtype=wp.int32, device=DEV)
    m = wp.array(np.full(64, 9, np.int32), dtype=wp.int32, device=DEV)
    wp.launch(
        atomic_min_refinement_map,
        dim=2,
        inputs=[px, pl, m, *nb, extent, 0.0],
        device=DEV,
    )
    out = m.numpy().reshape(nb)
    assert out[0, 0, 0] == 0
    assert out[2, 2, 2] == 2
    assert out[3, 3, 3] == 9, "untouched blocks must stay unset"


def test_grid_follows_particles():
    """End-to-end: particle levels -> refinement map -> graded MSBG."""
    g = MSBG((8, 8, 8), n0=16, h0=1 / 128, device=DEV, max_level=3)
    extent = g.n0 * g.h0

    # a ball of fine particles in the middle, coarse elsewhere
    rng = np.random.default_rng(0)
    pos = rng.random((20000, 3)).astype(np.float32)
    r = np.linalg.norm(pos - 0.5, axis=1)
    lev = np.clip((r * 8).astype(np.int32), 0, 3)

    px = wp.array(pos, dtype=wp.vec3, device=DEV)
    pl = wp.array(lev, dtype=wp.int32, device=DEV)
    m = wp.array(np.full(g.nblk, 3, np.int32), dtype=wp.int32, device=DEV)
    from pfflip.particles import atomic_min_refinement_map

    wp.launch(
        atomic_min_refinement_map, dim=len(pos), inputs=[px, pl, m, *g.nb, extent, 0.0], device=DEV
    )
    g.build(m.numpy())

    assert 0 in g.level_histogram(), "no finest blocks were created"
    assert g.compression > 1.5, f"adaptive grid saved only {g.compression:.2f}x"
    # the finest blocks must sit near the middle
    lv = g.level_h.reshape(g.nb)
    fine = np.argwhere(lv == 0)
    assert np.abs(fine.mean(axis=0) - 3.5).max() < 1.5
