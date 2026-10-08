"""Adaptive PF-FLIP simulator on an MSBG (paper sections 4, 5).

Implements the dual particle-grid adaptivity loop of paper 5.4 verbatim:

 1. block refinement map ``M[i] = min(l_p, M[i])`` by atomic minimum over the
    blocks each particle's CFL-enlarged bounding box touches
 2. regularise ``M`` to 2:1 grading
 3. (re)allocate the MSBG and its data channels
 4. P2G on the multiresolution grid
 5. auxiliary fields: phase field, then the reduced-resolution ``d~``
 6. per-particle refinement / probabilistic coarsening
 7. one PF-FLIP step and advection with MSBG interpolation

Steps 4-7 fold in the section 3 pipeline (phase field, pressure projection,
G2P) at multiresolution.

The pressure solve here is Jacobi-CG (:mod:`pfflip.mr_solver`); the paper's
section 6 multiresolution multigrid is stage 3 and will drop in behind the same
interface.
"""

from __future__ import annotations

import math
import time

import numpy as np
import warp as wp

from . import mr_kernels as MK
from . import mr_operator as mro
from . import particles as P
from .config import Config
from .mr_solver import MRSolver
from .msbg import MSBG


class AdaptiveSimulator:
    def __init__(self, cfg: Config, nblocks, n0=16, max_level=3, beta_bw=1.0):
        self.cfg = cfg
        self.dev = cfg.device
        self.beta_bw = beta_bw

        # finest spacing h0 fixes the domain: nblocks * n0 cells per axis
        self.grid = MSBG(nblocks, n0=n0, h0=cfg.dx, device=self.dev, max_level=max_level)
        self.max_level = self.grid.max_level
        self.block_extent = n0 * cfg.dx
        self.domain = tuple(n * self.block_extent for n in self.grid.nb)

        self.particles = P.Particles(self.dev, capacity=1024)
        self._ncells = -1
        self._level_sig = None
        self._scalar = wp.zeros(1, dtype=float, device=self.dev)
        self._level_map = wp.zeros(self.grid.nblk, dtype=wp.int32, device=self.dev)
        self._block_phase = wp.zeros(self.grid.nblk * 2, dtype=wp.int32, device=self.dev)

        # eq. (7) normalisation, calibrated once at level 0 (see mr_kernels)
        self.rho_tilde_0 = self._calibrate()
        self.rho_0l = self.rho_tilde_0 * cfg.rho_l
        self.rho_min = cfg.eta_phi * cfg.rho_g * self.rho_tilde_0

        self.time = 0.0
        self.step_index = 0
        self.stats: list[dict] = []

    # ------------------------------------------------------------------ setup
    def _calibrate(self) -> float:
        cfg = self.cfg
        per_axis = round(cfg.ppc ** (1.0 / 3.0))
        step = cfg.dx / per_axis
        r = cfg.r_p
        R = int(math.ceil(r / step)) + 1
        acc = 0.0
        for a in range(-R, R + 1):
            for b in range(-R, R + 1):
                for c in range(-R, R + 1):
                    d2 = ((a + 0.5) ** 2 + (b + 0.5) ** 2 + (c + 0.5) ** 2) * step * step
                    t = max(1.0 - d2 / (r * r), 0.0)
                    acc += t * t * t
        return acc

    def set_particles(self, pos, vel, ptype, level=None):
        self.particles.set(pos, vel, ptype, level)

    # ------------------------------------------------------------------ channels
    def _ensure_channels(self):
        """Reallocate channels and the solver when the grid topology changes.

        Keyed on the *refinement map*, not on the cell count: the multigrid
        hierarchy is built from the per-block levels, so two different maps
        that happen to yield the same number of cells must not share a solver.
        """
        n = self.grid.ncells
        sig = hash(self.grid.level_h.tobytes())
        if sig == self._level_sig:
            return
        self._level_sig = sig
        realloc = n != self._ncells
        self._ncells = n
        if not realloc:
            # same size, different topology: buffers are fine, solver is not
            self.solver = MRSolver(self.grid, self.dev)
            return
        z = lambda m=1: wp.zeros(n * m, dtype=float, device=self.dev)  # noqa: E731
        self.fmass, self.fmom = z(3), z(3)
        self.fvel, self.fvel_star, self.fdvel = z(3), z(3), z(3)
        self.fbeta = z(3)
        self.cmass, self.cwsum, self.phi = z(), z(), z()
        self.pressure, self.rhs = z(), z()
        self.is_liquid = wp.zeros(n, dtype=wp.int32, device=self.dev)
        self.solver = MRSolver(self.grid, self.dev)

    # ------------------------------------------------------------------ step 1-3
    def _rebuild_grid(self, dt_hint: float):
        """Paper 5.4 steps 1-3: refinement map, grading, allocation."""
        # start from the coarsest level and pull down with an atomic minimum
        self._level_map.fill_(self.max_level)
        cfl_eps = max(self.cfg.cfl, 1.0) * self.cfg.dx
        wp.launch(
            P.atomic_min_refinement_map,
            dim=self.particles.n,
            inputs=[self.particles.x, self.particles.level, self._level_map,
                    *self.grid.nb, self.block_extent, cfl_eps],
            device=self.dev,
        )
        self.grid.build(self._level_map.numpy())  # build() applies 2:1 grading
        self._ensure_channels()

    # ------------------------------------------------------------------ step 5
    def _distance_field(self) -> wp.array:
        """Reduced-resolution auxiliary ``d~`` from the thresholded phase field.

        A block is "interface" when it contains both phases.  Distance is then
        measured in **block units**, which is exactly the normalisation the
        criterion ``l_new = floor(|beta_bw d~| / m_0)`` expects with
        ``m_0 = n_0`` (paper 5.2, 5.4 step 6).
        """
        self._block_phase.zero_()
        wp.launch(
            MK.interface_flag_per_block,
            dim=self.grid.ncells,
            inputs=[self.grid.g, self.is_liquid, self._block_phase],
            device=self.dev,
        )
        bp = self._block_phase.numpy().reshape(-1, 2)
        mixed = (bp[:, 0] > 0) & (bp[:, 1] > 0)
        d = P.block_distance_field(mixed.reshape(self.grid.nb))
        return wp.array(d, dtype=float, device=self.dev)

    # ------------------------------------------------------------------ main
    def dt_from_cfl(self) -> float:
        cfg = self.cfg
        self._scalar.zero_()
        wp.launch(MK.particle_max_speed, dim=self.particles.n,
                  inputs=[self.particles.v, self._scalar], device=self.dev)
        vmax = float(self._scalar.numpy()[0])
        g = float(np.linalg.norm(cfg.gravity))
        vmax = max(vmax, math.sqrt(2.0 * g * cfg.dx) if g > 0 else 0.0, 1e-8)
        return float(np.clip(cfg.cfl * cfg.dx / vmax, cfg.dt_min, cfg.dt_max))

    def step(self, dt: float | None = None, verbose=False) -> dict:
        cfg = self.cfg
        ps = self.particles
        if dt is None:
            dt = self.dt_from_cfl()
        t0 = time.perf_counter()

        # (1-3) dual adaptivity: particles drive the grid
        self._rebuild_grid(dt)
        ps.update_mass_radius(cfg.r_p, cfg.rho_l, cfg.rho_g)
        n = ps.n
        ncell = self.grid.ncells

        # (4) P2G
        for a in (self.fmass, self.fmom, self.cmass, self.cwsum):
            a.zero_()
        wp.launch(MK.p2g, dim=n,
                  inputs=[self.grid.g, ps.x, ps.v, ps.mass, ps.rad,
                          self.fmass, self.fmom, self.cmass, self.cwsum],
                  device=self.dev)
        wp.launch(MK.face_velocity, dim=ncell * 3,
                  inputs=[self.fmass, self.fmom, self.fvel, 1e-12 * self.rho_0l],
                  device=self.dev)

        # (5) phase field -> beta
        wp.launch(MK.cell_phase, dim=ncell,
                  inputs=[self.grid.g, self.cmass, self.phi, self.is_liquid,
                          self.rho_min, cfg.alpha_phi, self.rho_0l],
                  device=self.dev)
        wp.launch(MK.face_beta, dim=ncell,
                  inputs=[self.grid.g, self.fmass, self.is_liquid, self.fbeta,
                          self.rho_min, cfg.alpha_phi, self.rho_0l, cfg.rho_l, cfg.rho_g],
                  device=self.dev)
        wp.launch(MK.zero_wall_velocity, dim=ncell,
                  inputs=[self.grid.g, self.fbeta, self.fvel], device=self.dev)

        wp.copy(self.fvel_star, self.fvel)

        # forces
        wp.launch(MK.add_gravity, dim=ncell,
                  inputs=[self.grid.g, self.fvel, self.fbeta, *cfg.gravity, dt],
                  device=self.dev)

        # pressure projection
        wp.launch(MK.build_rhs, dim=ncell,
                  inputs=[self.grid.g, self.fvel, self.cwsum, self.rhs,
                          dt, self.rho_tilde_0, cfg.k_div],
                  device=self.dev)
        it, rel = self.solver.solve(self.fbeta, self.rhs, self.pressure,
                                    tol=cfg.pcg_tol, max_iter=cfg.pcg_max_iter)
        wp.launch(MK.project, dim=ncell,
                  inputs=[self.grid.g, self.pressure, self.fbeta, self.fvel, dt],
                  device=self.dev)

        self._scalar.zero_()
        wp.launch(MK.max_divergence, dim=ncell,
                  inputs=[self.grid.g, self.fvel, self._scalar], device=self.dev)
        div = float(self._scalar.numpy()[0])

        # G2P with the per-particle alpha_FLIP of paper 5.3
        wp.launch(MK.field_difference, dim=ncell * 3,
                  inputs=[self.fvel, self.fvel_star, self.fdvel], device=self.dev)
        nu_l = (1.0 - cfg.alpha_flip_liquid) * cfg.dx**2 / (6.0 * dt)
        nu_g = (1.0 - cfg.alpha_flip_air) * cfg.dx**2 / (6.0 * dt)
        wp.launch(P.per_particle_alpha_flip, dim=n,
                  inputs=[self.grid.g, ps.x, ps.type, ps.alpha, nu_l, nu_g, dt, 0.999],
                  device=self.dev)
        wp.launch(MK.g2p, dim=n,
                  inputs=[self.grid.g, ps.x, ps.v, ps.alpha, self.fvel, self.fdvel],
                  device=self.dev)

        # (6) per-particle refinement / probabilistic coarsening
        dist = self._distance_field()
        n_before, n_after = ps.resample(
            dist, self.block_extent, beta_bw=self.beta_bw,
            max_level=self.max_level, r0=cfg.r_p, seed=self.step_index,
        )

        # (7) advect
        eps = 1e-4 * cfg.dx
        lo = wp.vec3(eps, eps, eps)
        hi = wp.vec3(*[s - eps for s in self.domain])
        wp.launch(MK.advect_rk3, dim=ps.n,
                  inputs=[self.grid.g, ps.x, self.fvel, dt, lo, hi], device=self.dev)
        wp.launch(MK.reflect_at_walls, dim=ps.n,
                  inputs=[ps.x, ps.v, lo, hi], device=self.dev)

        wp.synchronize_device(self.dev)
        self.time += dt
        self.step_index += 1
        rec = {
            "step": self.step_index, "t": self.time, "dt": dt,
            "cells": ncell, "compression": self.grid.compression,
            "particles": ps.n, "dn": n_after - n_before,
            "cg_iter": it, "cg_res": rel, "max_div": div,
            "levels": self.grid.level_histogram(),
            "wall_s": time.perf_counter() - t0,
        }
        self.stats.append(rec)
        return rec
