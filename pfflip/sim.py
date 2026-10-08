"""PF-FLIP simulator -- stage 1: uniform MAC grid, no adaptivity.

Implements the time step of paper section 3 (see docs/ALGORITHM.md appendix B
for the full adaptive ordering; the steps that belong to stages 2-3 are marked
TODO below):

 1. P2G: splat mass + momentum to MAC faces with eq. (6)          [3.2]
 2. phase field phi from the reused weight accumulators, eq. (7)  [3.3]
    -> face coefficients beta = 1/rho, eq. (9)
 3. external forces on u~*                                        [3.5]
 4. solve the variable-coefficient Poisson equation (8)           [6]
 5. project                                                       [3.5]
 6. G2P with per-phase alpha_FLIP, eqs. (11)-(12)                 [3.6]
 7. extrapolate liquid velocity into the air phase                [3.7]
 8. advect with RK3                                               [3.7]

TODO (stage 2): MSBG blocks, adaptive particles (section 4, 5).
TODO (stage 3): adaptive relaxation / active-block queue (section 6.4).
TODO: escaped-particle droplets and spray (sections 3.4, 7).
"""

from __future__ import annotations

import math
import time

import numpy as np
import warp as wp

from . import kernels as K
from .config import Config
from .poisson import MGPCG


class Simulator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dev = cfg.device
        nx, ny, nz = cfg.res
        self.res = (nx, ny, nz)

        z = lambda shape: wp.zeros(shape, dtype=float, device=self.dev)  # noqa: E731
        zi = lambda shape: wp.zeros(shape, dtype=wp.int32, device=self.dev)  # noqa: E731

        self.face_shapes = ((nx + 1, ny, nz), (nx, ny + 1, nz), (nx, ny, nz + 1))

        # face-sampled fields, one array per axis
        self.vel = [z(s) for s in self.face_shapes]          # u~* then u
        self.vel_star = [z(s) for s in self.face_shapes]     # saved before forces
        self.vel_liq = [z(s) for s in self.face_shapes]      # extrapolated field
        self.dvel = [z(s) for s in self.face_shapes]         # delta u for FLIP
        self.mass = [z(s) for s in self.face_shapes]
        self.mom = [z(s) for s in self.face_shapes]
        self.beta = [z(s) for s in self.face_shapes]
        self.phi_face = [z(s) for s in self.face_shapes]
        self.valid = [zi(s) for s in self.face_shapes]
        self._known = [zi(s) for s in self.face_shapes]
        self._known2 = [zi(s) for s in self.face_shapes]
        self._tmp_face = [z(s) for s in self.face_shapes]

        # cell-sampled fields
        self.cell_mass = z(self.res)
        self.cell_wsum = z(self.res)
        self.phi_cell = z(self.res)
        self.is_liquid = zi(self.res)
        self.pressure = z(self.res)
        self.rhs = z(self.res)

        self._scalar = wp.zeros(1, dtype=float, device=self.dev)

        self.solver = MGPCG(
            self.res,
            self.dev,
            omega=cfg.mg_omega,
            pre=cfg.mg_pre_sweeps,
            post=cfg.mg_post_sweeps,
            coarsest=cfg.mg_coarsest_sweeps,
            min_dim=cfg.mg_min_dim,
        )

        # particles (set by a scene via set_particles)
        self.px = None
        self.pv = None
        self.ptype = None
        self.pmass = None
        self.prad = None
        self.n_particles = 0

        # eq. (7) normalisation, calibrated from a perfect lattice
        self.rho_tilde_0 = self._calibrate_rho_tilde_0()
        self.rho_0l = self.rho_tilde_0 * cfg.rho_l
        self.rho_min = cfg.eta_phi * cfg.rho_g * self.rho_tilde_0

        self.time = 0.0
        self.step_index = 0
        self.stats: list[dict] = []

    # ------------------------------------------------------------------ setup
    def _calibrate_rho_tilde_0(self) -> float:
        """``rho~_0``: the kernel weight sum for perfectly uniform particles.

        Paper 3.3 defines it exactly this way.  Evaluating it numerically (on a
        2x2x2-per-cell lattice matching the seeding) is more robust than the
        closed form ``n_p * 64 pi r^3 / 315``, and automatically stays correct
        if ``ppc`` changes.
        """
        cfg = self.cfg
        dx, r = cfg.dx, cfg.r_p
        per_axis = round(cfg.ppc ** (1.0 / 3.0))
        step = dx / per_axis
        R = int(math.ceil(r / step)) + 1
        acc = 0.0
        for a in range(-R, R + 1):
            for b in range(-R, R + 1):
                for c in range(-R, R + 1):
                    # lattice sites offset by half a sub-cell from the sample point
                    d2 = ((a + 0.5) ** 2 + (b + 0.5) ** 2 + (c + 0.5) ** 2) * step * step
                    t = max(1.0 - d2 / (r * r), 0.0)
                    acc += t * t * t
        return acc

    def set_particles(self, pos: np.ndarray, vel: np.ndarray, ptype: np.ndarray):
        cfg = self.cfg
        n = len(pos)
        self.n_particles = n
        self.px = wp.array(pos.astype(np.float32), dtype=wp.vec3, device=self.dev)
        self.pv = wp.array(vel.astype(np.float32), dtype=wp.vec3, device=self.dev)
        self.ptype = wp.array(ptype.astype(np.int32), dtype=wp.int32, device=self.dev)
        # m_p = rho_phase * V_p with V_p = 1 (paper eq. 5); the physical volume
        # scale is absorbed into the eq. (7) normalisation rho~_0
        mass = np.where(ptype == 1, cfg.rho_l, cfg.rho_g).astype(np.float32)
        self.pmass = wp.array(mass, dtype=float, device=self.dev)
        self.prad = wp.array(np.full(n, cfg.r_p, np.float32), dtype=float, device=self.dev)

    # ------------------------------------------------------------------ step
    def dt_from_cfl(self) -> float:
        cfg = self.cfg
        self._scalar.zero_()
        wp.launch(K.particle_max_speed, dim=self.n_particles, inputs=[self.pv, self._scalar], device=self.dev)
        vmax = float(self._scalar.numpy()[0])
        g = float(np.linalg.norm(cfg.gravity))
        # include the velocity gravity will add this step
        vmax = max(vmax, math.sqrt(2.0 * g * cfg.dx) if g > 0 else 0.0, 1e-8)
        return float(np.clip(cfg.cfl * cfg.dx / vmax, cfg.dt_min, cfg.dt_max))

    def step(self, dt: float | None = None, verbose: bool = False) -> dict:
        cfg = self.cfg
        if dt is None:
            dt = self.dt_from_cfl()
        t0 = time.perf_counter()

        self._p2g()
        self._phase_field_and_coefficients()
        for ax in range(3):
            wp.copy(self.vel_star[ax], self.vel[ax])
        self._add_forces(dt)
        pcg_it, pcg_res = self._pressure_solve(dt, verbose=verbose)
        self._project(dt)
        div_after = self._max_divergence()
        self._g2p()
        self._extrapolate_liquid_velocity()
        self._advect(dt)

        wp.synchronize_device(self.dev)
        wall = time.perf_counter() - t0
        self.time += dt
        self.step_index += 1
        rec = {
            "step": self.step_index,
            "t": self.time,
            "dt": dt,
            "pcg_iter": pcg_it,
            "pcg_res": pcg_res,
            "max_div": div_after,
            "wall_s": wall,
        }
        self.stats.append(rec)
        return rec

    # ------------------------------------------------------------------ stages
    def _p2g(self):
        """Paper 3.2, eqs. (3)-(6)."""
        for ax in range(3):
            self.mass[ax].zero_()
            self.mom[ax].zero_()
        self.cell_mass.zero_()
        self.cell_wsum.zero_()

        n = self.n_particles
        for ax in range(3):
            wp.launch(
                K.p2g_face,
                dim=n,
                inputs=[self.px, self.pv, self.ptype, self.pmass, self.prad,
                        self.mass[ax], self.mom[ax], ax, self.cfg.dx],
                device=self.dev,
            )
        wp.launch(
            K.p2g_cell,
            dim=n,
            inputs=[self.px, self.pmass, self.prad, self.cell_mass, self.cell_wsum, self.cfg.dx],
            device=self.dev,
        )
        for ax in range(3):
            wp.launch(
                K.face_velocity_from_momentum,
                dim=self.face_shapes[ax],
                inputs=[self.mass[ax], self.mom[ax], self.vel[ax], self.valid[ax],
                        1e-12 * self.rho_0l],
                device=self.dev,
            )

    def _phase_field_and_coefficients(self):
        """Paper 3.3: phi from the reused weight accumulators, then beta."""
        cfg = self.cfg
        wp.launch(
            K.cell_phase,
            dim=self.res,
            inputs=[self.cell_mass, self.phi_cell, self.is_liquid,
                    self.rho_min, cfg.alpha_phi, self.rho_0l],
            device=self.dev,
        )
        for ax in range(3):
            wp.launch(
                K.face_beta,
                dim=self.face_shapes[ax],
                inputs=[self.mass[ax], self.is_liquid, self.phi_face[ax], self.beta[ax], ax,
                        self.rho_min, cfg.alpha_phi, self.rho_0l, cfg.rho_l, cfg.rho_g],
                device=self.dev,
            )
            wp.launch(
                K.zero_solid_faces,
                dim=self.face_shapes[ax],
                inputs=[self.beta[ax], self.vel[ax], ax],
                device=self.dev,
            )

    def _add_forces(self, dt: float):
        for ax in range(3):
            g = self.cfg.gravity[ax]
            if g != 0.0:
                wp.launch(K.add_gravity, dim=self.face_shapes[ax],
                          inputs=[self.vel[ax], g, dt], device=self.dev)
            wp.launch(K.zero_solid_faces, dim=self.face_shapes[ax],
                      inputs=[self.beta[ax], self.vel[ax], ax], device=self.dev)

    def _pressure_solve(self, dt: float, verbose: bool = False):
        cfg = self.cfg
        wp.launch(
            K.build_rhs,
            dim=self.res,
            inputs=[*self.vel, self.cell_wsum, self.rhs,
                    cfg.dx, dt, self.rho_tilde_0, cfg.k_div],
            device=self.dev,
        )
        self.solver.build_hierarchy(*self.beta)
        return self.solver.solve(
            self.rhs, self.pressure, tol=cfg.pcg_tol, max_iter=cfg.pcg_max_iter, verbose=verbose
        )

    def _project(self, dt: float):
        for ax in range(3):
            wp.launch(
                K.apply_pressure_gradient,
                dim=self.face_shapes[ax],
                inputs=[self.pressure, self.beta[ax], self.vel[ax], ax, self.cfg.dx, dt],
                device=self.dev,
            )

    def _max_divergence(self) -> float:
        self._scalar.zero_()
        wp.launch(K.divergence_norm, dim=self.res,
                  inputs=[*self.vel, self._scalar, self.cfg.dx], device=self.dev)
        return float(self._scalar.numpy()[0])

    def _g2p(self):
        """Paper 3.6, eqs. (11)-(12)."""
        for ax in range(3):
            wp.launch(K.field_difference, dim=self.face_shapes[ax],
                      inputs=[self.vel[ax], self.vel_star[ax], self.dvel[ax]], device=self.dev)
        wp.launch(
            K.g2p,
            dim=self.n_particles,
            inputs=[self.px, self.pv, self.ptype, *self.vel, *self.dvel,
                    self.cfg.dx, self.cfg.alpha_flip_liquid, self.cfg.alpha_flip_air],
            device=self.dev,
        )

    def _extrapolate_liquid_velocity(self):
        """Paper 3.7: liquid velocity is pushed a few cells into the air."""
        for ax in range(3):
            wp.launch(K.seed_liquid_velocity, dim=self.face_shapes[ax],
                      inputs=[self.vel[ax], self.is_liquid, self.vel_liq[ax], self._known[ax], ax],
                      device=self.dev)
        for _ in range(self.cfg.extrap_cells):
            for ax in range(3):
                wp.launch(K.extrapolate_sweep, dim=self.face_shapes[ax],
                          inputs=[self.vel_liq[ax], self._known[ax],
                                  self._tmp_face[ax], self._known2[ax]], device=self.dev)
                self.vel_liq[ax], self._tmp_face[ax] = self._tmp_face[ax], self.vel_liq[ax]
                self._known[ax], self._known2[ax] = self._known2[ax], self._known[ax]

    def _advect(self, dt: float):
        cfg = self.cfg
        eps = 1e-4 * cfg.dx
        lo = wp.vec3(eps, eps, eps)
        hi = wp.vec3(*[s - eps for s in cfg.domain_size])
        wp.launch(
            K.advect_rk3,
            dim=self.n_particles,
            inputs=[self.px, self.ptype, *self.vel, *self.vel_liq, dt, cfg.dx, lo, hi],
            device=self.dev,
        )
        wp.launch(K.reflect_at_walls, dim=self.n_particles,
                  inputs=[self.px, self.pv, lo, hi], device=self.dev)
