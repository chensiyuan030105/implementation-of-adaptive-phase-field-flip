"""Configuration for the PF-FLIP solver.

Parameter names and default values follow Braun, Bender & Thuerey,
"Adaptive Phase-Field-FLIP for Very Large Scale Two-Phase Fluid Simulation",
ACM TOG 44(4), 2025.  See docs/ALGORITHM.md appendix A for provenance of each
value; the paper section is cited in the comment next to each field.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- domain / discretisation -------------------------------------------
    res: tuple[int, int, int] = (64, 64, 64)
    """Cell counts of the (uniform, stage-1) MAC grid."""

    dx: float = 1.0
    """Grid spacing.  Interface thickness eps_phi == r_0 == dx (paper 3.3)."""

    ppc: int = 8
    """Particles per cell at seeding (2x2x2 jittered lattice)."""

    # ---- physics -----------------------------------------------------------
    rho_l: float = 1000.0
    """Liquid density."""

    density_ratio: float = 1000.0
    """rho_l / rho_g.  Paper uses 1000:1 (water/air)."""

    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)

    # ---- phase field (paper eq. 7, section 3.3) ----------------------------
    alpha_phi: float = 1.0
    """Noisiness / stiffness trade-off knob of eq. (7).  Default 1; 0.5 for
    the Rayleigh-Taylor test (paper 8.2)."""

    # ---- FLIP blending (paper 3.6, 5.3, 8.4) -------------------------------
    alpha_flip_liquid: float = 0.985
    alpha_flip_air: float = 0.96
    """Per-phase FLIP/PIC blend.  Eq. (13) ties these to the numerical
    viscosity nu = (1 - alpha) dx^2 / (6 dt)."""

    # ---- time stepping ----------------------------------------------------
    cfl: float = 1.0
    """Paper uses 3-4 in production with RK3 + local substepping; stage 1
    defaults to 1.0 for validation."""

    dt_max: float = 1.0 / 24.0
    dt_min: float = 1e-6

    # ---- divergence / volume correction (paper eq. 8, c_div) ---------------
    k_div: float = 0.0
    """Strength of the particle-density divergence correction c_div.  0
    disables it (default for stage-1 validation)."""

    # ---- pressure solve (paper section 6) ---------------------------------
    pcg_tol: float = 1e-4
    """Paper 8.3: 1e-4 for all production examples."""

    pcg_max_iter: int = 200
    mg_pre_sweeps: int = 3
    mg_post_sweeps: int = 3
    """3 measured faster than 2 end-to-end: the extra sweep cuts CG iterations
    by ~1/3 for less than its own cost.  See docs/FINDINGS.md."""
    mg_coarsest_sweeps: int = 60
    mg_omega: float = 6.0 / 7.0
    """Damping weight of the hybrid Gauss-Seidel smoother; 6/7 is the 3D
    optimum quoted in paper 6.3."""

    mg_min_dim: int = 4
    """Stop coarsening when any axis would drop below this."""

    # ---- velocity extrapolation (paper 3.7) -------------------------------
    extrap_cells: int = 3
    """Liquid velocity is extrapolated 'a few cells' into the air phase so
    that liquid particles advect with liquid velocities."""

    # ---- escaped particles / spray (paper 3.4, 7) --------------------------
    droplets: bool = True
    """Convert escaped particles into Lagrangian droplets."""

    delta_phi: float = 0.2
    """Escaped-particle threshold offset (paper 3.4): a liquid particle where
    phi < 0.5 - delta_phi becomes a droplet, an air particle where
    phi > 0.5 + delta_phi is a bubble and is deleted."""

    droplet_radius_mean: float = 0.25
    droplet_radius_spread: float = 0.5
    """Log-normal droplet radii on emission (paper 7, Babinsky & Sojka 2002),
    in units of dx."""

    droplet_drag_cd: float = 0.5
    """Sphere drag coefficient; a = 3 rho_g Cd |v_rel| v_rel / (8 rho_l r)."""

    evaporation_k: float = 0.0
    """Rate in dr/dt = -k/r (paper 7, Sirignano 2010).  0 disables it."""

    spray_shear_threshold: float = 0.0
    """Emit extra spray where |grad u| exceeds this (paper 7).  0 disables;
    a sensible value is a few times the mean interfacial shear."""

    spray_emission_rate: float = 0.02
    spray_resolution_factor: int = 2
    """Spray density field resolution relative to the pressure grid; the paper
    uses 2-4x at 16-bit float."""

    droplet_max_substeps: int = 8

    # ---- runtime -----------------------------------------------------------
    device: str = "cuda:0"
    seed: int = 42

    # ------------------------------------------------------------------ derived
    @property
    def rho_g(self) -> float:
        return self.rho_l / self.density_ratio

    @property
    def r_p(self) -> float:
        """Base particle radius r_0 = dx (paper 3.3)."""
        return self.dx

    @property
    def eta_phi(self) -> float:
        """Density scaling factor eta_phi = log(rho_l / rho_g) (paper eq. 7)."""
        import math

        return math.log(self.density_ratio)

    @property
    def domain_size(self) -> tuple[float, float, float]:
        return (self.res[0] * self.dx, self.res[1] * self.dx, self.res[2] * self.dx)
