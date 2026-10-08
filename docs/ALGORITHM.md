# Adaptive Phase-Field-FLIP for Very Large Scale Two-Phase Fluid Simulation — Implementation Spec (Sections 4–10)

Braun, Bender, Thuerey. ACM Trans. Graph. 44(4), SIGGRAPH 2025.

This document covers **pages 9–23** of the paper: Section 4 (Adaptivity / MSBG),
Section 5 (Adaptive Particles & Dual Adaptivity), Section 6 (Adaptive Poisson
Solver), Section 7 (Spray and Rendering), Section 8 (Results), Section 9
(Limitations), Section 10 (Conclusions). Sections 1–3 (PF-FLIP core) are covered
elsewhere; equations (6)–(13) from those sections are restated below only where
Sections 4–8 depend on them.

---

## 0. Carry-over definitions from Sections 3.x referenced later

These are needed because Sections 4–6 refer to them by number.

**Eq. (6) — P2G splatting kernel** (higher-order polynomial, SPH-style, cubic in
the squared distance, spherical support, no sqrt):

$$w_{ai,p} = \left(\max\!\left(1 - \left(\|\mathbf{x}_i + \mathbf{e}_a/2 - \mathbf{x}_p\|/r_p\right)^2,\; 0\right)\right)^3 \tag{6}$$

where $\mathbf{x}_i$ is the center of cell $i$, $\mathbf{e}_a$ the face-normal
unit vector of axis $a$, and $r_p$ the **particle radius**. Face velocity
$\tilde{\mathbf{u}}^*_{ai} = \mathbf{P}_{ai}/m_{ai}$.

**Eq. (7) — phase field from raw splatted mass density** $\tilde\rho$:

$$\phi(\tilde\rho) = \begin{cases} 0 & \text{if } \tilde\rho < \tilde\rho_{min},\\[4pt] \min\!\left(\sqrt{\dfrac{\max(\tilde\rho - \tilde\rho_{min},\,0)}{\alpha_\phi\,\tilde\rho_{0l}}},\; 1\right) & \text{else.}\end{cases} \tag{7}$$

with $\tilde\rho_{min} = \eta_\phi \rho_g \tilde\rho_0$,
$\eta_\phi = \log(\rho_l/\rho_g)$, and $\tilde\rho_0$ the density obtained by
applying the weighting kernel to perfectly uniformly distributed particles.
Interface thickness $\epsilon_\phi = r_0 = \Delta x$ (one grid cell).

**Eq. (8) — variable-coefficient pressure Poisson equation:**

$$\Delta t\,\nabla\cdot\big(\beta(\mathbf{x})\nabla p(\mathbf{x})\big) = \nabla\cdot\mathbf{u}^*(\mathbf{x}) + c_{div}(\mathbf{x}) \tag{8}$$

**Eq. (9) — face coefficients:**

$$\beta(\mathbf{x}) = 1/\rho(\mathbf{x}) \tag{9}$$

**Eq. (10) — 1D discretization on the MAC grid:**

$$\beta_{i+\frac12}\left(\frac{p_{i+1}-p_i}{\Delta x}\right) - \beta_{i-\frac12}\left(\frac{p_i - p_{i-1}}{\Delta x}\right) = \frac{1}{\Delta t}\left(u^*_{i+\frac12} - u^*_{i-\frac12}\right) \tag{10}$$

The Laplacian matrix $\mathcal{L}$ satisfies $\mathcal{L}p = \frac{1}{\Delta t}\mathcal{D}(u^*)$,
is symmetric positive definite, and uses a **7-point stencil** (4 distinct
coefficients per cell in the symmetric storage: 3 face coefficients + diagonal).

**Eq. (11)/(12) — G2P:** $\Delta\mathbf{u} = \mathbf{u} - \tilde{\mathbf{u}}^*$,
$\mathbf{u}_p^{new} = \alpha_{FLIP}(\mathbf{u}_p^{old} + I(\Delta\mathbf{u},\mathbf{x}_p)) + (1-\alpha_{FLIP})I(\mathbf{u},\mathbf{x}_p)$.

**Eq. (13) — numerical viscosity induced by $\alpha_{FLIP}$:**

$$\nu = (1-\alpha_{FLIP})\frac{\Delta x^2}{6\Delta t} \tag{13}$$

This is the hinge for Section 5.3 (equalizing spatially varying numerical
viscosity across resolution levels). Advection: **third-order Runge–Kutta** with
locally adaptive time stepping (Bridson 2015). Liquid velocity is **extrapolated
into the air phase by a few cells** so that liquid particles are advected with
corresponding velocities.

---

## 4. Adaptivity

Motivation: 3D fluid simulation cost scales as $O(n^4)$ in grid resolution $n$;
multiphase flow worsens scaling via the pressure solve.

### 4.1 Block Size Does Matter

Key claim: use **significantly larger blocks than prior work** — $n = 16$ or
$n = 32$ grid cells per block edge (vs. VDB $n=8$, SPGrid/block-octree
$n=4\ldots8$). Four arguments, with the overhead curves of Figure 8:

1. **Unused-cell overhead (smooth, 2D-manifold interfaces)** — curve (a) in
   Fig. 8:
   $$1 - O(n^2)/n^3$$
   which approaches 100% rapidly as $n$ grows. This is the argument *against*
   large blocks and the one prior work optimizes.

2. **Fractal interfaces** — curve (b). Real fluid interfaces are fractals of
   dimension $\approx 2.5$ (Sreenivasan 1991), not smooth 2D manifolds. The
   empty-cell overhead is then
   $$1 - O(n^{2.5})/n^3$$
   which is **significantly lower** than for 2D interfaces.

3. **Block boundary overhead** — curve (c). The fraction of boundary cells per
   total cells is
   $$\frac{n^3 - (n-2)^3}{n^3} = 1 - \left(1 - \frac{2}{n}\right)^3$$
   which decreases rapidly with increasing block size. This matters because
   stencil-based operations (multigrid smoothing) dominate simulation time and
   require costly operations at block boundaries.

4. **Hardware efficiency.** Larger CPU caches, wider SIMD lanes, larger virtual
   memory pages favor large blocks. $16^3$ or $32^3$ blocks now fit easily in L2
   or L3 cache. (Fig. 8 marks the ~L2 cache limit near block size 16 and the ~L3
   cache limit near block size 32; SPGrid and VDB are marked near block size 4–8;
   MSBG-16 and MSBG-32 near 16 and 32.)

5. **Reduced block count** — fewer blocks for a given domain size.

6. **Dense linear block index becomes feasible** — curve (d). Storage overhead
   of a dense index for a sparsely populated domain with **1% occupancy** and a
   single **4-byte data channel** is
   $$\frac{1}{0.01\,n^3}$$
   For $n = 16$ the dense index costs only **2% of total memory** (and the
   relative overhead shrinks further with each additional data channel).
   Dense linear indices are attractive for unmatched simplicity and hardware
   friendliness compared to tree-based alternatives.

**Why not trees:** tree depth affects random access times; parallelization of
tree operations is challenging; stencil-based operations are complicated by the
non-uniform block neighborhood structure.

### 4.2 MSBG — Multiresolution Sparse Block Grids

**Data layout.**

- A **dense linear array** of **pointers to blocks** serves as the global block
  index ("index grid"), covering the computational domain at block granularity.
  Dense indexing gives $O(1)$ random access.
- The **geometric size of every block is constant** (fixed physical extent).
  Only the **resolution** of each block varies. This is the inverse of
  conventional tree-based schemes, which adapt block *size* while keeping
  resolution constant.
- Terminology: **(geometric) block size** is chosen once per simulation;
  **block resolution** is spatially adapted during the course of the simulation.
  Finely resolved blocks are called **large**; coarse blocks are **small**.
- The resolution of each block is driven by a **block refinement map** $M$
  (one integer refinement level per block), computed from an application-specific
  refinement criterion (Section 5.4).
- **Base block resolution** $n_0$ = cells per block edge at refinement level 0.
  Default $n_0 = 16$ unless noted (also $n_0 = 32$ used, "MSBG-32"). A block at
  refinement level $l$ has edge resolution
  $$n = n_0\,2^{-l}$$
  i.e. **level 0 is the finest**, and increasing $l$ means coarser.
- Grid spacing at level $l$: $h_l = h_0 2^{l}$ (cells get larger with $l$).

**Multiresolution level limit.** Because the blocks have fixed geometric size,
the number of refinement levels is bounded by
$$L_{max} = \log_2(n_0) \quad (\text{i.e. } \log_2 16 = 4 \text{ or } \log_2 32 = 5)$$
In practice this is not a limitation: in 3D computational cost drops by a
**factor of eight** per coarser level. **All results in this study use a maximum
of three or four refinement levels.**

**Grading.** The refinement map is regularized to enforce a **2:1 grading
constraint**: resolution levels of neighboring blocks differ by **no more than
one**.

**Block boundary overhead at coarse levels** (curve (c), Fig. 8) increases at
coarser MSBG levels, but this is offset by the fact that most grid cells are
still contained in blocks at the finest resolution levels (factor-of-eight
reduction in active cells per coarser level).

**Discretization of differential operators** (gradients, divergence) and the
treatment of solid obstacles follows Losasso et al. 2004 and Weber et al. 2015,
straightforwardly adapted to MSBG.

**Coarse–fine pressure gradient (Fig. 9c).** At a coarse-fine transition the
pressure gradient between a coarse cell with pressure $p$ and a fine neighbor
with pressure $p_i$, across distance $h$, is discretized as
$$\frac{p_i - p}{h\,\Delta_g}, \qquad \Delta_g = \frac{3}{4}\ \ \text{(Losasso et al. 2004)}$$

**Stencil access across resolution boundaries.** Because block geometric size is
constant and grading is 2:1, every block has exactly **6 face neighbors**
(uniform access to 6 neighbor blocks), independent of resolution — in contrast
to block-octrees where a block can have **up to 24 neighbor blocks**. A
coarse-cell face at a coarse–fine boundary is backed by a **$2\times 2$ patch of
fine cell faces** in the neighbor block; the coarse face coefficient is the
average of the four fine face coefficients times a transition scaling factor
(Section 6.1, Algorithm 1).

### 4.3 Capabilities of MSBG

**Table 1. Comparison between MSBG and Block-Octree**

| | MSBG | Block-Octree |
|---|---|---|
| Construction, e.g. from particles | $O(n)$, easy parallelization | $O(n\log n)$, non-trivial parallelization |
| Random access | $O(1)$ | $O(\log n)$ |
| Stencil access | Uniform access to **6** neighbor blocks | Non-uniform access to variable number of **up to 24** neighbor blocks |
| Grading | Fast | Parallelization challenging |
| Adaptivity | Only restricted by block raster | Less flexible because "terminal siblings" have same resolution |
| Supports multi-color techniques | **Yes** | **No** |
| Code complexity | Low | High |
| Refinement depth | $\log_2(\text{base-resolution})$ | Unlimited |

**Multiresolution multi-color schemes.** Uniformly constant block *size* enables
block-based red-black multigrid smoothers and **lock-free 8-color particle
splatting**. Octrees cannot support this due to nonuniform block sizes.
Specifically, the authors employ a recently published **eight-color splatting
scheme** (Fang et al. 2018) for lock-free splatting in the P2G step, enabling
maximum parallelism of *multiresolution* particle–grid transfer. The splatting
weights for **all faces of a MAC grid cell are computed in parallel using
SIMD/AVX-vector instructions**. In practice: **P2G transfer rates of several
hundred million particles per second.**

**Spatiotemporally coherent memory access.** Active blocks are conceptually
ordered along a **space-filling (Morton) curve**. Unlike SPGrid (which integrates
Morton coding into cell address calculations, requiring expensive bit
interleaving), MSBG retains **simple linear indexing within blocks** and sorts
the **active block indices** by their Morton codes before traversal. The Morton
code is computed **once per block**, exploiting the extra indirection of the
dense block-pointer array. Block-based sorting (and other global-topology
operations) is extremely fast in MSBG because there are relatively few but large
blocks.

**Dynamic task-based scheduling.** Naive parallelization over MSBG blocks causes
load imbalance because the number of grid cells per block differs by resolution.
Remedy: **dynamic task-based scheduling** (Robison et al. 2008) as provided by
**TBB** (Pheatt 2008).

---

## 5. Adaptive Particles and Dual Adaptivity

Particle-based adaptivity must accompany grid adaptivity. Adaptivity is achieved
through **particle splitting** (coarse-to-fine) and **merging** (fine-to-coarse),
following Adams et al. 2007 for splitting; merging is replaced by a novel
**probabilistic coarsening** scheme.

**Particle level.** Each particle $p$ carries a **particle level** $l_p$
determining its radius:
$$r_p = r_0\,2^{l_p}$$
with base radius $r_0 = \Delta x$ (finest grid spacing). The particle level
$l_p$ is **orthogonal to the grid level** $l_i = M[b_i]$ of a given grid cell
within block $b_i$; $M$ is the MSBG block refinement map.

Particle mass/volume follow the radius: a level-$l_p$ particle represents
$2^{3 l_p}$ times the volume of a base particle; splitting one particle of level
$l_p$ produces **8 particles of level $l_p - 1$**.

### 5.1 Probabilistic Coarsening

Motivation: existing adaptive particle methods handle fine-to-coarse transitions
by **merging** several smaller particles into larger ones, which requires a
particle neighborhood search. PF-FLIP aims to retain FLIP as a
**neighborhood-search-free** method.

Observation: coarsening a uniform particle distribution by a **factor of two in
each spatial dimension** results in a uniform distribution with particle density
**eight times lower**, which can be obtained by **randomly removing 7/8 of the
particles**.

**Deletion probability.** Once a particle is marked for coarsening, remove it
with probability
$$prob_{del} = 1 - \frac{1}{\left(2^{\,l_{new} - l_{old}}\right)^3}$$

(For a single-level coarsening $l_{new} - l_{old} = 1$ this gives
$prob_{del} = 1 - 1/8 = 7/8$.) The surviving particles are promoted to level
$l_{new}$ (radius/mass/volume updated accordingly).

**Hysteresis.** Oscillations near the coarse-to-fine transition zone are
prevented by a simple hysteresis scheme using a **deferred-coarsening counter**
$D_p$ per particle: coarsening is only applied once $D_p \ge 3$ (see step (6) of
Section 5.4).

### 5.2 Refinement Criteria

**Default criterion: distance to the liquid–air interface.** PF-FLIP maintains no
distance field. Instead a **reduced-resolution auxiliary SDF $\tilde d(\mathbf{x})$**
is generated from the **downsampled and thresholded phase field** using a simple,
**first-order accurate marching algorithm**. This is sufficient because the
distance field is used **solely for particle refinement**.

**Optional criterion:** magnitude of the **curl of the velocity** (Popinet 2003),
to capture flow details away from the surface, particularly in the air phase
where turbulence is typically more intense than in the bulk liquid (Fig. 12, left
column). **For all examples in this study, distance-based refinement alone was
sufficient.**

### 5.3 Equalizing Spatially Varying Numerical Viscosity

With varying $\Delta x$, numerical/artificial viscosity varies unphysically in
space — an issue often overlooked in the AMR literature (exception: Ando et al.
2013). PF-FLIP exploits the **controllable numerical viscosity of Eq. (13)** by
adjusting $\alpha_{FLIP}$ per particle:
$$\nu = (1-\alpha_{FLIP})\frac{\Delta x^2}{6\Delta t} \;\Rightarrow\; \alpha_{FLIP} = 1 - \frac{6\,\nu\,\Delta t}{\Delta x^2}$$
In the two-phase framework, **both phase-specific physical viscosity and
numerical viscosity variations from adaptive grid spacing** are accounted for.

**Crucial difference from Ando et al. 2013:** corrections are based on **local
grid spacing**, *not* on the particle radius / cell size coupling. This decouples
particle radius from cell size, which is essential for **large time steps with
high CFL numbers**, where particles may cross grid resolution boundaries within
one step.

### 5.4 Dual Particle-Grid Adaptivity — The Coupled Algorithm

Per time step:

1. **Compute the block refinement map $M$.** For each particle $p$, determine all
   blocks $B_i$ that intersect the particle's bounding box (**enlarged by a
   CFL-dependent epsilon**) and set
   $$M[i] \leftarrow \min(l_p,\; M[i])$$
   using a **fast atomic minimum operation**.

2. **Regularize the block refinement map** to enforce the **2:1 grading
   constraint**, ensuring the resolution levels of neighboring blocks differ by
   no more than one.

3. **Initialize the MSBG grid** with the new refinement map and **allocate grid
   data channels**.

4. **Transfer data from the particles to the MSBG grid (P2G):**
   - For each particle $p$, determine its containing grid block $b_i$, grid level
     $l = M[b_i]$, and cell-within-block $c_{ik}$.
   - *Using the linear, dense global MSBG block index, this computation requires
     only shift and logical-and operations on the particle coordinates.*
   - Loop over all neighboring grid blocks $b_j$ and cells $c_{jl}$ within the
     particle's radius $r_p = r_0 2^{l_p}$. This corresponds to the usual
     **27-cell neighborhood** when the particle level matches the grid level,
     i.e. $l_p = l$. **Also allow $l_p = l + 1$** for a smooth transition of
     coarse particles into regions where the grid is finer by one level.
   - At each touched cell, splat the particle's **weighted mass and velocity**
     contributions to the **staggered cell locations** (Eq. 6).

5. **On the MSBG grid, compute auxiliary fields**, such as $\tilde d$, required
   for evaluating the particle refinement criterion.

6. **Per-particle refinement / coarsening.** For each particle $p$, use the
   refinement criterion interpolated at its position $\mathbf{x}_p$ to determine
   the new particle level, e.g.
   $$l_{new} = \left\lfloor \left| \beta_{bw}\,\tilde d(\mathbf{x}_p) \right| / m_0 \right\rfloor$$
   where $m_0$ is the **MSBG base block resolution** ($n_0$) and $\beta_{bw}$
   **controls the width of the refinement bands**.
   - **If $l_{new} < l_p$ (refine):** split the particle into **8 smaller
     particles $p'$**, **randomly jittered within the octants of the grid cell
     containing $p$**, and set $l_{p'} \leftarrow l_{new}$.
   - **If $l_{new} > l_p$ (coarsen):** increment the **deferred-coarsening
     counter** $D_p$ and mark $p$ for coarsening if $D_p \ge 3$.
   - **Reset $D_p = 0$ if $l_{new} \le l_p$.**
   - **If $p$ is marked for coarsening**, remove the particle with probability
     $$prob_{del} = 1 - 1/\left(2^{\,l_{new}-l_p}\right)^3$$

7. **Advance the PF-FLIP simulation one time step** and **advect the particles**
   to their new positions using MSBG for adaptive-grid velocity interpolation.

**Fig. 9 notes.** (a) Particle density and size adapt with distance to the fluid
interface; the figure shows grid spacings labelled $1/512$, $1/64$, $1/128$ (and
$1/512$ at the interface) across the air→water transition, i.e. finest cells hug
the interface. (b) Velocity field as sampled by particles, with grid cell size
adapted to particle size. (c) Pressure gradient discretization at a coarse-fine
transition: $\frac{p_i - p}{h\Delta_g}$; thin lines = cell boundaries, bold lines
= MSBG block boundaries.

---

## 6. Adaptive Poisson Solver

Problem: $\mathcal{L}x = b$ (Eq. 8) is large and highly ill-conditioned for
high-density-contrast multiphase flow. The multiresolution discretization gives
$\mathcal{L}$ favorable properties (symmetry, simple 7-point stencil), but the
nearly discontinuous coefficient structure deteriorates convergence even for
multigrid-preconditioned CG (MacLachlan et al. 2008). The proposed solver is a
**fully adaptive variant of multigrid-preconditioned conjugate gradient**.

Contributions:
- Multiresolution Galerkin-multigrid
- Adaptive relaxation with improved residual metric and adaptive error threshold
- Adaptive multigrid as a **preconditioner** for CG
- A memory-efficient parallel multiresolution smoother

### 6.1 Algebraic Aggregation Multigrid

Built on Shao et al. 2022 (cell-centered **unsmoothed algebraic aggregation**
Galerkin coarsening for fixed-resolution single-phase fluids), extended to
multiresolution adaptive grids with a novel adaptive smoother, and integrated
into a specialized conjugate gradient algorithm.

**Galerkin coarsening (Eq. 14).** Grid operators at the coarse level are computed
algebraically from the fine ones:
$$\mathcal{L}^{2h} = R^{2h}_{h}\,\mathcal{L}^{h}\,P^{h}_{2h} \tag{14}$$
where $h$ is the spacing of the finer grid, $P$ is **Prolongation** and
$R = P^{T}$ is **Restriction** (Trottenberg et al. 2000). With unsmoothed
aggregation, $P$ is piecewise-constant injection (a coarse cell aggregates a
$2\times2\times2$ block of fine cells) and $R = P^T$ is summation; hence the
Galerkin coarse face coefficient reduces to a **sum over the $2\times2$ fine
faces of the coarse cell face**, which is exactly what Algorithm 1 computes
(sum over the 4 fine faces, divided by 4, times a resolution-transition scaling
factor $s$).

**Per-cell stencil.** For each cell $j$ of block $i$ at multigrid level $L>0$,
**three face coefficients** $\mathcal{F}[i,L][j,a]$ of the symmetric 7-point
stencil are computed, $a \in \{1,2,3\}$ denoting face normal directions
$\mathbf{e}_a$. Face coefficients at the finest level come directly from the
phase field via Eq. (9). **The diagonal elements of the (implicitly) assembled
matrices are the sum of the off-diagonal (face) coefficients.**

**Resolution transition scaling constants:**
$$\sigma_{CF} = \frac{1}{4\Delta_g}, \qquad \sigma_{FC} = 2\,\sigma_{CF}, \qquad \Delta_g = \tfrac34$$
i.e. numerically $\sigma_{CF} = 1/3$ and $\sigma_{FC} = 2/3$. These follow from
the symmetry-preserving multiresolution discretization of the pressure equation
(Section 4.2, Fig. 9c) and the Galerkin coarsening principle Eq. (14).

#### Algorithm 1: Parallel Multiresolution Galerkin Multigrid Coarsening

```
Input : Face coefficients F at finest level L = 0
Output: Face coefficients for all coarse levels

for L <- 1 to M-1 do                                   # Multigrid levels
  for i <- 1 to N do in parallel                       # MSBG Blocks
    l <- level(i, L)                   # Actual resolution level at this multigrid level
    if l = level(i, L-1) then
      F[i, L] <- F[i, L-1]             # No finer child block, simply copy block by reference
      continue

    for a <- 1 to 3 do                                 # Face normal directions (x,y,z)
      # Determine scaling factor for resolution transition
      switch  l - level( neighborBlock(i, a), L-1 )  do
        case  0 :  s <- 2^l                            # Fine-fine
        case  1 :  s <- 2^l * sigma_CF                 # Coarse-fine
        case -1 :  s <- 2^l * sigma_FC                 # Fine-coarse

      n <- n0 * 2^(-l)                                 # Block resolution

      for j <- 1 to n^3 do                             # Loop over all cells in block
        sum <- 0
        # Loop over 2x2 fine cells of coarse cell face in direction a
        for k <- (1,1) to (2,2) do
          j' <- cellIndexInFineBlock(j, k, a)
          sum <- sum + F[i, L-1][j', a]
        F[i, L][j, a] <- s * sum / 4
```

### 6.2 Leveraging MSBG

**Two independent hierarchies** (Fig. 10):
- the **MSBG hierarchy**, defined by the **resolution level** $l$;
- the **multigrid hierarchy**, defined by the **multigrid level** $L$.

MSBG maintains multigrid levels *independently* from MSBG resolution levels. For
the coarsest multigrid levels — those coarser than MSBG's block resolution — the
solver uses **additional, uniform grids** (no MSBG blocks).

Distinctions from the multiresolution pyramids of Setaluri et al. 2014:
1. **MSBG does not require maintaining an additional octree**, as in SPGrid-based
   solvers.
2. The inherent multiresolution structure of MSBG **facilitates multigrid
   relaxation across resolution boundaries**, eliminating the need for additional
   inter-level transfer passes to accumulate and propagate ghost cells.

Fig. 10 (1D illustration) shows MSBG levels 0,1,2 across Block 0/1/2 mapping onto
Multigrid levels 0,1,2, with prolongation and restriction stencils spanning the
resolution transitions directly.

### 6.3 Block-based Relaxation — Two-Stage Red-Black Hybrid Gauss-Seidel

Standard practice: a fixed number (1–3) of weighted Jacobi or Gauss-Seidel
relaxation steps before residual restriction (pre-smoothing) and after error
prolongation (post-smoothing). Jacobi is parallel-friendly but converges slower
and needs a temporary destination buffer; Gauss-Seidel converges faster because
updates are immediately visible to neighbors.

**Proposed smoother: a variant of hybrid Gauss-Seidel (Baker et al. 2011) with
two red-black stages:**

- **Outer (block) stage:** red-black ordered, parallel half-passes of **Jacobi**
  relaxations **at the block level** (blocks colored by
  $(x_b + y_b + z_b) \bmod 2$).
- **Inner (cell) stage:** **Gauss-Seidel (in-place)** iterations for the unknowns
  *within* blocks, likewise with a **red-black ordering** (cell color
  $= \text{color}(j)$).
- In-place updates are computed **at the level of entire SIMD-words**, i.e.
  **8 floats for the Intel AVX CPU instruction set**.

Benefits:
- Improved convergence speed due to the **partial Gauss-Seidel** nature of the
  updates.
- **Memory footprint reduced by a factor of two** compared to standard
  red-black-ordered MSBG Jacobi, since red-black-ordered MSBG blocks can be
  updated in place without an additional destination buffer.
- Like the P2G transfer, this block-"colored" algorithm inherently relies on the
  **globally uniform block structure of MSBG** and is **not applicable to
  tree-based data structures** (Fig. 11a: multi-color/checkerboard techniques
  cannot be efficiently applied to octrees).

**Damping weight.** Due to the partial-Jacobi nature of the smoother, a weighting
factor $\omega$ (Adams et al. 2003) is required to guarantee overall convergence:
$$\boxed{\omega = \tfrac{6}{7}} \quad \text{(optimal in three dimensions, Trottenberg et al. 2000)}$$

Weighted Gauss-Seidel update for cell $j$:
$$\delta = \omega\,\frac{b_j - \mathcal{L}_j\mathbf{u}}{\mathcal{L}_{jj}}, \qquad u_j \leftarrow u_j + \delta$$

**Adaptive number of block-local sweeps.** To amortize the cost of moving block
data from RAM to CPU cache (and the block's local **"halo buffer"** holding
boundary data from neighboring blocks), **multiple relaxation sweeps are
performed on a block once its active data channels are loaded into cache**.
Unlike McAdams et al. 2010, the number of block-local sweeps is **not fixed**;
it is determined adaptively by monitoring the **block-local residual**, available
as the **absolute value of the local update during relaxation**. The loop
terminates when
$$r / r_0 < \tfrac{1}{3}$$
i.e. the residual is reduced by **1/3**, approximating the **optimal reduction
factor for a Gauss-Seidel smoother in 3D Poisson problems** (Mohr and Wienands
2004). A cap $k_{max}$ bounds the sweeps. This exact value, like other parameters
in Algorithm 2, is **not critical for overall convergence**, because the adaptive
smoother functions solely within a preconditioner.

### 6.4 Adaptive Relaxation

Based on **fully adaptive multigrid** (Kowarschik et al. 2006; Rüde 1993):
adaptively focus smoothing effort locally on regions with high residual errors.

**Active block list.**
- Maintained as a **dynamic list of active MSBG blocks**, implemented as a
  **fast lockless FIFO queue**.
- **Initialization** at multigrid level $L$: the list includes all
  **relaxation-relevant blocks** $b_i$ — those with
  $$l(i,L) \le L$$
  — **plus a one-block-wide transition zone at the next coarser level** (i.e.
  blocks that are resolution-boundary blocks with $l(i,L) = L+1$).
- Blocks are iteratively selected from the active list for relaxation.
- If a block's **maximum effective residual exceeds a threshold $\theta$** after
  relaxation, it is **reinserted** into the list.
- **Neighboring blocks are also added** if their **border cell residuals** exceed
  the threshold.
- Unlike Kowarschik et al. 2006 / Rüde 1993, the loop is **not** continued until
  the active set becomes empty. Instead it is **terminated after a fixed number
  of adaptive relaxation steps: 10–20** ($N_{iter,max}$). This is sufficient
  because the algorithm is employed **within a preconditioner** rather than as a
  standalone multigrid solver.

**Efficiency.** Fig. 12 shows that after ~2 smoothing steps the active set is
~10% of all blocks, and after 6 steps ~1%. After a few iterations the remaining
active block set often fits entirely into the CPU's **last-level cache**,
eliminating the RAM bandwidth bottleneck typical of relaxation-type
small-stencil sweeps during later iterations.

**Effective residual definition (high-contrast phase fields).** Kowarschik et al.
use the **scaled residual** (residual divided by the matrix diagonal). This is
ill-suited for two-phase flow: the **matrix diagonal is orders of magnitude
larger in the air phase than in the water phase**, compressing the dynamic range
of the residual error in the air phase by a factor of **1000**, making it
unlikely for air blocks to ever exceed the error threshold. Proposed alternative
metric:

> **Effective residual = the maximum change over all variables of a block after
> the local block relaxations, multiplied by the diagonal** — effectively
> "contrast-equalizing" the phases.

**Adaptive error threshold $\theta$.** Prior work uses a fixed user-defined
residual error threshold with no guidance on its value. A fixed $\theta$ is
problematic here, since adaptive-relaxation-based multigrid is used as a CG
preconditioner, where the dynamic range of the effective residual error varies
widely (often decreasing exponentially) with each CG iteration. Therefore
$\theta$ is determined **adaptively**:

1. Compute the **log-histogram of the initial effective residual**.
2. Apply fast histogram segmentation. In practice, a simple choice works well:
   $$\theta = \mathrm{mean}(\log r) + 2\,\mathrm{std}(\log r)$$
   (mean plus two standard deviations of the log-histogram).

Using MSBG, histogram construction is performed very efficiently **at the block
level, using only the per-block maxima of the residuals**.

#### Algorithm 2: Parallel Two-Stage-Red-Black Adaptive Multigrid Relaxation

```
Input: Multigrid level L, effective residual threshold theta

n, n' <- 0                         # Number of active blocks (current, new)
S0, S, S' <- {0}                   # Block status flags arrays

Function RelaxBlock(i, R):
  for i_rb <- 0 to 1 do                                  # Inner red-black scheme
    for each cell j of block i where color(j) = i_rb do
      delta <- omega * ( b_j - L_j u ) / L_jj            # Weighted Gauss-Seidel; omega = 6/7
      R[j]  <- delta
      u_j   <- u_j + delta
  return max(R)

Procedure ActivateBlock(i, R):
  if AtomicCompareExchange(S'[i], 1, 0) = 0 then
    B'[ AtomicIncrement(n') ] <- i
  # Activate affected neighbor blocks
  for each face f_{k in [1,6]} of block i do
    i2 <- NeighborBlock(i, k)
    if S0[i2] and max_{j in f_k} R[j] > theta then
      if AtomicCompareExchange(S'[i2], 1, 0) = 0 then
        B'[ AtomicIncrement(n') ] <- i2

for each block i do in parallel
  if level(i, L) <= L  or  ( IsResolutionBoundaryBlock(i) and level(i, L) = L+1 ) then
    B[ AtomicIncrement(n) ] <- i                         # Initial list of active blocks
    S0[i] <- 1

for i_iter <- 1 to N_iter,max do
  if n = 0 then break
  SortByMortonCode(B)                                    # Improve coherence of memory access
  for I_rb <- 0 to 1 do                                  # Outer red-black scheme
    for j <- 1 to n do in parallel
      i <- B[j]
      x_b, y_b, z_b <- BlockCoords(i)
      if (x_b + y_b + z_b) mod 2 = I_rb then continue
      r0 <- RelaxBlock(i, R);  k <- 1                     # R is a local buffer
      repeat
        r, R <- RelaxBlock(i, R);  k <- k + 1
      until r/r0 < 1/3  or  k >= k_max
      if r > theta then ActivateBlock(i, R)
  for j <- 1 to n do  S[ B[j] ] <- 0
  swap(n, n');  swapRef(B, B');  swapRef(S, S')
```

Notes on Algorithm 2 (from Fig. 11b caption): the figure shows the active
elements during the **first half of one parallel half-sweep**
($I_{rb} = i_{rb} = 0$), for SIMD-vector blocks at two different resolution
levels (0 and 1). A SIMD vector spans 8 consecutive cells of a block row.

### Flexible Preconditioned Conjugate Gradient

Standard multigrid-preconditioned CG requires the preconditioner to **preserve
symmetry**, which necessitates performing **exactly the same number of relaxation
sweeps during pre- and post-smoothing** (McAdams et al. 2010; Tatebe 1993) — a
requirement inherently violated by adaptive relaxation.

Resolution: use the **flexible preconditioned conjugate gradient method**
(Bouwmeester et al. 2015), which is robust even when the preconditioner is not
symmetric positive definite, implemented by replacing the standard
**Fletcher–Reeves** update step with the more robust **Polak–Ribière** variant.
It requires storing one additional vector; this overhead is offset by the
block-based red-black scheme (Section 6.3), which eliminates the need for a
temporary destination buffer during parallel iterations.

**V-cycle structure.** Standard V-cycle (Briggs et al. 2000; Mandel 1988;
Trottenberg et al. 2000) — pre-smooth, restrict residual, recurse, prolong error,
post-smooth — with the smoother replaced by Algorithm 2 and the coarse operators
built by Algorithm 1 via Eq. (14). Prolongation/restriction operate across MSBG
resolution boundaries directly (Fig. 10), so no separate ghost-cell accumulation
pass is needed. One V-cycle serves as one application of the CG preconditioner.

---

## 7. Spray and Rendering

- **Escaped FLIP particles** (Section 3.4) are converted into **purely Lagrangian
  droplet particles**, which move under **gravity** and **drag forces induced by
  the air velocity field** (Gissler et al. 2017; Song et al. 2005), integrated
  with **locally adaptive time stepping**.
- **Additional spray particles are emitted in areas of steep velocity gradients**
  (Ihmsen et al. 2012).
- Per-particle **droplet radii $r_p$** are sampled from a **log-normal
  distribution** (Babinsky and Sojka 2002) on droplet emission.
- Droplet radii are evolved by a **physics-based evaporation ODE** (Sirignano
  2010):
  $$\frac{dr_p}{dt} \sim -\frac{1}{r}$$
  integrated with **RK2**.
- Spray is simulated and rendered at **significantly higher resolution than the
  base pressure solver**: a **spray density field** at **16-bit float
  precision**, with a resolution **two to four times higher** than the pressure
  and velocity fields. Spray particles are simulated at the increased resolution
  and rasterized into it to yield a volumetric density for rendering. MSBG
  handles these sparse grids with **up to tens of thousands of voxels per
  dimension** (Section 8.1).
- Rendering: an **MSBG-based volumetric ray tracer with approximate multiple
  scattering**, rendering the spray density field and the water. The rendered
  surface is reconstructed from the liquid particles using **standard particle
  skinning and surface smoothing** (Bhattacharya et al. 2011).
- **Spray particle rasterization and water surface reconstruction are only
  necessary at render time, i.e. every 5–10 time steps.**
- High-curvature areas of the air–water interface are rendered with
  **foam-like scattering coefficients**. **No post-processing effects or
  procedural textures are applied.**

### Escaped particles / surface tension (Section 3.4, restated — referenced by §7)

Escaped particles $p^*$ at position $\mathbf{x}^*$ are candidates for conversion
into:
- **droplets** if $\phi(\mathbf{x}^*) < \tfrac12 - \delta_\phi$
- **bubbles** if $\phi(\mathbf{x}^*) > \tfrac12 + \delta_\phi$

$\delta_\phi$ controls the rate of droplet/bubble production, with
**default value 0.2**. Escaped droplets are subsequently simulated as purely
Lagrangian point masses (Section 7). **Bubble particles are simply deleted** in
the current implementation (their visual contribution is negligible at the target
large scale). When droplet particles **merge with the liquid body again, they are
re-converted to FLIP simulation particles**. Surface tension is **neglected**
entirely. No "particle bounce-back" heuristic is used — the velocity field
adheres to a physically more accurate no-slip condition between phases.

---

## 8. Results

**Hardware.** Unless stated otherwise, all tests and simulations were conducted
on a **single AMD workstation (Ryzen Threadripper) with 32 cores and 256 GB of
RAM**. The asteroid impact example (Fig. 13) was executed on a **laptop (Intel
Core i7, 14 cores, 32 GB RAM)**. **No GPU acceleration is used** — important for
very large-scale scenarios whose memory requirements exceed the limited RAM
capacity of even high-end GPUs.

### 8.1 Adaptive Grids with MSBG

**Benchmark setup (Fig. 14).** 1.6 billion particles rasterized (P2G) as spheres
with a radius of **3 grid cells** on a sparse grid with an effective resolution of
$10{,}000^2 \times 5{,}000$ cells, resulting in **10.6 billion active voxels**,
subsequently filtered using **ten iterations of Laplacian diffusion (PDE)**. Then
**one billion point queries** with **linear interpolation** of the filtered grid
values (G2P). Particles were taken from a snapshot of the asteroid impact scenario
(Fig. 13). Baseline: **OpenVDB version 11**, blocks $8^3$ (MSBG blocks $16^3$,
eight times the 3D volume).

**Table 2. MSBG performance across representative operations**

| Test | VDB | MSBG | Improvement |
|---|---|---|---|
| P2G (Mparts/s) ↑ | 16.7 | 62.6 | ×3.7 |
| PDE (Mcells/s) ↑ | 948 | 9097 | ×9.6 |
| G2P (Mparts/s) ↑ | 634 | 4051 | ×6.4 |
| Memory usage (GB) ↓ | 37.3 | 41.4 | ×0.9 |

Memory footprint increases only moderately by **11%** despite blocks being twice
as large per dimension.

**Table 3. Memory bandwidth utilization of a seven-point stencil Laplacian kernel
(percent of peak machine bandwidth vs. streaming copy baseline)**

| | Streaming copy | VDB | SPGrid | MSBG |
|---|---|---|---|---|
| Bandwidth utilization | 100% | 11% | 39% | **90%** |

(SPGrid number derived from Table 1 in Aanjaneya et al. 2017.) MSBG nearly
saturates machine bandwidth.

**100-billion-voxels example (Fig. 15).** 32k point samples of the original
Stanford bunny dataset, rasterized as a cloud of **32k small bunnies**, giving
$32k \times 32k = 1\mathrm{B}$ particles in total, onto a **$32k^3$ MSBG grid
(base block resolution 32)** as a narrow-band signed distance field, yielding
approximately **100 billion voxels active out of 35 trillion virtual voxels**
($32768^3$ resolution = 32 trillion virtual voxels per the figure caption). A
smooth surface was then reconstructed through **ten iterations of a mean
curvature flow PDE** on the MSBG grid, at a speed of **10 billion unknowns per
second per iteration**. To fit 100 billion voxels into main memory, a **single
MSBG data channel with half precision** was used, exploiting MSBG's
**block-multicolor-based PDE-stencil-operators** for solving the mean curvature
flow — a second-order PDE with wide stencil — in parallel via an **eight-color
scheme**, which eliminates the need for an additional temporary data channel.
The surface was rendered at **4K resolution in ≈ 30 seconds** using an MSBG-based
ray tracer.

### 8.2 Basic Phase-Field Tests

**(a) Advection of phase field in a prescribed velocity field** (shear flow
advection test, Fig. 16a). Passive advection of a spherical shape in a
prescribed, time-dependent velocity field inducing a typical shear flow (De Rosis
and Enan 2021; Fakhari et al. 2019; Li et al. 2022); setup detail per Li et al.
2022. Renderings at $t = 0$, maximum deformation $t = T/2$, and after reverse
advection $t = T$. Relative error:
$$E = \left(\frac{\sum_i \big(\phi(x_i,0) - \phi(x_i,T)\big)^2}{\phi^2(x_i,0)}\right)^{1/2}$$

**Table 4. Relative phase field error of the shear flow advection test**

| Model | Relative error |
|---|---|
| De Rosis and Enan 2021 | 0.0490 |
| Li et al. 2022 | 0.0500 |
| **Ours** | **0.0314** |

The hybrid Lagrangian–Eulerian advection method yields lower error than existing
purely Eulerian phase field transport schemes.

**(b) 3D Rayleigh–Taylor instability** (Fig. 16b), setup after He et al. 1999,
high density contrast:

| Parameter | Value |
|---|---|
| Atwood number $A = (\rho_l - \rho_g)/(\rho_l + \rho_g)$ | **0.9** |
| Surface tension | neglected |
| Grid resolution | $256 \times 64 \times 64$ |
| Domain length $L$ | **0.05** |
| Viscosity $\nu_l = \nu_g$ | **$4\times10^{-5}$** |
| Gravitational acceleration $g$ | **9.81** |
| $\alpha_\phi$ | **0.5** |
| $CFL$ | **0.5** |
| (all in SI units) | |

Spike-tip position development is in good agreement with the reference. In later
stages, because of the absence of surface tension, thin sheets of the interface
begin to break up into ligaments and droplets — not observed in He et al. 1999 —
highlighting that the relatively sharp phase field ($\epsilon_\phi = \Delta x$)
resolves fine interface structures near the Nyquist limit of the grid. The
reference image shown is at $t = 3.0$.

### 8.3 Adaptive Poisson Solver

**Test scenario: asteroid impact** (complex dynamic fluid–solid interactions,
high density contrast two-phase fluid–fluid interface geometry, intense
turbulence).

| Parameter | Value |
|---|---|
| Effective grid resolution | **$1536^3$** |
| Particles | **≈ 1.6 billion** |
| Metric | PCG iterations and total CPU time per time step, averaged over 10 steps during the most challenging phase |

**Fig. 17 comparison (logarithmic y-axis):**

| Method | PCG iterations per step | Note |
|---|---|---|
| Incomplete Cholesky (Bridson 2015) | **> $10^3$** (highest) | baseline |
| Geometric Multigrid (e.g. Setaluri et al. 2014) | ~ $10^2$ (≈ 90) | ≈ an order of magnitude improvement over IC |
| Galerkin MG (Shao et al. 2022) + boundary relaxation (McAdams et al. 2010) | ~ 20 | further ≈ **factor of four** reduction in cost vs geometric MG |
| **Adaptive Multigrid (ours)** | ~ 7 | most efficient; improves iteration count and CPU time by **an additional factor of three** |

**Comparison with high-performance CFD solvers.** Re-simulation of the "difficult
scenario" of **MacLachlan et al. 2008** — rising and disintegrating bubbles,
deliberately excluding surface tension, resulting in highly complex particles and
dynamic interface geometries. Residual error tolerance intentionally set to
$$\epsilon_{tol} = 10^{-8}$$
for this benchmark (in contrast to $10^{-4}$ used for all other examples).

**Table 5. Comparison with CFD pressure solvers for a two-phase flow benchmark**
(CG iterations per time step for BoxMG and DIC as reported in Fig. 7a of
MacLachlan et al. 2008, during the "difficult" phase)

| Method | Iterations per step | Stencil size | Pre-processing cost per step | Adaptive mesh refinement |
|---|---|---|---|---|
| **Ours** | **≈ 30** | **4** | **Low** | Native/MSBG |
| BoxMG-CG | ≈ 40 | 4 / 14 | High | Difficult |
| DIC-CG | > 300 | 4 | Medium | Medium |

Our solver requires fewer iterations than the best-performing method (BoxMG)
while incurring lower computational cost per iteration due to its smaller,
4-point stencil. BoxMG requires 14-point stencils on all but the finest
resolution levels. Pre-processing: ours requires only a **simple sweep over the
multigrid hierarchy to calculate three coefficients per cell** (Algorithm 1),
whereas BoxMG involves more expensive sweeps with nested loops to compute **14
coefficients for a symmetric 27-point stencil**.

**Fig. 18 setup.** Rising and disintegrating bubbles at $t = 0.0266$ seconds.
(a) Grid resolution of $200^3$ as used by MacLachlan et al. (re-simulation);
(b) our adaptive simulation at **ten times the original resolution**, i.e.
$2000^3$ compared to $200^3$ — at **approximately the same number of iterations
per time step**.

**Table 6. Timing breakdown of an average simulation step**
(for the scenario in Fig. 25, compared to the SPGrid-based solver of Setaluri et
al. 2014 at their highest resolution example, and Zeng et al. 2022 for a dam
break simulation. A dash indicates not available/applicable.)

| Method | P2G | Pressure | Advect | Other |
|---|---|---|---|---|
| **Ours** | 27% | **30%** | 13% | 30% |
| Setaluri et al. | – | 91% | 5% | 4% |
| Zeng et al. | – | 83% | – | 17% |

**Eliminating the pressure-solver bottleneck:** in all examples, the pressure
solve accounted for **no more than about one-third of total runtime**, regardless
of grid resolution or flow complexity.

### 8.4 Full Simulation Results

**Global settings for the large-scale examples:**

| Parameter | Value |
|---|---|
| Simulation time per time step | typically **30–100 seconds** |
| Time steps per frame | **2–7** |
| CFL number | **3–4** |
| $\alpha_{FLIP}$ (liquid) | **0.985** |
| $\alpha_{FLIP}$ (air) | **0.96** |
| Residual tolerance $\epsilon_{tol}$ | $10^{-4}$ |
| Vorticity confinement / artificial turbulence | **none** (all physics-based) |

Resolutions are expressed as **effective resolution** — the equivalent resolution
of a *uniform* grid required to resolve the phase interface at the resolution of
the finest MSBG level.

**Scene: Asteroid impact (Fig. 13).**
- 600 million particles, simulated **on a laptop**.
- Effective resolution **$1024^3$ for pressure**, **$3072^3$ for spray**.
- Average simulation time per time step: **≈ 70 s**.
- (Workstation variant used in §8.3: $1536^3$, ≈1.6 B particles.)

**Scene: Dam break with obstacle (Fig. 19).**
- Classic dam break similar to Li et al. 2022, 2024, with the **relative width of
  the simulation domain enlarged** to provide more room for splashing and
  water-air interaction.
- Effective resolution increased from $800 \times 400^2$ (Li et al. 2024) to
  **$1600^3$**.
- Two scales simulated: **small-scale (3 meters)** and **large-scale (400
  meters)**.
- Each conducted with both a standard **single-phase free-surface** solver and
  the proposed **two-phase** solver. Snapshots at $t = 0.8\,\text{s}$ (small) and
  $t = 12\,\text{s}$ (large).
- Finding: single-phase simulations look essentially the same regardless of scale
  (except for time scaling) because non-linear air-phase effects are missing;
  two-phase reproduces characteristic large-scale features (explosion-like jets,
  streamers of spray formed by vortex tubes of the turbulent air velocity).

**Scene: Numerical wave tank and two-phase turbulence (Figs. 20, 22).**
- Large solitary ocean waves in a numerical wave tank driven by a standard
  **piston-type wavemaker** (Wang et al. 2019).
- Fig. 20: effective resolution **$4096 \times 1024 \times 512$**.
- Fig. 22: part of a 2D slice (vorticity magnitude) from a 3D simulation with
  effective resolution **$4096 \times 1024 \times 512$** cells and a **total
  simulation domain size of 500 meters**; the annotated inflow velocity is
  **15 m/s** over a **60 m** span; resolves the air-water boundary layer, flow
  separation, free shear layer, and the turbulent wake behind the wave crest.

**Scene: Wave and obstacle at extreme resolution (Fig. 1).**
- Adaptive grid resolution increased to the limits of 256 GB of memory:
  **$6k \times 2k \times 1.6k$ for the pressure** and
  **$18k \times 6k \times 5k$ for the spray density field**.
- **More than 3 billion particles.**
- A **solid obstacle placed in the middle of the wave tank**.
- Average simulation time **≈ 2 minutes per time step**, **total duration 3.5
  days**.

**Scene: Dam discharge (Fig. 3).**
- Discharge flow of a large hydrodynamic dam, inspired by the **Three Gorges Dam
  in Sandouping, China**.
- Water flows at a velocity of **40 m s⁻¹** through a floodgate of **diameter
  5 m**, inclined at an angle of **18 degrees**.
- The floodgate is located at a **height of 9 m** above a reservoir pool with an
  **initial water depth of 3 m**.
- Total size of the simulation domain: **$L = 200$ m**.
- Resolved at **2048 cells for the pressure** and **8192 cells for the rendered
  water surface and spray density fields**.

**Scene: Tsunami wave hitting a complex coast line (Fig. 24).**
- Tsunami wave impacting a **procedurally generated fractal coast**, exhibiting
  geometric detail across all spatial frequencies.
- **Initial wave speed $v_0 = 30$ m s⁻¹.**
- **Effective resolution $4096 \times 1024^2$.**

**Scene: Ocean wave-train breaking on the beach (Figs. 4, 5, 21, 23, 25).**
- Kilometer-sized ocean scene with **multiple wave trains shoaling and breaking
  on an artificial beach**.
- The **wave generator is inclined** so waves interact with their own reflections
  from the domain boundary (Fig. 21 shows the setup with inclined wave generator;
  the domain is annotated ~1000 m).
- **Effective resolution $3072 \times 1536 \times 1024$.**
- **Approximately 2 billion particles.**
- Each time step required, on average, **60 seconds of CPU time**.
- Fig. 25: 2 billion particles. Fig. 23: detail zoom of the turbulent wave tube.

**Table 7. Statistics for the large scale simulations.**
*The number of active pressure cells is the number of FLIP particles divided by 8.*

| Figure | Effective resolution — Pressure | Effective resolution — Spray-density | Max particles FLIP/Spray | Min. per Step/frame |
|---|---|---|---|---|
| 1 | $6144 \times 2048 \times 1600$ | $18432 \times 6144 \times 4800$ | 3.1 / 1.0 B | 1.9 / 9.5 |
| 3 | $2048 \times 1024^2$ | $8192 \times 4096^2$ | 1.1 / 0.5 B | 0.7 / 4.0 |
| 4, 5, 25 | $3072 \times 1536 \times 1024$ | $12288 \times 6144 \times 4096$ | 2.0 / 0.1 B | 1.0 / 5.7 |
| 22 | $4096 \times 1024 \times 512$ | $12288 \times 3072 \times 1536$ | 1.0 / 0.2 B | 0.6 / 6.1 |
| 24 | $4096 \times 1024^2$ | $12288 \times 3072^2$ | 1.2 / 0.5 B | 0.8 / 7.0 |

("Min." = minutes.)

---

## 9. Limitations

1. **Resolution boundaries cannot cross the liquid–air interface.** This is an
   inherent constraint shared with most Cartesian grid-based adaptivity schemes.
   In principle it is possible to lift this restriction (Ando and Batty 2020);
   planned as future work.
2. **Maximum depth of refinement is restricted to the binary logarithm of the
   MSBG base block resolution**, i.e. $\log_2(16)$ or $\log_2(32)$. Not found
   limiting in practice, as in 3D each level of refinement reduces computational
   costs significantly, by a factor of eight.
3. **FLIP noise.** As with other FLIP-based methods, the approach can exhibit
   relatively high levels of noise in field quantities arising from
   high-frequency noise in the particle distribution. Not critical for
   large-scale, highly turbulent target scenarios (which inherently exhibit a
   high level of "natural" noise), but it could impede effectiveness for **very
   small-scale, surface-tension dominated scenarios, especially at low density
   ratios**.

## 10. Conclusions and Outlook

Future directions named: (a) extend spatial adaptivity to cases where resolution
boundaries intersect the liquid surface; (b) integration of simulated turbulent
wind fields to drive ocean waves with physical wind dynamics; (c) improve the
spray model, possibly by incorporating **droplet atomization models from CFD**;
(d) employ adaptive PF-FLIP in CFD applications such as large-scale numerical
wave tanks and wave–structure interaction in coastal/offshore engineering.

---

## A. Consolidated parameter table

| Symbol | Meaning | Default / used value | Where |
|---|---|---|---|
| $n_0$, $m_0$ | MSBG base block resolution (cells per block edge) | **16** (default); **32** also used ("MSBG-32") | §4.1, §4.2, §5.4 |
| $L_{max}$ | Max refinement depth | $\log_2 n_0$ = 4 (or 5); **3–4 levels used in all results** | §4.2, §9 |
| — | Grading constraint | **2:1** (neighbor levels differ by ≤ 1) | §4.2, §5.4(2) |
| $\Delta_g$ | Coarse-fine gradient constant | **3/4** | §4.2 |
| $\sigma_{CF}$ | Coarse-fine face scaling | $1/(4\Delta_g) = 1/3$ | §6.1 |
| $\sigma_{FC}$ | Fine-coarse face scaling | $2\sigma_{CF} = 2/3$ | §6.1 |
| $r_0$ | Base particle radius | $\Delta x$ (finest grid spacing) | §3.3, §5 |
| $r_p$ | Particle radius | $r_0 2^{l_p}$ | §5 |
| $l_p$ | Particle level | integer | §5 |
| $l_i = M[b_i]$ | Grid (block) refinement level | integer, from refinement map $M$ | §5 |
| $prob_{del}$ | Stochastic coarsening deletion probability | $1 - 1/(2^{l_{new}-l_{old}})^3$ (= 7/8 for one level) | §5.1 |
| $D_p$ | Deferred-coarsening counter (hysteresis) | coarsen only when $D_p \ge \mathbf{3}$; reset to 0 if $l_{new}\le l_p$ | §5.4(6) |
| $\beta_{bw}$ | Refinement band width control | application-specific | §5.4(6) |
| $\tilde d(\mathbf{x})$ | Reduced-resolution auxiliary SDF | from downsampled+thresholded $\phi$, first-order marching | §5.2 |
| $\epsilon_\phi$ | Phase-field interface thickness | $= r_0 = \Delta x$ (one cell) | §3.3 |
| $\alpha_\phi$ | Phase-field noisiness/stiffness knob (Eq. 7) | **1** (default); **0.5** for Rayleigh–Taylor | §3.3, §8.2 |
| $\eta_\phi$ | Density scaling factor | $\log(\rho_l/\rho_g)$ | §3.3 |
| $\rho_g$ | Air density | $\rho_l / 1000$ | §3.3 |
| $\delta_\phi$ | Escaped-particle droplet/bubble threshold offset | **0.2** | §3.4 |
| $\alpha_{FLIP}$ | FLIP/PIC blend (liquid) | **0.985** | §8.4 |
| $\alpha_{FLIP}$ | FLIP/PIC blend (air) | **0.96** | §8.4 |
| $\nu$ | Numerical viscosity | $(1-\alpha_{FLIP})\Delta x^2/(6\Delta t)$ | Eq. (13) |
| $\omega$ | Gauss-Seidel damping weight | **6/7** (optimal in 3D) | §6.3 |
| — | Block-local sweep stop criterion | $r/r_0 < \mathbf{1/3}$ | §6.3 |
| $k_{max}$ | Max block-local relaxation sweeps | cap (value unspecified) | Alg. 2 |
| $N_{iter,max}$ | Adaptive relaxation steps per smoother call | **10–20** | §6.4 |
| $\theta$ | Effective residual threshold | adaptive: **mean + 2·std of log-histogram** of initial effective residual | §6.4 |
| — | Effective residual metric | max change over block variables after local relaxation, **× diagonal** | §6.4 |
| $\epsilon_{tol}$ | PCG residual tolerance | **$10^{-4}$** (all examples); **$10^{-8}$** (MacLachlan benchmark) | §8.3 |
| $CFL$ | CFL number | **3–4** (production); **0.5** (Rayleigh–Taylor) | §8.4, §8.2 |
| — | Stencil | 7-point (4 stored coefficients) | §6.1 |
| — | CG variant | Flexible PCG, **Polak–Ribière** update | §6.4 |
| — | SIMD word | **8 floats (AVX)** | §6.3 |
| — | Splatting colors | **8-color** lock-free scheme (Fang et al. 2018) | §4.3 |
| — | Spray density field | **16-bit float**, **2–4×** pressure/velocity resolution | §7 |
| — | Render cadence | every **5–10** time steps | §7 |
| $dr_p/dt$ | Droplet evaporation | $\sim -1/r$, RK2 | §7 |
| — | Advection integrator | **3rd-order Runge-Kutta**, locally adaptive time stepping | §3.7 |
| — | Threading | TBB, dynamic task-based scheduling | §4.3 |
| — | Block ordering | Morton code, computed once per block | §4.3 |

---

## B. Implementation checklist (ordering of one time step)

1. Build block refinement map $M$ from particles (atomic min over bounding boxes
   expanded by CFL epsilon).
2. Grade $M$ to 2:1.
3. (Re)allocate MSBG blocks + data channels for the new $M$.
4. P2G: 8-color lock-free splat of mass + momentum to MAC faces using Eq. (6),
   27-cell neighborhood for $l_p = l$, extended for $l_p = l+1$.
5. Phase field $\phi$ from the accumulated weights via Eq. (7); tag cells
   $\phi<0.5$ as air; set $\beta$ face coefficients via Eq. (9) with $\phi$-based
   density interpolation ($\phi=1$ for liquid-liquid faces, $\phi=0$ for air-air
   faces).
6. Add gravity / external forces to $\tilde{\mathbf{u}}^*$.
7. Compute auxiliary $\tilde d$ from the downsampled/thresholded $\phi$.
8. Build the multigrid hierarchy: Algorithm 1 (Galerkin coarsening of face
   coefficients across all multigrid levels $L = 1 \ldots M-1$).
9. Solve Eq. (8) with flexible PCG (Polak–Ribière), preconditioned by one V-cycle
   whose smoother is Algorithm 2. Compute $\theta$ once per preconditioner
   application from the log-histogram of block-max effective residuals.
10. Project: $\mathbf{u} = \mathbf{u}^* - \Delta t\,\frac{1}{\rho}\nabla p$.
11. G2P with per-particle $\alpha_{FLIP}$ chosen from phase and local $\Delta x$
    via Eq. (13) (Section 5.3), using Eq. (11)–(12).
12. Particle level update, splitting (8-way, octant-jittered) / stochastic
    coarsening with $prob_{del}$ and counter $D_p$.
13. Escaped-particle classification ($\delta_\phi = 0.2$): convert to Lagrangian
    droplets (delete bubbles); emit extra spray in steep-velocity-gradient
    regions; integrate droplet motion (gravity + air drag) and the evaporation
    ODE.
14. Extrapolate liquid velocity a few cells into the air phase; advect all
    particles with RK3 and locally adaptive sub-stepping.

---

## C. Notes on fidelity of this transcription

- Algorithms 1 and 2 are transcribed line-for-line from the paper's algorithm
  boxes; indentation reflects the printed indentation.
- The refinement-criterion formula in §5.4 step (6) prints the subscript of
  $\beta$ in a small italic font; it is read here as $\beta_{bw}$ ("band width"),
  consistent with the accompanying text "controls the width of refinement bands".
- Only Eq. (14) is newly numbered on pages 9–23; all other numbered equations in
  this document ((6)–(13)) come from Sections 3.x and are restated for
  completeness.
