"""Variable-coefficient pressure Poisson solver (paper section 6).

Stage 1 implements the *uniform-grid* specialisation of the paper's solver:

* **Unsmoothed algebraic aggregation Galerkin multigrid** (paper 6.1, eq. 14).
  Prolongation ``P`` is piecewise-constant injection over 2x2x2 aggregates and
  ``R = P^T`` is summation, so the Galerkin coarse face coefficient collapses to
  the sum of the four fine faces straddling the coarse face (Algorithm 1 with
  all blocks at resolution level 0, where its transition factor ``s`` is 1).
* **Red-black weighted Gauss-Seidel smoother** with ``omega = 6/7`` (paper 6.3).
  On the GPU the paper's two-stage *block* red-black / *cell* Gauss-Seidel
  hybrid degenerates to plain cell-level red-black, which has the same
  partial-Gauss-Seidel character without needing a destination buffer.
* **Flexible PCG with Polak-Ribiere** update (paper 6.4), so a non-symmetric
  preconditioner application stays safe.

The discrete system is ``A p = b`` with
``(A p)_i = sum_f beta_f (p_i - p_nbr)``, i.e. eq. (10) scaled by ``dx^2``.
``beta_f = 0`` on solid/domain faces encodes homogeneous Neumann conditions, so
with both phases filling a closed box ``A`` is singular with the constant
null space -- handled by projecting ``b`` and ``p`` to zero mean.

Not yet implemented (stages 2-3): MSBG multiresolution levels, adaptive
relaxation with the active-block queue and adaptive threshold ``theta``.
"""

from __future__ import annotations

import warp as wp


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
@wp.func
def _bx(b: wp.array3d(dtype=float), i: int, j: int, k: int) -> float:
    return b[i, j, k]


@wp.kernel
def compute_diag(
    bx: wp.array3d(dtype=float),
    by: wp.array3d(dtype=float),
    bz: wp.array3d(dtype=float),
    diag: wp.array3d(dtype=float),
):
    """Diagonal = sum of the six face coefficients (paper 6.1)."""
    i, j, k = wp.tid()
    diag[i, j, k] = (
        bx[i, j, k] + bx[i + 1, j, k] + by[i, j, k] + by[i, j + 1, k] + bz[i, j, k] + bz[i, j, k + 1]
    )


@wp.func
def _apply_row(
    p: wp.array3d(dtype=float),
    bx: wp.array3d(dtype=float),
    by: wp.array3d(dtype=float),
    bz: wp.array3d(dtype=float),
    diag: wp.array3d(dtype=float),
    i: int,
    j: int,
    k: int,
) -> float:
    nx = p.shape[0]
    ny = p.shape[1]
    nz = p.shape[2]
    # beta is zero on the outer faces, so clamped reads are harmless
    s = bx[i, j, k] * p[wp.max(i - 1, 0), j, k]
    s += bx[i + 1, j, k] * p[wp.min(i + 1, nx - 1), j, k]
    s += by[i, j, k] * p[i, wp.max(j - 1, 0), k]
    s += by[i, j + 1, k] * p[i, wp.min(j + 1, ny - 1), k]
    s += bz[i, j, k] * p[i, j, wp.max(k - 1, 0)]
    s += bz[i, j, k + 1] * p[i, j, wp.min(k + 1, nz - 1)]
    return diag[i, j, k] * p[i, j, k] - s


@wp.kernel
def apply_A(
    p: wp.array3d(dtype=float),
    bx: wp.array3d(dtype=float),
    by: wp.array3d(dtype=float),
    bz: wp.array3d(dtype=float),
    diag: wp.array3d(dtype=float),
    out: wp.array3d(dtype=float),
):
    i, j, k = wp.tid()
    out[i, j, k] = _apply_row(p, bx, by, bz, diag, i, j, k)


@wp.kernel
def residual(
    p: wp.array3d(dtype=float),
    b: wp.array3d(dtype=float),
    bx: wp.array3d(dtype=float),
    by: wp.array3d(dtype=float),
    bz: wp.array3d(dtype=float),
    diag: wp.array3d(dtype=float),
    r: wp.array3d(dtype=float),
):
    i, j, k = wp.tid()
    r[i, j, k] = b[i, j, k] - _apply_row(p, bx, by, bz, diag, i, j, k)


@wp.kernel
def rbgs_sweep(
    p: wp.array3d(dtype=float),
    b: wp.array3d(dtype=float),
    bx: wp.array3d(dtype=float),
    by: wp.array3d(dtype=float),
    bz: wp.array3d(dtype=float),
    diag: wp.array3d(dtype=float),
    color: int,
    omega: float,
):
    """Weighted Gauss-Seidel half-sweep, paper 6.3:
    ``delta = omega (b_j - L_j u) / L_jj``, updated in place.
    """
    i, j, k = wp.tid()
    if ((i + j + k) & 1) != color:
        return
    d = diag[i, j, k]
    if d <= 0.0:
        return
    res = b[i, j, k] - _apply_row(p, bx, by, bz, diag, i, j, k)
    p[i, j, k] += omega * res / d


# --- inter-grid transfer (unsmoothed aggregation) ---------------------------
@wp.kernel
def galerkin_coarsen_faces(fine: wp.array3d(dtype=float), coarse: wp.array3d(dtype=float), axis: int):
    """Coarse face coefficient = sum of the 4 fine faces it covers.

    This is eq. (14) with piecewise-constant ``P`` (paper 6.1 / Algorithm 1),
    specialised to a uniform grid where the transition factor ``s`` is 1.
    """
    I, J, K = wp.tid()
    # the coarse face lies in the fine face plane 2*I (resp. 2*J / 2*K); the
    # two in-plane directions are swept by the (a, b) loop below
    i0 = 2 * I
    j0 = 2 * J
    k0 = 2 * K
    nf0 = fine.shape[0]
    nf1 = fine.shape[1]
    nf2 = fine.shape[2]
    acc = float(0.0)
    for a in range(2):
        for b in range(2):
            if axis == 0:
                ii = i0
                jj = j0 + a
                kk = k0 + b
            elif axis == 1:
                ii = i0 + a
                jj = j0
                kk = k0 + b
            else:
                ii = i0 + a
                jj = j0 + b
                kk = k0
            if ii < nf0 and jj < nf1 and kk < nf2:
                acc += fine[ii, jj, kk]
    coarse[I, J, K] = acc


@wp.kernel
def restrict_sum(fine: wp.array3d(dtype=float), coarse: wp.array3d(dtype=float)):
    """``R = P^T``: sum the 8 fine cells of each aggregate."""
    I, J, K = wp.tid()
    acc = float(0.0)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                acc += fine[2 * I + a, 2 * J + b, 2 * K + c]
    coarse[I, J, K] = acc


@wp.kernel
def prolong_inject_add(coarse: wp.array3d(dtype=float), fine: wp.array3d(dtype=float)):
    """``P``: piecewise-constant injection, accumulated onto the fine level."""
    i, j, k = wp.tid()
    fine[i, j, k] += coarse[i // 2, j // 2, k // 2]


# --- vector utilities ------------------------------------------------------
@wp.kernel
def axpy(a: float, x: wp.array3d(dtype=float), y: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    y[i, j, k] += a * x[i, j, k]


@wp.kernel
def scale_add(y: wp.array3d(dtype=float), a: float, x: wp.array3d(dtype=float)):
    """``y <- x + a * y``"""
    i, j, k = wp.tid()
    y[i, j, k] = x[i, j, k] + a * y[i, j, k]


@wp.kernel
def dot_kernel(x: wp.array3d(dtype=float), y: wp.array3d(dtype=float), out: wp.array(dtype=float)):
    i, j, k = wp.tid()
    wp.atomic_add(out, 0, x[i, j, k] * y[i, j, k])


@wp.kernel
def sum_kernel(x: wp.array3d(dtype=float), out: wp.array(dtype=float)):
    i, j, k = wp.tid()
    wp.atomic_add(out, 0, x[i, j, k])


@wp.kernel
def max_abs_kernel(x: wp.array3d(dtype=float), out: wp.array(dtype=float)):
    i, j, k = wp.tid()
    wp.atomic_max(out, 0, wp.abs(x[i, j, k]))


@wp.kernel
def shift_kernel(x: wp.array3d(dtype=float), c: float):
    i, j, k = wp.tid()
    x[i, j, k] += c


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
class _Level:
    def __init__(self, res, device):
        nx, ny, nz = res
        self.res = (nx, ny, nz)
        z = lambda shape: wp.zeros(shape, dtype=float, device=device)  # noqa: E731
        self.bx = z((nx + 1, ny, nz))
        self.by = z((nx, ny + 1, nz))
        self.bz = z((nx, ny, nz + 1))
        self.diag = z((nx, ny, nz))
        self.p = z((nx, ny, nz))
        self.b = z((nx, ny, nz))
        self.r = z((nx, ny, nz))


class MGPCG:
    """Multigrid-preconditioned flexible conjugate gradient.

    Parameters
    ----------
    res : tuple of 3 ints
        Finest-level cell counts.
    """

    def __init__(self, res, device, *, omega=6.0 / 7.0, pre=2, post=2, coarsest=60, min_dim=4):
        self.device = device
        self.omega = omega
        self.pre = pre
        self.post = post
        self.coarsest = coarsest

        self.levels: list[_Level] = []
        r = tuple(res)
        while True:
            self.levels.append(_Level(r, device))
            if min(r) // 2 < min_dim or any(n % 2 for n in r):
                break
            r = tuple(n // 2 for n in r)
        self.nlev = len(self.levels)

        self.ncells = res[0] * res[1] * res[2]
        self._scalar = wp.zeros(1, dtype=float, device=device)
        # CG work vectors, kept separate from the V-cycle's level-0 buffers
        # (``_Level.p/b/r``) which the preconditioner overwrites on every call
        self.rhs = wp.zeros(res, dtype=float, device=device)
        self.res_v = wp.zeros(res, dtype=float, device=device)
        self.z = wp.zeros(res, dtype=float, device=device)
        self.q = wp.zeros(res, dtype=float, device=device)
        self.Ap = wp.zeros(res, dtype=float, device=device)
        self.r_prev = wp.zeros(res, dtype=float, device=device)

    # ---------------------------------------------------------------- reductions
    def _reduce(self, kernel, *arrays, init=0.0):
        self._scalar.fill_(init)
        wp.launch(kernel, dim=arrays[0].shape, inputs=[*arrays, self._scalar], device=self.device)
        return float(self._scalar.numpy()[0])

    def _dot(self, a, b):
        return self._reduce(dot_kernel, a, b)

    def _mean(self, a):
        return self._reduce(sum_kernel, a) / self.ncells

    def _max_abs(self, a):
        return self._reduce(max_abs_kernel, a)

    def _remove_mean(self, a):
        wp.launch(shift_kernel, dim=a.shape, inputs=[a, -self._mean(a)], device=self.device)

    # ---------------------------------------------------------------- hierarchy
    def build_hierarchy(self, bx, by, bz):
        """Fill level 0 coefficients and Galerkin-coarsen them all the way down."""
        L0 = self.levels[0]
        wp.copy(L0.bx, bx)
        wp.copy(L0.by, by)
        wp.copy(L0.bz, bz)
        for lv in range(self.nlev):
            L = self.levels[lv]
            wp.launch(compute_diag, dim=L.res, inputs=[L.bx, L.by, L.bz, L.diag], device=self.device)
            if lv + 1 < self.nlev:
                C = self.levels[lv + 1]
                for axis, (f, c) in enumerate(((L.bx, C.bx), (L.by, C.by), (L.bz, C.bz))):
                    wp.launch(
                        galerkin_coarsen_faces, dim=c.shape, inputs=[f, c, axis], device=self.device
                    )

    # ---------------------------------------------------------------- smoothing
    def _smooth(self, L, nsweeps, reverse=False):
        colors = (1, 0) if reverse else (0, 1)
        for _ in range(nsweeps):
            for color in colors:
                wp.launch(
                    rbgs_sweep,
                    dim=L.res,
                    inputs=[L.p, L.b, L.bx, L.by, L.bz, L.diag, color, self.omega],
                    device=self.device,
                )

    def _vcycle(self, lv=0):
        L = self.levels[lv]
        if lv == self.nlev - 1:
            self._smooth(L, self.coarsest)
            self._remove_mean(L.p)
            return
        self._smooth(L, self.pre)
        wp.launch(
            residual,
            dim=L.res,
            inputs=[L.p, L.b, L.bx, L.by, L.bz, L.diag, L.r],
            device=self.device,
        )
        C = self.levels[lv + 1]
        wp.launch(restrict_sum, dim=C.res, inputs=[L.r, C.b], device=self.device)
        C.p.zero_()
        self._vcycle(lv + 1)
        wp.launch(prolong_inject_add, dim=L.res, inputs=[C.p, L.p], device=self.device)
        # post-smoothing with reversed colour order keeps the cycle (nearly)
        # symmetric, which is what the flexible CG of paper 6.4 tolerates
        self._smooth(L, self.post, reverse=True)

    def precondition(self, r_in, z_out):
        """Apply one V-cycle as ``z = M^-1 r``."""
        L0 = self.levels[0]
        wp.copy(L0.b, r_in)
        L0.p.zero_()
        self._vcycle(0)
        wp.copy(z_out, L0.p)
        self._remove_mean(z_out)

    # ---------------------------------------------------------------- solve
    def solve(self, b, p_out, *, tol=1e-4, max_iter=200, verbose=False):
        """Flexible PCG with Polak-Ribiere beta (paper 6.4).

        Returns ``(iterations, relative_residual)``.
        """
        dev = self.device
        res = self.levels[0].res

        wp.copy(self.rhs, b)
        self._remove_mean(self.rhs)  # compatibility: pure-Neumann system
        b_norm = self._max_abs(self.rhs)
        if b_norm == 0.0:
            p_out.zero_()
            return 0, 0.0

        x = p_out
        x.zero_()
        wp.copy(self.res_v, self.rhs)

        # keep the operator for apply_A: hierarchy level-0 coefficients
        L0 = self.levels[0]
        bx, by, bz, diag = L0.bx, L0.by, L0.bz, L0.diag

        self.precondition(self.res_v, self.z)
        wp.copy(self.q, self.z)  # search direction
        rz = self._dot(self.res_v, self.z)

        it = 0
        rel = 1.0
        for it in range(1, max_iter + 1):
            wp.launch(apply_A, dim=res, inputs=[self.q, bx, by, bz, diag, self.Ap], device=dev)
            qAq = self._dot(self.q, self.Ap)
            if qAq <= 0.0:
                break
            alpha = rz / qAq
            wp.launch(axpy, dim=res, inputs=[alpha, self.q, x], device=dev)
            wp.copy(self.r_prev, self.res_v)
            wp.launch(axpy, dim=res, inputs=[-alpha, self.Ap, self.res_v], device=dev)

            rel = self._max_abs(self.res_v) / b_norm
            if verbose:
                print(f"    pcg it={it:3d}  |r|inf/|b|inf = {rel:.3e}")
            if rel < tol:
                break

            self.precondition(self.res_v, self.z)
            # Polak-Ribiere: beta = <z, r - r_prev> / <z_prev, r_prev>
            rz_new = self._dot(self.res_v, self.z)
            num = rz_new - self._dot(self.r_prev, self.z)
            beta = max(num / rz, 0.0) if rz != 0.0 else 0.0
            wp.launch(scale_add, dim=res, inputs=[self.q, beta, self.z], device=dev)
            rz = rz_new

        self._remove_mean(x)
        return it, rel
