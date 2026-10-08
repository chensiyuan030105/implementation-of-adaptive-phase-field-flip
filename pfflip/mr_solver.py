"""Matrix-free Krylov solver on an MSBG (paper section 6).

Flexible preconditioned CG with a **Polak-Ribiere** update (paper 6.4), so a
preconditioner that is only approximately symmetric stays safe.  Two
preconditioners are available:

* ``"mg"`` (default) -- one V-cycle of :class:`pfflip.mr_multigrid.MRMultigrid`,
  the multiresolution multigrid of section 6;
* ``"mg-adaptive"`` -- the same V-cycle with the adaptive relaxation smoother
  of 6.4 (Algorithm 2);
* ``"jacobi"`` -- diagonal scaling, kept for comparison.  At the paper's
  1000:1 density contrast this needs hundreds of iterations, which is exactly
  the ill-conditioning section 6 exists to remove.

The system is singular (pure Neumann, both phases fill the box), so the
constant null vector is projected out of the right-hand side and the solution.
"""

from __future__ import annotations

import warp as wp

from . import mr_operator as mro
from .msbg import Grid


@wp.kernel
def _axpy(a: float, x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    y[i] += a * x[i]


@wp.kernel
def _scale_add(y: wp.array(dtype=float), a: float, x: wp.array(dtype=float)):
    """``y <- x + a y``"""
    i = wp.tid()
    y[i] = x[i] + a * y[i]


@wp.kernel
def _dot(x: wp.array(dtype=float), y: wp.array(dtype=float), out: wp.array(dtype=float)):
    i = wp.tid()
    wp.atomic_add(out, 0, x[i] * y[i])


@wp.kernel
def _sum(x: wp.array(dtype=float), out: wp.array(dtype=float)):
    i = wp.tid()
    wp.atomic_add(out, 0, x[i])


@wp.kernel
def _max_abs(x: wp.array(dtype=float), out: wp.array(dtype=float)):
    i = wp.tid()
    wp.atomic_max(out, 0, wp.abs(x[i]))


@wp.kernel
def _shift(x: wp.array(dtype=float), c: float):
    i = wp.tid()
    x[i] += c


@wp.kernel
def _jacobi(r: wp.array(dtype=float), diag: wp.array(dtype=float), z: wp.array(dtype=float)):
    i = wp.tid()
    d = diag[i]
    z[i] = wp.where(d > 0.0, r[i] / d, 0.0)


class MRSolver:
    def __init__(self, grid, device, preconditioner="mg", **mg_kwargs):
        self.grid = grid
        self.dev = device
        self.kind = preconditioner
        self.mg = None
        if preconditioner in ("mg", "mg-adaptive"):
            from .mr_multigrid import MRMultigrid

            if preconditioner == "mg-adaptive":
                mg_kwargs.setdefault("smoother", "adaptive")
            self.mg = MRMultigrid(grid, device, **mg_kwargs)
        n = grid.ncells
        self.n = n
        z = lambda: wp.zeros(n, dtype=float, device=device)  # noqa: E731
        self.diag = z()
        self.r = z()
        self.zv = z()
        self.q = z()
        self.Ap = z()
        self.b = z()
        self.r_prev = z()
        self._s = wp.zeros(1, dtype=float, device=device)

    # ----------------------------------------------------------------- utils
    def _reduce(self, kernel, *arrays, init=0.0):
        self._s.fill_(init)
        wp.launch(kernel, dim=self.n, inputs=[*arrays, self._s], device=self.dev)
        return float(self._s.numpy()[0])

    def _remove_mean(self, a):
        m = self._reduce(_sum, a) / self.n
        wp.launch(_shift, dim=self.n, inputs=[a, -m], device=self.dev)

    def prepare(self, bface):
        wp.launch(mro.compute_diag, dim=self.n, inputs=[self.grid.g, bface, self.diag], device=self.dev)
        if self.mg is not None:
            self.mg.build_hierarchy(bface)

    def _apply_preconditioner(self, r, z):
        if self.mg is not None:
            self.mg.precondition(r, z)
        else:
            wp.launch(_jacobi, dim=self.n, inputs=[r, self.diag, z], device=self.dev)
            self._remove_mean(z)

    # ----------------------------------------------------------------- solve
    def solve(self, bface, rhs, p_out, *, tol=1e-4, max_iter=2000, verbose=False):
        dev, n, g = self.dev, self.n, self.grid.g
        self.prepare(bface)

        wp.copy(self.b, rhs)
        self._remove_mean(self.b)
        bnorm = self._reduce(_max_abs, self.b)
        if bnorm == 0.0:
            p_out.zero_()
            return 0, 0.0

        p_out.zero_()
        wp.copy(self.r, self.b)
        self._apply_preconditioner(self.r, self.zv)
        wp.copy(self.q, self.zv)
        rz = self._reduce(_dot, self.r, self.zv)

        it, rel = 0, 1.0
        for it in range(1, max_iter + 1):
            wp.launch(mro.apply_A, dim=n, inputs=[g, self.q, bface, self.Ap], device=dev)
            qAq = self._reduce(_dot, self.q, self.Ap)
            if qAq <= 0.0:
                break
            alpha = rz / qAq
            wp.launch(_axpy, dim=n, inputs=[alpha, self.q, p_out], device=dev)
            # snapshot r *before* the update: Polak-Ribiere needs r_{k-1}
            wp.copy(self.r_prev, self.r)
            wp.launch(_axpy, dim=n, inputs=[-alpha, self.Ap, self.r], device=dev)
            rel = self._reduce(_max_abs, self.r) / bnorm
            if verbose and it % 50 == 0:
                print(f"    cg {it:5d}  {rel:.3e}")
            if rel < tol:
                break
            self._apply_preconditioner(self.r, self.zv)
            rz_new = self._reduce(_dot, self.r, self.zv)
            if rz == 0.0 or rz_new <= 0.0:
                # float32 stagnation: the residual can no longer be reduced, so
                # beta is meaningless.  Stop rather than divide by zero.
                break
            # Polak-Ribiere: beta = <z, r - r_prev> / <z_prev, r_prev>.  The
            # re-discretised V-cycle is not exactly symmetric, so the flexible
            # form is required (paper 6.4).
            beta = max((rz_new - self._reduce(_dot, self.r_prev, self.zv)) / rz, 0.0)
            wp.launch(_scale_add, dim=n, inputs=[self.q, beta, self.zv], device=dev)
            rz = rz_new

        self._remove_mean(p_out)
        return it, rel
