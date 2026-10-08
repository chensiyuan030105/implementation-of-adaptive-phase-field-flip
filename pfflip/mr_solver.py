"""Matrix-free Krylov solver on an MSBG.

Stage 2 ships a Jacobi-preconditioned CG, which is enough to *validate* the
multiresolution discretisation of :mod:`pfflip.mr_operator` but is not the
paper's solver: section 6's multiresolution Galerkin multigrid with adaptive
relaxation (Algorithms 1 and 2) is stage 3.  Expect Jacobi-CG to need many
iterations at the paper's 1000:1 density contrast -- that is exactly the
ill-conditioning section 6 exists to remove.

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
    def __init__(self, grid, device):
        self.grid = grid
        self.dev = device
        n = grid.ncells
        self.n = n
        z = lambda: wp.zeros(n, dtype=float, device=device)  # noqa: E731
        self.diag = z()
        self.r = z()
        self.zv = z()
        self.q = z()
        self.Ap = z()
        self.b = z()
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
        wp.launch(_jacobi, dim=n, inputs=[self.r, self.diag, self.zv], device=dev)
        self._remove_mean(self.zv)
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
            wp.launch(_axpy, dim=n, inputs=[-alpha, self.Ap, self.r], device=dev)
            rel = self._reduce(_max_abs, self.r) / bnorm
            if verbose and it % 50 == 0:
                print(f"    cg {it:5d}  {rel:.3e}")
            if rel < tol:
                break
            wp.launch(_jacobi, dim=n, inputs=[self.r, self.diag, self.zv], device=dev)
            self._remove_mean(self.zv)
            rz_new = self._reduce(_dot, self.r, self.zv)
            if rz == 0.0 or rz_new <= 0.0:
                # float32 stagnation: the residual can no longer be reduced, so
                # beta is meaningless.  Stop rather than divide by zero.
                break
            wp.launch(_scale_add, dim=n, inputs=[self.q, rz_new / rz, self.zv], device=dev)
            rz = rz_new

        self._remove_mean(p_out)
        return it, rel
