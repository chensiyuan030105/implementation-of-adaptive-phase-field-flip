"""Validate the variable-coefficient Poisson solver against a direct solve.

Checks, on a grid small enough to assemble explicitly:
  1. the matrix-free operator ``apply_A`` matches the assembled 7-point matrix;
  2. the assembled matrix is symmetric and has zero row sums (pure Neumann);
  3. MG-PCG reaches the requested tolerance and agrees with a direct solve up
     to the constant null space, including at a 1000:1 coefficient contrast
     (the regime paper 6 is about).
"""

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import warp as wp

from pfflip.poisson import MGPCG, apply_A

DEV = "cuda:0" if wp.get_cuda_device_count() else "cpu"
RES = (16, 16, 16)


def random_beta(res, rng, contrast=1.0):
    """Face coefficients with zero on the domain boundary faces."""
    nx, ny, nz = res
    bx = rng.random((nx + 1, ny, nz)).astype(np.float32)
    by = rng.random((nx, ny + 1, nz)).astype(np.float32)
    bz = rng.random((nx, ny, nz + 1)).astype(np.float32)
    if contrast != 1.0:
        # a slab of "heavy" cells -> beta small, mimicking the liquid phase
        for arr in (bx, by, bz):
            arr[:, : arr.shape[1] // 2, :] /= contrast
    bx[0] = 0.0
    bx[-1] = 0.0
    by[:, 0] = 0.0
    by[:, -1] = 0.0
    bz[:, :, 0] = 0.0
    bz[:, :, -1] = 0.0
    return bx, by, bz


def assemble(res, bx, by, bz):
    """Assemble ``A p_i = sum_f beta_f (p_i - p_nbr)`` as a sparse matrix."""
    nx, ny, nz = res
    idx = np.arange(nx * ny * nz).reshape(res)
    rows, cols, vals = [], [], []
    diag = np.zeros(res, np.float64)

    def add(bface, axis):
        nonlocal rows, cols, vals
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_lo[axis] = slice(0, res[axis] - 1)
        sl_hi[axis] = slice(1, res[axis])
        # interior faces of this axis have index 1 .. res[axis]-1
        sl_f = [slice(None)] * 3
        sl_f[axis] = slice(1, res[axis])
        b = bface[tuple(sl_f)].astype(np.float64)
        a_id = idx[tuple(sl_lo)].ravel()
        b_id = idx[tuple(sl_hi)].ravel()
        bb = b.ravel()
        rows.extend(a_id)
        cols.extend(b_id)
        vals.extend(-bb)
        rows.extend(b_id)
        cols.extend(a_id)
        vals.extend(-bb)
        diag[tuple(sl_lo)] += b
        diag[tuple(sl_hi)] += b

    add(bx, 0)
    add(by, 1)
    add(bz, 2)
    rows.extend(idx.ravel())
    cols.extend(idx.ravel())
    vals.extend(diag.ravel())
    n = nx * ny * nz
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


@pytest.fixture(scope="module")
def _wp():
    wp.init()
    return wp


@pytest.mark.parametrize("contrast", [1.0, 1000.0])
def test_operator_matches_assembled(_wp, contrast):
    rng = np.random.default_rng(0)
    bx, by, bz = random_beta(RES, rng, contrast)
    A = assemble(RES, bx, by, bz)

    assert abs(A - A.T).max() < 1e-12, "operator must be symmetric"
    assert np.abs(np.asarray(A.sum(axis=1)).ravel()).max() < 1e-9, "row sums must vanish"

    solver = MGPCG(RES, DEV)
    solver.build_hierarchy(
        wp.array(bx, dtype=float, device=DEV),
        wp.array(by, dtype=float, device=DEV),
        wp.array(bz, dtype=float, device=DEV),
    )
    p = rng.random(RES).astype(np.float32)
    pw = wp.array(p, dtype=float, device=DEV)
    out = wp.zeros(RES, dtype=float, device=DEV)
    L0 = solver.levels[0]
    wp.launch(apply_A, dim=RES, inputs=[pw, L0.bx, L0.by, L0.bz, L0.diag, out], device=DEV)

    ref = (A @ p.astype(np.float64).ravel()).reshape(RES)
    got = out.numpy()
    assert np.abs(got - ref).max() < 1e-3 * max(1.0, np.abs(ref).max())


@pytest.mark.parametrize("contrast", [1.0, 1000.0])
def test_mgpcg_solves(_wp, contrast):
    rng = np.random.default_rng(1)
    bx, by, bz = random_beta(RES, rng, contrast)
    A = assemble(RES, bx, by, bz)

    b = rng.standard_normal(RES).astype(np.float64)
    b -= b.mean()  # compatibility condition

    # direct solve with the null space pinned by a Lagrange-style shift
    n = A.shape[0]
    Areg = A + sp.eye(n) * 1e-10
    ref = spla.spsolve(Areg.tocsc(), b.ravel())
    ref -= ref.mean()

    solver = MGPCG(RES, DEV)
    solver.build_hierarchy(
        wp.array(bx, dtype=float, device=DEV),
        wp.array(by, dtype=float, device=DEV),
        wp.array(bz, dtype=float, device=DEV),
    )
    bw = wp.array(b.astype(np.float32), dtype=float, device=DEV)
    pw = wp.zeros(RES, dtype=float, device=DEV)
    # 1e-5 is the float32 floor here: the reductions use atomics, so both the
    # residual and its ordering vary run to run.  Still 10x tighter than the
    # 1e-4 the paper uses for all production examples (8.3).
    it, rel = solver.solve(bw, pw, tol=1e-5, max_iter=400)

    assert rel < 1e-5, f"MG-PCG stalled at {rel:.2e} after {it} iterations"
    got = pw.numpy().astype(np.float64).ravel()
    got -= got.mean()
    err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30)
    assert err < 1e-3, f"solution mismatch {err:.2e} (contrast {contrast})"
