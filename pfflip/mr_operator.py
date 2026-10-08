"""Variable-coefficient Poisson operator on an MSBG (paper 4.2, 6.1).

Discretisation
--------------
Finite volume.  The coupling between two adjacent cells A and B is

    T = beta_face * Area / dist,      Area = min(h_A, h_B)**2,
                                      dist = (h_A + h_B) / 2

and the row of cell A is ``sum_over_faces T * (p_A - p_B)``, which is symmetric
by construction and positive semi-definite.  ``beta_face = 0`` on solid faces.

This reproduces every transition constant the paper quotes, which is the
cross-check that the discretisation is the intended one:

| transition        | this file                       | paper (Alg. 1)        |
|-------------------|---------------------------------|-----------------------|
| fine-fine (lvl l) | ``beta h^2 / h = beta h_l``     | ``s = 2^l``           |
| coarse l <-> fine | ``beta (H/2)^2/(3H/4) = beta H/3``| ``2^l sigma_CF = 2^l/3`` |
| fine l <-> coarse | ``beta h^2/(3h/2) = 2 beta h/3``| ``2^l sigma_FC = 2^l 2/3`` |

So ``Delta_g = 3/4`` (paper 4.2, Losasso et al. 2004) is not a tuned constant:
it is exactly the centre-to-centre distance across a 2:1 jump measured in units
of the *coarse* spacing, ``H/2 + H/4 = 3H/4``.

Under 2:1 grading a cell sees either 1 or 4 neighbours per face direction, so a
coarse cell abutting finer blocks has more than 7 entries in its row.  The
paper's "7-point stencil, 3 face coefficients per cell" describes the *storage*
(each cell stores its three low-side faces); the gather expands the 2x2 patch.

Face ownership: a face is stored by the cell that has it on its **low** side.
Both neighbours read it from the same place, which is what keeps the operator
exactly symmetric across resolution jumps.
"""

from __future__ import annotations

import warp as wp

from .msbg import Cell, Grid, cell_origin_finest, locate


# ---------------------------------------------------------------------------
@wp.func
def _face_coeff(g: Grid, bface: wp.array(dtype=float), a: Cell, b: Cell, axis: int) -> float:
    """beta of the face between A (low side) and B (high side) along ``axis``.

    Whichever cell is finer owns it as a low face; if A is coarser the face
    belongs to B, otherwise B's low face is a *different* face and the shared
    one is A's high face -- which B stores when B is the finer of the two.
    Resolved by always reading the low face of the cell on the high side.
    """
    return bface[b.addr * 3 + axis]


@wp.func
def _step(axis: int) -> wp.vec3i:
    return wp.vec3i(
        wp.where(axis == 0, 1, 0),
        wp.where(axis == 1, 1, 0),
        wp.where(axis == 2, 1, 0),
    )


@wp.func
def apply_row(
    g: Grid,
    p: wp.array(dtype=float),
    bface: wp.array(dtype=float),
    addr: int,
    want_diag: int,
) -> float:
    """``sum_f T_f (p_A - p_B)`` for cell ``addr``; or the diagonal if asked.

    Walks the 6 face directions.  In the low direction the face is this cell's
    own; in the high direction it belongs to the neighbour.  Under 2:1 grading
    the high side may be backed by a 2x2 patch of finer cells, each of which
    contributes its own term.
    """
    o = cell_origin_finest(g, addr)
    la = o[3]
    ext = 1 << la  # this cell spans `ext` finest cells per edge
    hA = g.h0 * float(ext)
    pa = p[addr]

    acc = float(0.0)
    diag = float(0.0)

    for axis in range(3):
        e = _step(axis)
        for side in range(2):
            # finest coordinate just outside this cell on the chosen side
            if side == 0:
                fx = o[0] - e[0]
                fy = o[1] - e[1]
                fz = o[2] - e[2]
            else:
                fx = o[0] + e[0] * ext
                fy = o[1] + e[1] * ext
                fz = o[2] + e[2] * ext

            nb = locate(g, fx, fy, fz)
            if nb.addr < 0:
                continue  # domain wall: no-flux

            if nb.level >= la:
                # neighbour is same size or coarser -> exactly one of them
                hB = g.h0 * float(1 << nb.level)
                area = wp.min(hA, hB)
                area = area * area
                dist = 0.5 * (hA + hB)
                # the face is owned by the cell on the high side of the pair
                if side == 0:
                    beta = bface[addr * 3 + axis]
                else:
                    beta = bface[nb.addr * 3 + axis]
                t = beta * area / dist
                diag += t
                acc += t * (pa - p[nb.addr])
            else:
                # neighbour is finer: a 2x2 patch of sub-cells across this face
                hB = g.h0 * float(1 << nb.level)
                sub = 1 << nb.level
                area = hB * hB
                dist = 0.5 * (hA + hB)
                # sweep the two in-plane axes
                ax1 = (axis + 1) % 3
                ax2 = (axis + 2) % 3
                e1 = _step(ax1)
                e2 = _step(ax2)
                for u in range(2):
                    for v in range(2):
                        du = u * sub
                        dv = v * sub
                        gx = fx + e1[0] * du + e2[0] * dv
                        gy = fy + e1[1] * du + e2[1] * dv
                        gz = fz + e1[2] * du + e2[2] * dv
                        nc = locate(g, gx, gy, gz)
                        if nc.addr < 0:
                            continue
                        if side == 0:
                            # the sub-face is on the high side of the fine cell,
                            # i.e. it is *this* cell's low face -- but this cell
                            # is coarse, so the fine cell owns nothing here;
                            # use this cell's own low face coefficient
                            beta = bface[addr * 3 + axis]
                        else:
                            beta = bface[nc.addr * 3 + axis]
                        t = beta * area / dist
                        diag += t
                        acc += t * (pa - p[nc.addr])

    if want_diag == 1:
        return diag
    return acc


@wp.kernel
def compute_diag(g: Grid, bface: wp.array(dtype=float), diag: wp.array(dtype=float)):
    a = wp.tid()
    diag[a] = apply_row(g, diag, bface, a, 1)


@wp.kernel
def apply_A(
    g: Grid,
    p: wp.array(dtype=float),
    bface: wp.array(dtype=float),
    out: wp.array(dtype=float),
):
    a = wp.tid()
    out[a] = apply_row(g, p, bface, a, 0)


@wp.kernel
def residual(
    g: Grid,
    p: wp.array(dtype=float),
    b: wp.array(dtype=float),
    bface: wp.array(dtype=float),
    r: wp.array(dtype=float),
):
    a = wp.tid()
    r[a] = b[a] - apply_row(g, p, bface, a, 0)


@wp.kernel
def rbgs_sweep(
    g: Grid,
    p: wp.array(dtype=float),
    b: wp.array(dtype=float),
    bface: wp.array(dtype=float),
    diag: wp.array(dtype=float),
    color: int,
    omega: float,
):
    """Red-black weighted Gauss-Seidel, ``omega = 6/7`` (paper 6.3).

    The colour is taken from the cell's finest-level coordinate so that the
    checkerboard stays consistent across resolution jumps.
    """
    a = wp.tid()
    o = cell_origin_finest(g, a)
    if (((o[0] >> o[3]) + (o[1] >> o[3]) + (o[2] >> o[3])) & 1) != color:
        return
    d = diag[a]
    if d <= 0.0:
        return
    p[a] += omega * (b[a] - apply_row(g, p, bface, a, 0)) / d


@wp.kernel
def cell_volume(g: Grid, vol: wp.array(dtype=float)):
    a = wp.tid()
    o = cell_origin_finest(g, a)
    h = g.h0 * float(1 << o[3])
    vol[a] = h * h * h
