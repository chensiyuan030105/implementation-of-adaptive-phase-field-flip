"""Frame output and quick-look rendering."""

from __future__ import annotations

import os

import numpy as np


def save_frame(sim, path: str, *, particles: bool = True, stride: int = 1):
    """Dump one frame as a compressed ``.npz``.

    Stores the cell-centred phase field (which is what paper section 7 renders,
    at a higher resolution than the pressure grid) and optionally a strided
    subset of the particles.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "t": np.float32(sim.time),
        "step": np.int32(sim.step_index),
        "dx": np.float32(sim.cfg.dx),
        "phi": sim.phi_cell.numpy().astype(np.float16),
    }
    if particles:
        px = sim.px.numpy()[::stride]
        pt = sim.ptype.numpy()[::stride]
        data["px"] = px.astype(np.float16)
        data["ptype"] = pt.astype(np.int8)
    np.savez_compressed(path, **data)


def render_slices(sim, path: str, *, axis: int = 2, dpi: int = 110):
    """Save a quick-look PNG: phase field slice + particle scatter + speed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phi = sim.phi_cell.numpy()
    nz = phi.shape[axis]
    sl = [slice(None)] * 3
    sl[axis] = nz // 2
    img = phi[tuple(sl)].T  # (y, x) for imshow with origin lower

    px = sim.px.numpy()
    pt = sim.ptype.numpy()
    sz = sim.cfg.domain_size
    half = sz[axis] * 0.5
    band = abs(px[:, axis] - half) < 1.5 * sim.cfg.dx
    other = [a for a in range(3) if a != axis]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    im = axes[0].imshow(
        img, origin="lower", extent=[0, sz[other[0]], 0, sz[other[1]]], vmin=0, vmax=1, cmap="Blues"
    )
    axes[0].set_title(f"phase field $\\phi$   t = {sim.time:.3f}s")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    liq = band & (pt == 1)
    air = band & (pt == 0)
    axes[1].scatter(px[air, other[0]], px[air, other[1]], s=0.3, c="#cccccc", linewidths=0, label="air")
    axes[1].scatter(px[liq, other[0]], px[liq, other[1]], s=0.4, c="#1f6fb4", linewidths=0, label="liquid")
    axes[1].set_xlim(0, sz[other[0]])
    axes[1].set_ylim(0, sz[other[1]])
    axes[1].set_aspect("equal")
    axes[1].set_title(f"particles (slab)   step {sim.step_index}")
    axes[1].legend(loc="upper right", markerscale=12, framealpha=0.9)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
