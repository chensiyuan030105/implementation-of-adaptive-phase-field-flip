#!/usr/bin/env python3
"""Re-render saved field dumps without re-simulating.

Frames written by a run script store the cell-centred phase field and the
high-resolution spray density; everything the renderer needs.  Decoupling the
two lets camera, lighting and look be iterated at render cost only.

    python scripts/rerender.py out/tower/fld07.npz --preset golden --elevation 4
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pfflip import render  # noqa: E402
from pfflip.config import Config  # noqa: E402


class _Frame:
    """Minimal stand-in exposing the attributes :func:`pfflip.render.render` uses."""

    def __init__(self, path, device="cuda:0"):
        d = np.load(path)
        phi = d["phi"].astype(np.float32)
        spray = d["spray"].astype(np.float32)
        self.dev = device
        self.dx_spray = float(d["dxs"])
        self.time = float(d["t"])
        self.phi_cell = wp.array(phi, dtype=float, device=device)
        self._spray16 = wp.array(spray.astype(np.float16), dtype=wp.float16, device=device)
        self.cfg = Config(res=phi.shape, dx=float(d["dx"]), device=device)

    def rasterize_spray(self):
        return self._spray16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--out", default=None, help="output png (single frame) or directory")
    ap.add_argument("--preset", default="overcast", choices=["overcast", "golden", "storm"])
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--azimuth", type=float, default=38.0)
    ap.add_argument("--elevation", type=float, default=6.0)
    ap.add_argument("--distance", type=float, default=0.62)
    ap.add_argument("--fov", type=float, default=42.0)
    ap.add_argument("--spp", type=int, default=3)
    ap.add_argument("--steps", type=int, default=760)
    ap.add_argument("--k-spray", type=float, default=130.0)
    ap.add_argument("--exposure", type=float, default=1.0)
    ap.add_argument("--target-y", type=float, default=0.26)
    ap.add_argument("--sun", type=float, nargs=3, default=(-0.62, 0.34, -0.66))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # the renderer's orbit target is a fraction of the domain; expose the
    # vertical part so the camera can be dropped to just above the water
    render.orbit_camera.__defaults__ = ((0.5, args.target_y, 0.5),)

    for i, f in enumerate(args.frames):
        fr = _Frame(f, args.device)
        if args.out and args.out.endswith(".png"):
            out = args.out
        else:
            base = os.path.splitext(os.path.basename(f))[0]
            out = os.path.join(args.out or os.path.dirname(f) or ".", f"{base}_{args.preset}.png")
        render.render(
            fr, out, width=args.width, height=args.height, azimuth=args.azimuth,
            elevation=args.elevation, distance=args.distance, fov=args.fov,
            spp=args.spp, steps=args.steps, k_spray=args.k_spray,
            preset=args.preset, exposure=args.exposure, sun_dir=tuple(args.sun),
            seed=i,
        )
        print(f"{out}   t={fr.time:.3f}s", flush=True)


if __name__ == "__main__":
    main()
