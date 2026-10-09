#!/usr/bin/env python3
"""Render a frame sequence in parallel across GPUs and encode it to video.

Frames are independent, so they shard cleanly: each worker process claims
``i % nworkers == rank`` and renders only those.  With 8 H100s this turns a
ten-minute sequence render into about a minute.

    python scripts/render_sequence.py out/anim --fps 24 --gpus 8
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def render_shard(files, outdir, rank, device, opts):
    """Worker: render every file whose index matches this rank."""
    import numpy as np
    import warp as wp

    from pfflip import render
    from pfflip.config import Config

    render.orbit_camera.__defaults__ = (tuple(opts["target"]),)

    class _Frame:
        def __init__(self, path):
            d = np.load(path)
            phi = d["phi"].astype(np.float32)
            self.dev = device
            self.dx_spray = float(d["dxs"])
            self.time = float(d["t"])
            self.phi_cell = wp.array(phi, dtype=float, device=device)
            self._spray = wp.array(d["spray"].astype(np.float16), dtype=wp.float16,
                                   device=device)
            self.cfg = Config(res=phi.shape, dx=float(d["dx"]), device=device)

            sp = os.path.join(os.path.dirname(path) or ".", "solid.npz")
            self.solid = None
            if os.path.exists(sp):
                self.solid = wp.array(np.load(sp)["solid"], dtype=float, device=device)

        def rasterize_spray(self):
            return self._spray

    for i, f in enumerate(files):
        if i % opts["nworkers"] != rank:
            continue
        out = os.path.join(outdir, f"img{i:04d}.png")
        render.render(
            _Frame(f), out, width=opts["width"], height=opts["height"],
            azimuth=opts["azimuth"], elevation=opts["elevation"],
            distance=opts["distance"], fov=opts["fov"], spp=opts["spp"],
            steps=opts["steps"], preset=opts["preset"], exposure=opts["exposure"],
            sun_dir=opts["sun"], seed=i,
        )
        print(f"[gpu{rank}] {out} t={_Frame(f).time:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--preset", default="golden")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--azimuth", type=float, default=46.0)
    ap.add_argument("--elevation", type=float, default=11.0)
    ap.add_argument("--distance", type=float, default=0.92)
    ap.add_argument("--target", type=float, nargs=3, default=(0.5, 0.32, 0.5),
                    help="orbit target as a fraction of the domain")
    ap.add_argument("--fov", type=float, default=42.0)
    ap.add_argument("--spp", type=int, default=2)
    ap.add_argument("--steps", type=int, default=640)
    ap.add_argument("--exposure", type=float, default=1.0)
    ap.add_argument("--sun", type=float, nargs=3, default=(-0.62, 0.34, -0.66))
    ap.add_argument("--rank", type=int, default=None, help="internal: worker id")
    args = ap.parse_args()

    # solid.npz holds the static obstacle mask, not a frame
    files = [f for f in sorted(glob.glob(os.path.join(args.indir, "*.npz")))
             if os.path.basename(f) != "solid.npz"]
    if not files:
        sys.exit(f"no .npz frames in {args.indir}")
    outdir = args.outdir or os.path.join(args.indir, "png")
    opts = dict(
        nworkers=args.gpus, width=args.width, height=args.height,
        azimuth=args.azimuth, elevation=args.elevation, distance=args.distance,
        target=tuple(args.target), fov=args.fov, spp=args.spp, steps=args.steps,
        preset=args.preset, exposure=args.exposure, sun=tuple(args.sun),
    )

    if args.rank is not None:
        render_shard(files, outdir, args.rank, f"cuda:{args.rank}", opts)
        return

    os.makedirs(outdir, exist_ok=True)
    print(f"{len(files)} frames -> {outdir} on {args.gpus} GPUs", flush=True)
    procs = [
        subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--rank", str(r)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        for r in range(args.gpus)
    ]
    for p in procs:
        p.wait()

    made = sorted(glob.glob(os.path.join(outdir, "img*.png")))
    print(f"rendered {len(made)}/{len(files)} frames", flush=True)
    if len(made) != len(files):
        sys.exit("some frames failed to render")

    video = args.video or os.path.join(args.indir, "sequence.mp4")
    if shutil.which("ffmpeg"):
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
            "-i", os.path.join(outdir, "img%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", video,
        ], check=True)
        print(f"wrote {video}", flush=True)


if __name__ == "__main__":
    main()
