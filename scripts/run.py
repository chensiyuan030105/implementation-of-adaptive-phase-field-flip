#!/usr/bin/env python3
"""Run a PF-FLIP scene and write frames.

    python scripts/run.py --scene dam_break --res 64 --frames 120
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pfflip import Config, Simulator, io, scenes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="dam_break", choices=sorted(scenes.SCENES))
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--cfl", type=float, default=1.0)
    ap.add_argument("--ppc", type=int, default=8)
    ap.add_argument("--out", default="out")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--png", action="store_true", help="also write quick-look PNGs")
    ap.add_argument("--npz-stride", type=int, default=4)
    args = ap.parse_args()

    r = args.res
    cfg = Config(
        res=(r, r, r),
        dx=1.0 / r,
        cfl=args.cfl,
        ppc=args.ppc,
        device=args.device,
        dt_max=1.0 / args.fps,
    )
    if args.scene == "rayleigh_taylor":  # paper 8.2
        cfg.alpha_phi, cfg.cfl = 0.5, 0.5

    sim = Simulator(cfg)
    pos, vel, ptype = scenes.SCENES[args.scene](cfg)
    sim.set_particles(pos, vel, ptype)

    outdir = os.path.join(args.out, f"{args.scene}_{r}")
    os.makedirs(outdir, exist_ok=True)
    print(f"scene={args.scene} res={cfg.res} particles={len(pos):,} "
          f"(liquid {100*(ptype==1).mean():.1f}%)  -> {outdir}")

    frame_dt = 1.0 / args.fps
    t_start = time.perf_counter()
    for frame in range(args.frames):
        target = (frame + 1) * frame_dt
        nsub = 0
        while sim.time < target - 1e-9:
            dt = min(sim.dt_from_cfl(), target - sim.time)
            rec = sim.step(dt)
            nsub += 1
        io.save_frame(sim, os.path.join(outdir, f"frame_{frame:04d}.npz"), stride=args.npz_stride)
        if args.png:
            io.render_slices(sim, os.path.join(outdir, f"frame_{frame:04d}.png"))
        print(f"frame {frame:4d}  t={sim.time:6.3f}  substeps={nsub:3d}  "
              f"pcg={rec['pcg_iter']:3d}  div={rec['max_div']:.2e}  "
              f"wall={time.perf_counter()-t_start:7.1f}s", flush=True)

    import json

    with open(os.path.join(outdir, "stats.json"), "w") as f:
        json.dump(sim.stats, f)
    print(f"done in {time.perf_counter()-t_start:.1f}s")


if __name__ == "__main__":
    main()
