#!/usr/bin/env python3
"""Generate the rough-terrain hills as a dense grid of discrete box pillars.

Each pillar is an independent convex box with a flat, up-facing top, so a foot
resting on it sees a single clean contact normal. Pillars overlap (spacing <
side length) to leave no gaps, and their top heights are randomised within the
foot-lift budget so the surface reads as gentle stepping stones instead of a
continuous heightfield (whose multi-cell contact points were stalling the gait).
"""

import os
import numpy as np

X0, X1 = 3.0, 7.0      # x extent (m)
Y0, Y1 = -2.0, 2.0     # y extent (m)
SPACING = 0.2          # pillar center spacing (m), < side length -> overlap
HALF_XY = 0.15         # pillar half-size in x/y (side 0.30 m, > foot ~0.18 m)
HALF_Z = 0.1           # pillar half-height (buried ~0.17 m)
H_MAX = 0.03           # max top height above ground (m)
RAMP = 0.4             # edge ramp length (m) for entry/exit blend

FRICTION = "1.8 0.5 0.5"
CONDIM = 4


def main():
    rng = np.random.default_rng(7)
    xs = np.arange(X0 + SPACING / 2, X1, SPACING)
    ys = np.arange(Y0 + SPACING / 2, Y1, SPACING)

    lines = ['<mujoco model="hills">', "  <worldbody>"]
    k = 0
    for y in ys:
        for x in xs:
            # Random top height, ramped to 0 at the entry/exit edges so the
            # robot walks onto the hills gradually.
            h = rng.uniform(0.0, H_MAX)
            ramp = 1.0
            if x < X0 + RAMP:
                ramp = min(ramp, (x - X0) / RAMP)
            if x > X1 - RAMP:
                ramp = min(ramp, (X1 - x) / RAMP)
            h *= ramp
            z = h - HALF_Z
            grey = rng.uniform(0.45, 0.65)
            lines.append(
                f'    <geom name="hill_{k}" type="box" '
                f'size="{HALF_XY} {HALF_XY} {HALF_Z}" '
                f'pos="{x:.4f} {y:.4f} {z:.4f}" '
                f'rgba="{grey:.3f} {grey:.3f} {grey:.3f} 1" '
                f'friction="{FRICTION}" condim="{CONDIM}"/>'
            )
            k += 1
    lines.append("  </worldbody>")
    lines.append("</mujoco>")

    out = os.path.join(os.path.dirname(__file__), "..", "models", "hills.xml")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.abspath(out)}: {k} pillars ({len(xs)}x{len(ys)})")


if __name__ == "__main__":
    main()
