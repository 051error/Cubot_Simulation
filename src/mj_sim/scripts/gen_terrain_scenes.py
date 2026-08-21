#!/usr/bin/env python3
"""Generate the two near-spawn RL terrain scenes (hill + stairs).

Obstacles start at x=0.25 because the foot tips reach x~0.2 at spawn, so the
robot steps onto terrain within a 4 s / 200-step episode. The hill reuses the
discrete-pillar idea from gen_terrain_hills.py (flat up-facing tops, buried so
only 0~3 cm sticks out); the stairs are three 3 cm steps reused from terrain.xml.
scene.xml is the flat terrain and needs no generation.
"""

import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")

FRICTION = "1.8 0.5 0.5"
CONDIM = 4


def _scene_header(model_name, center, extent):
    """Return the shared scene scaffolding up to the <worldbody> open tag.

    model_name  MJCF model attribute
    center      <statistic> center string, e.g. "0.8 0 0.15"
    extent      <statistic> extent string, e.g. "1.5"
    """
    return f'''<mujoco model="{model_name}">
  <include file="cubot.xml"/>

  <option gravity="0 0 -9.81" timestep="0.005"/>

  <statistic center="{center}" extent="{extent}"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/>
    <global azimuth="130" elevation="-25"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="512"/>
    <texture name="groundplane" type="2d" builtin="checker" mark="edge"
             rgb1="0.2 0.2 0.2" rgb2="0.35 0.35 0.35"
             markrgb="0.5 0.5 0.5" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
             texrepeat="8 8" reflectance="0.15"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true" castshadow="false"/>
'''


def _ground_line():
    """Return the shared flat-ground geom line (spawn/warmup surface)."""
    return (f'    <geom name="flat_ground" type="plane" size="0 0 0.05" '
            f'material="groundplane" friction="{FRICTION}" condim="{CONDIM}"/>')


def _pillar_lines(rng, x0, x1, y0, y1, spacing, half_xy, half_z, h_max, ramp):
    """Return geom lines for a grid of discrete pillars.

    rng      seeded numpy RNG
    x0,x1    pillar x extent (m)
    y0,y1    pillar y extent (m)
    spacing  pillar center spacing (m), < 2*half_xy -> overlap, no gaps
    half_xy  pillar half-size in x/y (m)
    half_z   pillar half-height (m); tops end up 0..h_max above ground
    h_max    max top height above ground (m)
    ramp     entry/exit blend length (m) where heights ramp from 0
    """
    xs = np.arange(x0 + spacing / 2, x1, spacing)
    ys = np.arange(y0 + spacing / 2, y1, spacing)
    lines = []
    k = 0
    for y in ys:
        for x in xs:
            h = rng.uniform(0.0, h_max)
            r = 1.0
            if x < x0 + ramp:
                r = min(r, (x - x0) / ramp)
            if x > x1 - ramp:
                r = min(r, (x1 - x) / ramp)
            h *= r
            z = h - half_z
            grey = rng.uniform(0.45, 0.65)
            lines.append(
                f'    <geom name="hill_{k}" type="box" '
                f'size="{half_xy} {half_xy} {half_z}" '
                f'pos="{x:.4f} {y:.4f} {z:.4f}" '
                f'rgba="{grey:.3f} {grey:.3f} {grey:.3f} 1" '
                f'friction="{FRICTION}" condim="{CONDIM}"/>'
            )
            k += 1
    return lines


def _stairs_lines():
    """Return geom lines for three near-spawn steps plus a top platform."""
    rgba = "0.50 0.45 0.40 1"
    platform_rgba = "0.45 0.42 0.38 1"
    return [
        f'    <geom name="step1" type="box" size="0.1 0.5 0.015" pos="0.35 0 0.015" rgba="{rgba}" friction="{FRICTION}" condim="{CONDIM}"/>',
        f'    <geom name="step2" type="box" size="0.1 0.5 0.030" pos="0.55 0 0.030" rgba="{rgba}" friction="{FRICTION}" condim="{CONDIM}"/>',
        f'    <geom name="step3" type="box" size="0.1 0.5 0.045" pos="0.75 0 0.045" rgba="{rgba}" friction="{FRICTION}" condim="{CONDIM}"/>',
        f'    <geom name="platform" type="box" size="0.25 0.5 0.045" pos="1.0 0 0.045" rgba="{platform_rgba}" friction="{FRICTION}" condim="{CONDIM}"/>',
    ]


def main():
    rng = np.random.default_rng(7)

    hill = [_scene_header("cubot_hill", "0.8 0 0.15", "1.5"), _ground_line()]
    hill += _pillar_lines(rng, x0=0.25, x1=1.45, y0=-0.6, y1=0.6,
                          spacing=0.2, half_xy=0.15, half_z=0.1,
                          h_max=0.03, ramp=0.3)
    hill += ["  </worldbody>", "</mujoco>"]

    stairs = [_scene_header("cubot_stairs", "0.7 0 0.15", "1.5"), _ground_line()]
    stairs += _stairs_lines()
    stairs += ["  </worldbody>", "</mujoco>"]

    for name, lines in (("hill.xml", hill), ("stairs.xml", stairs)):
        out = os.path.join(MODELS_DIR, name)
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
