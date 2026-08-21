#!/usr/bin/env python3
"""Generate the 360° rangefinder sites + sensors and inject them into cubot.xml.

24 azimuths (15° apart) x 3 elevations (-15°, -25°, -35°) = 72 rangefinders
mounted on a ring around the box_container body. Each sensor measures along its
site's +Z axis (the `zaxis` attribute); raw distances are later converted to a
robot-centric local height map by height_map.py. Re-run this script to regenerate
the block.
"""

import os
import numpy as np

N_AZ = 24
ELEV_DEG = [-15.0, -25.0, -35.0]   # elev index: 0 = -15° far, 1 = -25° mid, 2 = -35° near

# The sites sit on a ring around the box_container (world z ~= 0.3068 m), above
# the legs, so the rays are never clipped by the 6 legs during normal walking.
# Each direction's 3 elevations share the same ring position (MuJoCo needs one
# site per ray direction, so the 3 co-located sites act as one shared mount).
RING_RADIUS = 0.12
RING_Z = 0.08
SITE_SIZE = 0.001

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(SCRIPT_DIR, "..", "models", "cubot.xml")


def _fmt(v):
    """Format a float with trailing-zero trimming for compact XML."""
    s = f"{v:.6f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def build_site_lines():
    """Return the 72 <site> lines unindented (azimuth outer, elevation inner)."""
    lines = []
    for az_i in range(N_AZ):
        th = np.deg2rad(az_i * 15.0)
        px = RING_RADIUS * np.cos(th)
        py = RING_RADIUS * np.sin(th)
        for ev_i, ev in enumerate(ELEV_DEG):
            ev_rad = np.deg2rad(ev)
            ce = np.cos(ev_rad)   # horizontal-projection scale
            dz = np.sin(ev_rad)   # vertical component (negative = downward)
            dx = np.cos(th) * ce
            dy = np.sin(th) * ce
            name = f"rf_{az_i}_{ev_i}"
            pos = f"{_fmt(px)} {_fmt(py)} {_fmt(RING_Z)}"
            zaxis = f"{_fmt(dx)} {_fmt(dy)} {_fmt(dz)}"
            lines.append(
                f'<site name="{name}" pos="{pos}" '
                f'zaxis="{zaxis}" size="{SITE_SIZE}"/>'
            )
    return lines


def build_sensor_lines():
    """Return the 72 <rangefinder> lines in the same order as the sites."""
    return [
        f'    <rangefinder name="rf_{az_i}_{ev_i}" site="rf_{az_i}_{ev_i}"/>'
        for az_i in range(N_AZ)
        for ev_i in range(len(ELEV_DEG))
    ]


def main():
    with open(XML_PATH) as f:
        xml = f.read()

    if "rf_0_0" in xml:
        print(f"Skipped: {XML_PATH} already contains rangefinder blocks.")
        return

    # Insert the 72 sites just before the first lid body (inside box_container).
    # The anchor already carries its 10-space indent, so the first site line is
    # left unindented and subsequent lines add the indent explicitly.
    site_anchor = '<body name="lid_rear"'
    assert xml.count(site_anchor) == 1, "site anchor not unique"
    site_lines = build_site_lines()
    sites = site_lines[0] + "\n" + "".join(
        "          " + ln + "\n" for ln in site_lines[1:]
    ) + "          "
    xml = xml.replace(site_anchor, sites + site_anchor, 1)

    # Insert the 72 rangefinder sensors right after </actuator>.
    act_anchor = "  </actuator>"
    assert xml.count(act_anchor) == 1, "actuator anchor not unique"
    sensor_block = (
        "  <sensor>\n" + "\n".join(build_sensor_lines()) + "\n  </sensor>\n"
    )
    xml = xml.replace(act_anchor, act_anchor + "\n" + sensor_block, 1)

    with open(XML_PATH, "w") as f:
        f.write(xml)

    print(f"Wrote {N_AZ * len(ELEV_DEG)} sites + sensors to {XML_PATH}")


if __name__ == "__main__":
    main()
