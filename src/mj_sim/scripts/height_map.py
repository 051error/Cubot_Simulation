"""Convert the 72 raw rangefinder distances to a robot-centric local height map.

Shared by train_rl.py and rl_inference.py so the height-map features the policy
sees during training are byte-for-byte identical to what it sees at inference.
"""

import numpy as np

N_AZ = 24                # 24 azimuths, 15° apart, 0° = forward (+X)
N_ELEV = 3               # per azimuth: -15° far, -25° mid, -35° near
RANGEFINDER_DIM = N_AZ * N_ELEV    # 72
HEIGHT_MAP_DIM = N_AZ * N_ELEV     # 72: per azimuth [near(-35°), mid(-25°), far(-15°)]

# sin(elevation) for the three downward rays used to triangulate ground height.
SIN_15 = np.sin(np.deg2rad(15.0))
SIN_25 = np.sin(np.deg2rad(25.0))
SIN_35 = np.sin(np.deg2rad(35.0))

# Sentinel for a ray that hits nothing (MuJoCo rangefinder miss = -1.0).
NO_GROUND_HEIGHT = -1.0   # no ground under the downward ray -> deep drop


def rangefinder_to_height_map(ranges):
    """Convert raw distances to a 72-element robot-centric ground-height map.

    ranges  (72,) float32, order azimuth outer / elevation inner [-15°, -25°, -35°]
    returns height_map (72,) ground height relative to the sensor (negative =
            below), ordered per azimuth [near(-35°), mid(-25°), far(-15°)].
    """
    r = np.asarray(ranges, dtype=np.float32).reshape(N_AZ, N_ELEV)

    dist_15 = r[:, 0]  # -15° ray (far)
    dist_25 = r[:, 1]  # -25° ray (mid)
    dist_35 = r[:, 2]  # -35° ray (near)

    # Ground height below the sensor: h = -dist * sin(elevation).
    h_far = -dist_15 * SIN_15
    h_mid = -dist_25 * SIN_25
    h_near = -dist_35 * SIN_35

    # A miss (negative distance) means no ground along that ray.
    h_near = np.where(dist_35 < 0.0, NO_GROUND_HEIGHT, h_near)
    h_mid = np.where(dist_25 < 0.0, NO_GROUND_HEIGHT, h_mid)
    h_far = np.where(dist_15 < 0.0, NO_GROUND_HEIGHT, h_far)

    height_map = np.stack([h_near, h_mid, h_far], axis=1).ravel()   # (72,)

    return height_map.astype(np.float32)
