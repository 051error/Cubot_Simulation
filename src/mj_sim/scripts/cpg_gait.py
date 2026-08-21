"""Python port of the C++ tripod CPG + foot trajectory (gait.hpp / gait.cpp).

Shared by train_rl.py and rl_inference.py so the RL policy's CPG matches the
NORMAL/TURN modes exactly: identical constants, oscillator dynamics, foot sweep
and body->coxa transform as RobotController (robot_ctrl.cpp).
"""

import numpy as np

from leg_ik import LegIK

N_LEGS = 6

# ── Hopf oscillator constants (identical to C++ TripodGait) ───────────────
DT    = 0.005
ALPHA = 5.0
MU    = 0.06
K_CPG = 2.0

# ── Foot trajectory constants (identical to C++ FootTrajectory) ───────────
STRIDE_X  = 0.05   # fore/aft sweep half-length (m)
STRIDE_Y  = 0.04   # lateral sweep half-length (m)
LIFT      = 0.06   # swing lift height (m)
SPEED_REF = 0.05   # speed (m/s) for full amplitude

# Nominal foot position (body frame) at crouch — gait.cpp NOMINAL_FOOT.
NOMINAL_FOOT = np.array([
    [ 0.2289, -0.1679, -0.1238],  # rf
    [-0.0016, -0.2522, -0.1238],  # rm
    [-0.2311, -0.1657, -0.1238],  # rr
    [ 0.2311,  0.1657, -0.1238],  # lf
    [ 0.0016,  0.2522, -0.1238],  # lm
    [-0.2289,  0.1679, -0.1238],  # lr
])

# Per-leg mounting geometry (body frame): coxa_x, coxa_y, coxa_z, c1_y_x, c1_y_y.
LEG_GEOM = np.array([
    [ 0.1248, -0.06164, 0.001116, -0.707107,  0.707107],  # rf
    [ 0.0,    -0.1034,  0.001116,  0.0,       1.0     ],  # rm
    [-0.1248, -0.06164, 0.001116,  0.707107,  0.707107],  # rr
    [ 0.1248,  0.06164, 0.001116, -0.707107, -0.707107],  # lf
    [ 0.0,     0.1034,  0.001116,  0.0,      -1.0     ],  # lm
    [-0.1248,  0.06164, 0.001116,  0.707107, -0.707107],  # lr
])

# Crouched reference qpos (coxa=0, thigh=-0.7593, tibia=-0.7108 x6).
JOINT_REF = np.array([0.0, -0.7593, -0.7108] * 6)


class TripodGait:
    """Hopf-oscillator tripod CPG (identical to C++ TripodGait)."""

    def __init__(self):
        # Groups A={0,2,4}, B={1,3,5}, pi apart.
        self.x = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        self.y = np.zeros(N_LEGS)

    def reset(self):
        """Re-initialise the oscillators to the tripod start phase."""
        self.x = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        self.y = np.zeros(N_LEGS)

    def step(self, omega, rot_bias=0.0):
        """Advance one Hopf step (DT=0.005); omega cadence, rot_bias turn."""
        # Same-group attract, opposite-group repel.
        if omega < 0.01:
            omega = 0.01
        x, y = self.x, self.y
        in_a = np.array([True, False, True, False, True, False])
        for i in range(N_LEGS):
            w = np.where(in_a == in_a[i], K_CPG, -K_CPG)
            w[i] = 0.0
            coupling = np.sum(w * (x - x[i])) / (N_LEGS - 1)
            omega_i = omega + (rot_bias if i >= 3 else -rot_bias)
            r2 = x[i] * x[i] + y[i] * y[i]
            dx = ALPHA * (MU - r2) * x[i] - omega_i * y[i] + coupling
            dy = ALPHA * (MU - r2) * y[i] + omega_i * x[i]
            x[i] += dx * DT
            y[i] += dy * DT

    def phase(self, i):
        """>0 stance, <0 swing."""
        r = np.hypot(self.x[i], self.y[i])
        return self.x[i] / r if r > 1e-9 else 1.0

    def orthogonal(self, i):
        """Monotonic sweep in [-1, 1]."""
        r = np.hypot(self.x[i], self.y[i])
        return self.y[i] / r if r > 1e-9 else 0.0


class FootTrajectory:
    """Straight-line body-frame foot sweep (identical to C++ FootTrajectory)."""

    def compute(self, leg, phase, orthogonal, vx, vy):
        """Foot target in body frame from CPG phase.

        leg 0..5; phase >0 stance/<0 swing; orthogonal sweep [-1,1];
        vx,vy body-frame velocity (forward +X, left +Y).
        """
        nom = NOMINAL_FOOT[leg]
        speed = np.hypot(vx, vy)

        ph = np.clip(phase, -1.0, 1.0)
        yy = np.clip(orthogonal, -1.0, 1.0)

        # Straight sweep opposite to velocity; yy back-sweeps once per stance.
        sx = 0.0
        sy = 0.0
        if speed > 1e-6:
            g = min(speed / SPEED_REF, 1.0)
            sx = -STRIDE_X * (vx / speed) * g * yy
            sy = -STRIDE_Y * (vy / speed) * g * yy

        fx = nom[0] + sx
        fy = nom[1] + sy
        fz = nom[2]

        if ph < 0.0:
            fz += LIFT * 0.5 * (1.0 - np.cos(np.pi * ph))  # raised-cosine lift

        return fx, fy, fz


def compute_joint_targets(gait, traj, vx, vy):
    """CPG -> foot targets -> body->coxa -> IK, returning 18 joint targets.

    Mirrors RobotController::timer_callback NORMAL branch (robot_ctrl.cpp):
    vx,vy are body-frame velocity; the cadence is omega = 2*pi*speed/0.15 + 4.
    Idle (speed < 0.01) returns the crouched JOINT_REF while the CPG still
    advances slowly, matching the C++ idle branch.
    """
    speed = np.hypot(vx, vy)
    if speed < 0.01:
        gait.step(0.0, 0.0)
        return JOINT_REF.copy()

    omega = 2.0 * np.pi * speed / 0.15 + 4.0
    gait.step(omega, 0.0)

    q = np.empty(18, dtype=np.float32)
    for leg in range(N_LEGS):
        ph = gait.phase(leg)
        yy = gait.orthogonal(leg)
        fx, fy, fz = traj.compute(leg, ph, yy, vx, vy)

        # Body -> coxa frame: c1 +X = body +Z, c1 +Z = (-c1_y_y, c1_y_x, 0).
        g = LEG_GEOM[leg]
        dx = fx - g[0]
        dy = fy - g[1]
        dz = fz - g[2]
        c1x = dz
        c1y = g[3] * dx + g[4] * dy
        c1z = -g[4] * dx + g[3] * dy

        ang = LegIK.solve(c1x, c1y, c1z)
        q[leg * 3: leg * 3 + 3] = ang
    return q
