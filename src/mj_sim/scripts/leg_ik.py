"""Analytic IK for the 3-DOF hexapod leg — port of C++ LegIK (ik_solver.cpp).

The forward-kinematics helpers (_leg_fk / _leg_fk_batch) are kept for
analysis_workspace_velocity.py; they are independent of the analytic IK below.
"""

import numpy as np


# ══════════════════════════════════════════════════════════════════════════
# Analytic IK (identical to C++ LegIK in ik_solver.cpp)
# ══════════════════════════════════════════════════════════════════════════

L0 = 0.054        # coxa link length (m)
L1 = 0.0661       # femur link length (m)
L2 = 0.1632       # tibia link, joint->foot tip (m)
A10 = -1.791926   # thigh dir at thigh=0 (rad)
OFFSET = 1.163566 # thigh<->tibia straight angle (rad)


class LegIK:
    """Analytic 3-DOF leg IK in the coxa frame (+X up, +Y outward, +Z fore/aft)."""

    @staticmethod
    def solve(x, y, z):
        """Solve IK for one leg.

        x,y,z  foot target in coxa frame
        returns {coxa, thigh, tibia} qpos (rad)
        """
        angles = np.empty(3, dtype=np.float32)
        angles[0] = np.arctan2(-z, -y)   # coxa yaw from horizontal direction

        # Planar 2-link (thigh L1 + tibia L2) in the X-Y plane.
        r = np.hypot(y, z)
        dX = x
        dY = L0 - r
        D = np.hypot(dX, dY)

        cos_q2 = np.clip(
            (D * D - L1 * L1 - L2 * L2) / (2.0 * L1 * L2), -1.0, 1.0)
        q2 = -np.arccos(cos_q2)   # negative = knee bent back

        q1 = np.arctan2(dY, dX) \
           - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))

        # Map planar angles back to MuJoCo qpos.
        angles[1] = A10 - q1
        angles[2] = q2 + OFFSET
        return angles

    @staticmethod
    def is_reachable(x, y, z):
        """True if the foot target lies within the leg's reach envelope."""
        r = np.hypot(y, z)
        D = np.hypot(x, L0 - r)
        return bool(abs(L1 - L2) <= D <= L1 + L2)

    @staticmethod
    def clear_cache():
        pass  # analytic IK is stateless


# ══════════════════════════════════════════════════════════════════════════
# Forward kinematics (kept for analysis_workspace_velocity.py)
# ══════════════════════════════════════════════════════════════════════════

def _RotX(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _quat_to_R(qw, qx, qy, qz):
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qw*qz, 2*qx*qz + 2*qw*qy],
        [2*qx*qy + 2*qw*qz, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qw*qx],
        [2*qx*qz - 2*qw*qy, 2*qy*qz + 2*qw*qx, 1 - 2*qx**2 - 2*qy**2],
    ])

# ── Body XML parameters (from cubot.xml) ──────────────────────────────────
# c1_rf body
_p_c1  = np.array([0.1248, -0.06164, 0.001116])
_R_c1  = _quat_to_R(-0.653252, -0.27061, 0.653311, -0.270586)
# c2_rf body (child of c1_rf, NO joint)
_p_c2  = np.array([0.0, -0.054, 0.0])
_R_c2  = _quat_to_R(9.38372e-07, -0.706967, 9.38e-07, 0.707247)
# thigh_rf body (child of c2_rf, j_thigh_rf)
_p_th  = np.array([0.0, 0.0, 0.0])
_R_th  = _quat_to_R(1.76038e-12, -1.0, 1.32679e-06, 1.32679e-06)
# tibia_rf body (child of thigh_rf, j_tibia_rf)
_p_tb  = np.array([0.0, -0.0645, -0.0145])
_R_tb  = _quat_to_R(9.38231e-07, -9.3814e-07, -0.707073, 0.707141)

# Foot tip in tibia local frame (lowest mesh vertex).
_FOOT_TIP = np.array([0.00162782, 0.16052104, 0.02951023])


def _leg_fk(theta0, theta1, theta2):
    """FK: (theta0,theta1,theta2) -> foot position in c1_rest frame.

    Body hierarchy (parent->child):
      MP_BODY -> c1_rf (j_c1_rf, POST) -> c2_rf (fixed)
              -> thigh_rf (j_thigh_rf, PRE) -> tibia_rf (j_tibia_rf, POST)
    """
    R_c1_mp = _R_c1 @ _RotX(theta0)
    p_c1_mp = _p_c1

    R_c2_c1 = _R_c2
    p_c2_c1 = _p_c2
    R_c2_mp = R_c1_mp @ R_c2_c1
    p_c2_mp = p_c1_mp + R_c1_mp @ p_c2_c1

    R_th_c2 = _RotX(theta1) @ _R_th
    p_th_c2 = _RotX(theta1) @ _p_th
    R_th_mp = R_c2_mp @ R_th_c2
    p_th_mp = p_c2_mp + R_c2_mp @ p_th_c2

    R_tb_th = _R_tb @ _RotX(theta2)
    p_tb_th = _p_tb
    R_tb_mp = R_th_mp @ R_tb_th
    p_tb_mp = p_th_mp + R_th_mp @ p_tb_th

    p_foot_mp = p_tb_mp + R_tb_mp @ _FOOT_TIP
    return _R_c1.T @ (p_foot_mp - _p_c1)


def _leg_fk_batch(t0, t1, t2):
    """Vectorized FK: (N,) joint arrays -> (N,3) foot positions in c1_rest."""
    N = len(t0)
    c0, s0 = np.cos(t0), np.sin(t0)
    c1, s1 = np.cos(t1), np.sin(t1)
    c2, s2 = np.cos(t2), np.sin(t2)

    rc1_00 = _R_c1[0, 0] * np.ones(N)
    rc1_01 = _R_c1[0, 1] * c0 + _R_c1[0, 2] * s0
    rc1_02 = -_R_c1[0, 1] * s0 + _R_c1[0, 2] * c0
    rc1_10 = _R_c1[1, 0] * np.ones(N)
    rc1_11 = _R_c1[1, 1] * c0 + _R_c1[1, 2] * s0
    rc1_12 = -_R_c1[1, 1] * s0 + _R_c1[1, 2] * c0
    rc1_20 = _R_c1[2, 0] * np.ones(N)
    rc1_21 = _R_c1[2, 1] * c0 + _R_c1[2, 2] * s0
    rc1_22 = -_R_c1[2, 1] * s0 + _R_c1[2, 2] * c0

    R_c1_mp = np.empty((N, 3, 3))
    R_c1_mp[:, 0, 0] = rc1_00; R_c1_mp[:, 0, 1] = rc1_01; R_c1_mp[:, 0, 2] = rc1_02
    R_c1_mp[:, 1, 0] = rc1_10; R_c1_mp[:, 1, 1] = rc1_11; R_c1_mp[:, 1, 2] = rc1_12
    R_c1_mp[:, 2, 0] = rc1_20; R_c1_mp[:, 2, 1] = rc1_21; R_c1_mp[:, 2, 2] = rc1_22

    p_c1_mp = np.tile(_p_c1, (N, 1))

    R_c2_mp = R_c1_mp @ _R_c2
    p_c2_mp = p_c1_mp + np.einsum('nij,j->ni', R_c1_mp, _p_c2)

    Rx1 = np.empty((N, 3, 3))
    Rx1[:, 0, 0] = 1;  Rx1[:, 0, 1] = 0;   Rx1[:, 0, 2] = 0
    Rx1[:, 1, 0] = 0;  Rx1[:, 1, 1] = c1;   Rx1[:, 1, 2] = -s1
    Rx1[:, 2, 0] = 0;  Rx1[:, 2, 1] = s1;   Rx1[:, 2, 2] = c1
    R_th_c2 = Rx1 @ _R_th
    p_th_c2 = np.einsum('nij,j->ni', Rx1, _p_th)

    R_th_mp = R_c2_mp @ R_th_c2
    p_th_mp = p_c2_mp + np.einsum('nij,nj->ni', R_c2_mp, p_th_c2)

    Rx2 = np.empty((N, 3, 3))
    Rx2[:, 0, 0] = 1;  Rx2[:, 0, 1] = 0;   Rx2[:, 0, 2] = 0
    Rx2[:, 1, 0] = 0;  Rx2[:, 1, 1] = c2;   Rx2[:, 1, 2] = -s2
    Rx2[:, 2, 0] = 0;  Rx2[:, 2, 1] = s2;   Rx2[:, 2, 2] = c2
    R_tb_th = _R_tb @ Rx2

    p_tb_th = np.tile(_p_tb, (N, 1))

    R_tb_mp = R_th_mp @ R_tb_th
    p_tb_mp = p_th_mp + np.einsum('nij,nj->ni', R_th_mp, p_tb_th)

    p_foot_mp = p_tb_mp + np.einsum('nij,j->ni', R_tb_mp, _FOOT_TIP)

    return (p_foot_mp - np.tile(_p_c1, (N, 1))) @ _R_c1
