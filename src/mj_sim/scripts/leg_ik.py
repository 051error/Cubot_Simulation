"""KD-tree Inverse Kinematics for PhantomX 3-DOF hexapod leg.

c1_rest frame: X = vertical (world Z, up+), Y = fwd from hip, Z = lateral.
Precomputed FK grid → cKDTree nearest-neighbor lookup → ~330 µs per solve.

Joint rotation conventions (verified against MuJoCo):
  j_c1_rf   (θ0): POST-multiply — body rotates in its own local frame
  j_thigh_rf (θ1): PRE-multiply  — body rotates in parent frame
  j_tibia_rf (θ2): POST-multiply — body rotates in its own local frame
  Post-multiply: position offset does NOT rotate with the joint
  Pre-multiply:  position offset DOES rotate with the joint

Usage:
    from leg_ik import LegIK
    angles = LegIK.solve(x_m, y_m, z_m, leg_idx=0)
"""

import numpy as np
from scipy.spatial import cKDTree as _cKDTree


# ══════════════════════════════════════════════════════════════════════════
# PhantomX leg parameters (extracted from cubot.xml MJCF model)
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

# Foot tip in tibia local frame (lowest mesh vertex, computed from geom_pos + R_geom @ mesh_lowest)
_FOOT_TIP = np.array([0.00162782, 0.16052104, 0.02951023])


# ══════════════════════════════════════════════════════════════════════════
# Forward Kinematics (scalar, for verification)
# ══════════════════════════════════════════════════════════════════════════

def _leg_fk(theta0, theta1, theta2):
    """FK: (θ0,θ1,θ2) → foot position in c1_rest frame.

    Body hierarchy (parent→child):
      MP_BODY → c1_rf (j_c1_rf, POST) → c2_rf (fixed)
              → thigh_rf (j_thigh_rf, PRE) → tibia_rf (j_tibia_rf, POST)
    """
    # c1 in MP_BODY: POST-multiply
    R_c1_mp = _R_c1 @ _RotX(theta0)
    p_c1_mp = _p_c1  # post: position does NOT rotate

    # c2 in c1: FIXED (no joint between c1 and c2)
    R_c2_c1 = _R_c2
    p_c2_c1 = _p_c2
    R_c2_mp = R_c1_mp @ R_c2_c1
    p_c2_mp = p_c1_mp + R_c1_mp @ p_c2_c1

    # thigh in c2: PRE-multiply
    R_th_c2 = _RotX(theta1) @ _R_th
    p_th_c2 = _RotX(theta1) @ _p_th  # pre: position rotates
    R_th_mp = R_c2_mp @ R_th_c2
    p_th_mp = p_c2_mp + R_c2_mp @ p_th_c2

    # tibia in thigh: POST-multiply
    R_tb_th = _R_tb @ _RotX(theta2)
    p_tb_th = _p_tb  # post: position does NOT rotate
    R_tb_mp = R_th_mp @ R_tb_th
    p_tb_mp = p_th_mp + R_th_mp @ p_tb_th

    # Foot tip in MP_BODY, then transform to c1_rest
    p_foot_mp = p_tb_mp + R_tb_mp @ _FOOT_TIP
    return _R_c1.T @ (p_foot_mp - _p_c1)


# ══════════════════════════════════════════════════════════════════════════
# Vectorized FK for fast grid precomputation
# ══════════════════════════════════════════════════════════════════════════

def _leg_fk_batch(t0, t1, t2):
    """Vectorized FK: (N,) joint arrays → (N,3) foot positions in c1_rest."""
    N = len(t0)
    c0, s0 = np.cos(t0), np.sin(t0)
    c1, s1 = np.cos(t1), np.sin(t1)
    c2, s2 = np.cos(t2), np.sin(t2)

    # ── c1 in MP: POST-multiply ──
    # R_c1_mp = _R_c1 @ RotX(θ0)
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

    # p_c1_mp = _p_c1 (post: no rotation of position)
    p_c1_mp = np.tile(_p_c1, (N, 1))

    # ── c2 in c1: FIXED ──
    # R_c2_mp = R_c1_mp @ _R_c2
    R_c2_mp = R_c1_mp @ _R_c2
    # p_c2_mp = p_c1_mp + R_c1_mp @ _p_c2
    p_c2_mp = p_c1_mp + np.einsum('nij,j->ni', R_c1_mp, _p_c2)

    # ── thigh in c2: PRE-multiply ──
    # R_th_c2 = RotX(θ1) @ _R_th
    Rx1 = np.empty((N, 3, 3))
    Rx1[:, 0, 0] = 1;  Rx1[:, 0, 1] = 0;   Rx1[:, 0, 2] = 0
    Rx1[:, 1, 0] = 0;  Rx1[:, 1, 1] = c1;   Rx1[:, 1, 2] = -s1
    Rx1[:, 2, 0] = 0;  Rx1[:, 2, 1] = s1;   Rx1[:, 2, 2] = c1
    R_th_c2 = Rx1 @ _R_th

    # p_th_c2 = RotX(θ1) @ _p_th (= 0 since _p_th=[0,0,0])
    p_th_c2 = np.einsum('nij,j->ni', Rx1, _p_th)

    # R_th_mp = R_c2_mp @ R_th_c2
    R_th_mp = R_c2_mp @ R_th_c2
    # p_th_mp = p_c2_mp + R_c2_mp @ p_th_c2
    p_th_mp = p_c2_mp + np.einsum('nij,nj->ni', R_c2_mp, p_th_c2)

    # ── tibia in thigh: POST-multiply ──
    # R_tb_th = _R_tb @ RotX(θ2)
    Rx2 = np.empty((N, 3, 3))
    Rx2[:, 0, 0] = 1;  Rx2[:, 0, 1] = 0;   Rx2[:, 0, 2] = 0
    Rx2[:, 1, 0] = 0;  Rx2[:, 1, 1] = c2;   Rx2[:, 1, 2] = -s2
    Rx2[:, 2, 0] = 0;  Rx2[:, 2, 1] = s2;   Rx2[:, 2, 2] = c2
    R_tb_th = _R_tb @ Rx2  # POST: _R_tb on left

    # p_tb_th = _p_tb (post: no rotation of position)
    p_tb_th = np.tile(_p_tb, (N, 1))

    # R_tb_mp = R_th_mp @ R_tb_th
    R_tb_mp = R_th_mp @ R_tb_th
    # p_tb_mp = p_th_mp + R_th_mp @ p_tb_th
    p_tb_mp = p_th_mp + np.einsum('nij,nj->ni', R_th_mp, p_tb_th)

    # ── Foot tip in MP, then to c1_rest ──
    p_foot_mp = p_tb_mp + np.einsum('nij,j->ni', R_tb_mp, _FOOT_TIP)

    # p_foot_c1_rest = _R_c1.T @ (p_foot_mp - _p_c1)
    return (p_foot_mp - np.tile(_p_c1, (N, 1))) @ _R_c1  # = _R_c1.T @ (p - _p_c1)


# ══════════════════════════════════════════════════════════════════════════
# Precompute FK lookup table at module import
# ══════════════════════════════════════════════════════════════════════════

_FK_RES = 0.04  # rad — ~2.4mm avg accuracy, 439K points, ~0.33ms/query
_tv = np.arange(-1.5, 1.5 + _FK_RES / 2, _FK_RES)
_T0, _T1, _T2 = np.meshgrid(_tv, _tv, _tv, indexing='ij')
_t0f, _t1f, _t2f = _T0.ravel(), _T1.ravel(), _T2.ravel()
_fk_grid_p = _leg_fk_batch(_t0f, _t1f, _t2f).astype(np.float32)
_fk_grid_q = np.column_stack([_t0f, _t1f, _t2f]).astype(np.float32)
_kd_tree = _cKDTree(_fk_grid_p)


# ══════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════

class LegIK:
    """IK via KD-tree lookup on precomputed FK grid.

    FK grid: 76³ ≈ 439K points at 0.04 rad resolution.
    Query time: ~330 µs per solve, ~2.4mm avg accuracy.
    RL policy compensates for residual IK error via feedback.

    Args:
        x, y, z: foot target position in c1_rest frame (meters).
                 X = vertical (positive up), Y = forward, Z = lateral.
        leg_idx: ignored (stateless lookup). Kept for API compatibility.

    Returns:
        np.ndarray[float32, shape=(3,)]: joint angles [θ₀, θ₁, θ₂] in radians.
    """

    # Leg indexing matches XML body order:
    #   Right side: rf=0, rm=1, rr=2
    #   Left side:  lf=3, lm=4, lr=5  (mirrored kinematics)
    _LEFT_LEGS = {3, 4, 5}

    @staticmethod
    def solve(x, y, z, leg_idx=0):
        """Solve IK for given foot target in c1_rest frame.

        For left-side legs, the c1_rest frame is mirrored left-right compared
        to c1_rf (the reference leg whose FK grid we precomputed). We must
        negate both the target Z (lateral) and the resulting θ₀ (coxa angle).
        """
        if leg_idx in LegIK._LEFT_LEGS:
            z = -z  # mirror lateral coordinate for left legs
            angles = _fk_grid_q[_kd_tree.query([x, y, z], k=1)[1]].astype(np.float32)
            angles[0] = -angles[0]  # negate c1 angle for mirrored leg
            return angles
        else:
            _dist, _kidx = _kd_tree.query([x, y, z], k=1)
            return _fk_grid_q[_kidx].astype(np.float32)

    @staticmethod
    def clear_cache():
        pass  # stateless KD-tree lookup

    @staticmethod
    def is_reachable(x, y, z):
        _dist, _kidx = _kd_tree.query([x, y, z], k=1)
        p = _leg_fk(float(_fk_grid_q[_kidx, 0]),
                     float(_fk_grid_q[_kidx, 1]),
                     float(_fk_grid_q[_kidx, 2]))
        return np.linalg.norm(p - [x, y, z]) < 0.01
