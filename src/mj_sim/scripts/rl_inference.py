#!/usr/bin/env python3
"""CPG+RL inference node — aligned with arXiv:2310.07744.

Architecture (must match train_rl.py exactly):
  RL policy → CPG foot-trajectory params (8D) → Hopf oscillators
  → foot positions in coxa frame (6×3D) → IK solver → joint angles (18D)
  → publish /rl_action

Observation layout:
  projected_gravity(3) + body_vel(3) + feet_pos(18) + feet_vel(18)
  + foot_contact(6) + cmd(3) + prev_action(8) + osc_state(12) = 71
"""

import os, sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Float64MultiArray
from mj_sim.msg import LowState, UpCmd
from stable_baselines3 import PPO

CTRL_DT = 0.02          # 50Hz, must match train_rl.py
CTRL_EVERY_N = 4        # low_state arrives at 200Hz; run policy every 4 msgs = 50Hz
N_CPG_PARAMS = 8        # must match train_rl.py
OBS_DIM = 3 + 3 + 18 + 18 + 6 + 3 + N_CPG_PARAMS + 12  # 71

from leg_ik import LegIK
from hexapod_cpg import HexapodCPG


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def quat_to_projected_gravity(qw, qx, qy, qz):
    return np.array([
        -2.0 * (qx * qz - qw * qy),
        -2.0 * (qy * qz + qw * qx),
        -(1.0 - 2.0 * (qx * qx + qy * qy)),
    ], dtype=np.float32)


def find_model():
    candidates = []
    try:
        share = get_package_share_directory("mj_sim")
        candidates.append(os.path.join(share, "rl_checkpoints", "best_model.zip"))
    except Exception:
        pass

    script_dir = os.path.dirname(os.path.realpath(__file__))
    p = os.path.normpath(script_dir)
    for _ in range(8):
        candidate = os.path.join(p, "src", "mj_sim", "rl_checkpoints", "best_model.zip")
        if os.path.exists(candidate):
            candidates.append(candidate)
            break
        parent = os.path.normpath(os.path.join(p, ".."))
        if parent == p:
            break
        p = parent

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "best_model.zip not found. Searched:\n  " + "\n  ".join(candidates) +
        "\nTrain the model first:  python3 src/mj_sim/scripts/train_rl.py")


# ══════════════════════════════════════════════════════════════════════════
# RL Policy Node
# ══════════════════════════════════════════════════════════════════════════

class RLPolicyNode(Node):
    def __init__(self):
        super().__init__("rl_policy")

        model_path = find_model()
        self.model = PPO.load(model_path)
        self.get_logger().info(f"Loaded CPG+RL model: {model_path}")

        self.cpg = HexapodCPG(n_legs=6, dt=CTRL_DT)
        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._prev_action = np.zeros(N_CPG_PARAMS, dtype=np.float32)
        self._msg_count = 0   # decimate 200Hz low_state down to 50Hz

        self.state_sub = self.create_subscription(
            LowState, "/mujoco/low_state", self.state_callback, 10)
        self.cmd_sub = self.create_subscription(
            UpCmd, "/upper_ctrl", self.cmd_callback, 10)
        self.action_pub = self.create_publisher(
            Float64MultiArray, "/rl_action", 10)

        self.get_logger().info("CPG+RL policy ready. Sub: /mujoco/low_state + /upper_ctrl")

    def cmd_callback(self, msg: UpCmd):
        self.cmd = np.array([msg.linear_x, msg.linear_y, msg.angular_z], dtype=np.float32)

    def state_callback(self, msg: LowState):
        # low_state is published at 200Hz but the CPG/policy run at 50Hz.
        # Only process every CTRL_EVERY_N-th message so CPG dt=0.02 stays
        # wall-clock aligned (matches train_rl.py env cadence).
        self._msg_count += 1
        if self._msg_count % CTRL_EVERY_N != 0:
            return

        imu = msg.imu_quat
        projected_gravity = quat_to_projected_gravity(imu[0], imu[1], imu[2], imu[3])
        osc_state = self.cpg.get_osc_state()

        obs = np.concatenate([
            projected_gravity,
            np.array(msg.body_vel, dtype=np.float32) / [0.3, 0.3, 2.0],
            np.array(msg.feet_pos, dtype=np.float32) / 0.2,
            np.array(msg.feet_vel, dtype=np.float32) / 0.5,
            np.array(msg.foot_contact, dtype=np.float32),
            self.cmd.copy() / np.array([0.05, 0.05, 1.0], dtype=np.float32),
            self._prev_action.copy(),
            osc_state,
        ])
        obs = np.clip(obs, -10.0, 10.0)

        # RL → CPG foot params → foot positions → IK → joint angles
        action, _ = self.model.predict(obs, deterministic=True)
        self._prev_action = action.copy()

        # No forward bias — RL directly controls all 8 CPG params.
        # action[0] ∈ [-1,1] → coxa_amp ∈ [0.005, 0.025]m.
        foot_targets = self.cpg.step(action)
        joint_targets = np.empty(18, dtype=np.float32)
        for i in range(6):
            j = i * 3
            angles = LegIK.solve(
                foot_targets[j + 2],  # c1_rest X = vertical (CPG z)
                foot_targets[j + 1],  # c1_rest Y = forward  (CPG y)
                foot_targets[j + 0],  # c1_rest Z = lateral  (CPG x)
                leg_idx=i,
            )
            joint_targets[j:j+3] = angles

        scaled = np.clip(joint_targets, -2.5, 2.5)
        cmd = Float64MultiArray()
        cmd.data = scaled.tolist()
        self.action_pub.publish(cmd)


def main():
    rclpy.init()
    node = RLPolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
