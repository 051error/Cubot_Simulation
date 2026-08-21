#!/usr/bin/env python3
"""CPG+RL inference node — residual joint-angle policy.

Pipeline: C++ tripod CPG target (200Hz) + RL residual (50Hz) -> /rl_action.
Observation (159D): gravity(3)+body_vel(3)+joint_pos(18)+joint_vel(18)
+contact(6)+cmd(3)+prev_action(18)+q_target(18)+height_map(72).
"""

import os
import numpy as np
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Float64MultiArray
from mj_sim.msg import LowState, UpCmd
from stable_baselines3 import PPO

CTRL_EVERY_N = 4        # low_state arrives at 200Hz; run policy every 4 msgs = 50Hz
ACTION_SCALE = 0.2      # action in [-1,1] -> joint-angle increment (rad), matches train_rl.py
OBS_DIM = 3 + 3 + 18 + 18 + 6 + 3 + 18 + 18 + 72  # 159

from cpg_gait import TripodGait, FootTrajectory, compute_joint_targets, JOINT_REF
from height_map import rangefinder_to_height_map


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

        self.cpg = TripodGait()
        self.traj = FootTrajectory()
        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._prev_action = np.zeros(18, dtype=np.float32)
        self._current_action = np.zeros(18, dtype=np.float32)
        self._q_target = JOINT_REF.copy()
        self._msg_count = 0   # decimate 200Hz low_state down to 50Hz

        self.state_sub = self.create_subscription(
            LowState, "/mujoco/low_state", self.state_callback, 10)
        self.cmd_sub = self.create_subscription(
            UpCmd, "/upper_ctrl", self.cmd_callback, 10)
        self.action_pub = self.create_publisher(
            Float64MultiArray, "/rl_action", 10)

        self.get_logger().info("CPG+RL policy ready. Sub: /mujoco/low_state + /upper_ctrl")

    def cmd_callback(self, msg: UpCmd):
        # RL handles only the left stick (linear_x/y). Rotation is delegated to
        # the dedicated TURN mode, so the right stick (angular_z) is ignored here.
        self.cmd = np.array([msg.linear_x, msg.linear_y, 0.0], dtype=np.float32)

    def state_callback(self, msg: LowState):
        self._msg_count += 1

        # CPG integrates at 200Hz (every low_state); the RL residual is held at
        # 50Hz. Publish q_target + residual every message so the full-rate CPG
        # motion reaches the robot (matches train_rl.py's inner-loop cadence).
        self._q_target = compute_joint_targets(self.cpg, self.traj,
                                               self.cmd[0], self.cmd[1])

        if self._msg_count % CTRL_EVERY_N == 0:
            obs = self._build_obs(msg)
            action, _ = self.model.predict(obs, deterministic=True)
            self._current_action = np.asarray(action, dtype=np.float32)
            self._prev_action = self._current_action.copy()

        q = self._q_target + self._current_action * ACTION_SCALE
        cmd = Float64MultiArray()
        cmd.data = np.clip(q, -2.5, 2.5).tolist()
        self.action_pub.publish(cmd)

    def _build_obs(self, msg: LowState):
        imu = msg.imu_quat
        projected_gravity = quat_to_projected_gravity(imu[0], imu[1], imu[2], imu[3])

        # 360° rangefinder -> robot-centric local height map (72 ground heights).
        height_map = rangefinder_to_height_map(np.asarray(msg.rangefinder))

        obs = np.concatenate([
            projected_gravity,                               # 3
            np.array(msg.body_vel, dtype=np.float32) / [0.3, 0.3, 2.0],   # 3
            np.array(msg.leg_pos, dtype=np.float32) / 1.0,   # 18
            np.array(msg.leg_vel, dtype=np.float32) / 5.0,   # 18
            np.array(msg.foot_contact, dtype=np.float32),    # 6
            self.cmd.copy() / np.array([0.05, 0.05, 1.0], dtype=np.float32),  # 3
            self._prev_action.copy(),                        # 18
            self._q_target / 1.0,                            # 18
            height_map / 0.5,                                # 72
        ])
        return np.clip(obs, -10.0, 10.0).astype(np.float32)


def main():
    rclpy.init()
    node = RLPolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
