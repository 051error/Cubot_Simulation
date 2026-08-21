#!/usr/bin/env python3
"""MuJoCo simulator for PhantomX robot — loads MJCF scene."""

import os, sys, time, threading

# Portable mujoco import: try env var first, then common paths
_mj_path = os.environ.get("MUJOCO_PYTHON_PATH", "")
if _mj_path:
    sys.path.insert(0, _mj_path)
elif os.path.isdir(os.path.expanduser("~/myenv/lib/python3.12/site-packages/mujoco")):
    sys.path.insert(0, os.path.expanduser("~/myenv/lib/python3.12/site-packages"))

import mujoco, mujoco.viewer
import warnings
warnings.filterwarnings("ignore", message=".*Wayland.*")
import numpy as np
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Float64MultiArray
from mj_sim.msg import LowState


class MujocoSimulator(Node):
    BALANCE_DEG = 2.0  # attitude threshold for "balanced" vs "tilted"

    # Camera follow modes, cycled with the '1' key.
    CAM_FREE = 0       # manual orbit (drag to move)
    CAM_TRACKING = 1   # follow body position + heading
    CAM_FOLLOW = 2     # fixed overhead, follow position only
    CAM_LABELS = {
        0: "CAM: free (drag to orbit, 1 to switch)",
        1: "CAM: follow (pos + heading, 1 to switch)",
        2: "CAM: overhead (pos only, 1 to switch)",
    }

    def __init__(self):
        super().__init__("mujoco_simulator")

        mj_share = get_package_share_directory("mj_sim")
        scene_path = os.path.join(mj_share, "models", "terrain.xml")
        self.get_logger().info(f"Loading: {scene_path}")

        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self._body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "MP_BODY"
        )

        # Crouched home pose = the control "zero". The 18 leg joints are plain
        # absolute hinge angles (their `ref` stays at the default 0 — a non-zero
        # ref/qpos0 would shift the qpos zero point and break the geometry).
        # coxa=0, thigh=-0.7593, tibia=-0.7108 (×6), lid=0. The root free-joint
        # is lowered to -0.05 so this crouch keeps all 6 foot tips planted.
        crouch_legs = np.array([0.0, -0.7593, -0.7108] * 6)   # 18 leg qpos
        self.crouch_qpos = crouch_legs
        self.home_ctrl = np.concatenate([crouch_legs, [0.0]])  # 18 legs + lid ctrl
        self.data.qpos[7:25] = crouch_legs   # start crouched (qpos0 is straight)
        self.data.ctrl[:] = self.home_ctrl

        self.get_logger().info(
            f"Loaded: bodies={self.model.nbody} nq={self.model.nq} nu={self.model.nu}"
        )

        self.low_state_pub = self.create_publisher(LowState, "/mujoco/low_state", 10)
        self.last_ctrl_pub = self.create_publisher(Float64MultiArray, "/mujoco/last_ctrl", 10)
        self.low_cmd_sub = self.create_subscription(
            Float64MultiArray, "/mujoco/low_cmd", self.low_cmd_callback, 10
        )
        self.receive_data = False
        self.viewer_running = True
        self.cam_mode = self.CAM_TRACKING
        self.cmd_buffer = np.zeros(self.model.nu)
        # Init lid to closed (qpos=0, corresponds to ctrl=1)
        if self.model.nu > 18:
            self.cmd_buffer[18] = 0.0
        self.create_timer(0.005, self.publish_sensor_data)
        self.get_logger().info("Ready.")

    def low_cmd_callback(self, msg):
        if len(msg.data) != self.model.nu: return
        self.cmd_buffer[:] = msg.data
        self.receive_data = True

    def publish_sensor_data(self):
        if self.model is None or self.data is None:
            return

        msg = LowState()
        m = self.model
        d = self.data

        # Leg joints: 6 legs x 3 DOF = 18 (skip root free-joint qpos[7:25])
        msg.leg_pos[:]    = d.qpos[7:25]
        msg.leg_vel[:]    = d.qvel[6:24]
        msg.leg_torque[:] = d.qfrc_actuator[:18]

        # IMU from MP_BODY frame
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "MP_BODY")
        if body_id >= 0:
            quat = d.xquat[body_id].copy()
            msg.imu_quat[:] = quat
            msg.imu_gyro[:] = d.qvel[3:6]  # world-frame body angular velocity

            # ── Task-space fields for RL policy ──────────────────────────
            w, x, y, z = quat
            R = np.array([
                [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
                [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
                [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
            ])

            # Body velocity in body frame: vx, vy, wz
            world_vel = d.qvel[0:6] if m.nv >= 6 else np.zeros(6)
            body_lin = R @ world_vel[0:3]
            body_ang = R @ world_vel[3:6]
            msg.body_vel[0] = body_lin[0]
            msg.body_vel[1] = body_lin[1]
            msg.body_vel[2] = body_ang[2]

            # Foot positions / velocities in body frame, + contacts
            body_pos = d.xpos[body_id].copy()
            foot_names = [
                "tibia_rf", "tibia_rm", "tibia_rr",
                "tibia_lf", "tibia_lm", "tibia_lr",
            ]
            dt = 0.005  # 200 Hz timer period
            prev_feet = getattr(self, '_prev_foot_pos', None)
            cur_feet = []

            for i, name in enumerate(foot_names):
                fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
                if fid >= 0:
                    # Position
                    fp_world = d.xpos[fid].copy()
                    cur_feet.append(fp_world)
                    fp_body = R @ (fp_world - body_pos)
                    j = i * 3
                    msg.feet_pos[j:j+3] = fp_body

                    # Velocity via position delta (robust, no cvel needed)
                    if prev_feet is not None and i < len(prev_feet):
                        fv_world = (fp_world - prev_feet[i]) / dt
                    else:
                        fv_world = np.zeros(3)
                    msg.feet_vel[j:j+3] = R @ fv_world

                    # Contact: compute foot tip (mesh bottom), not body origin
                    tibia_rot = d.xmat[fid].reshape(3, 3)
                    foot_tip_tibia = np.array([0.00162782, 0.16052104, 0.02951023])
                    foot_tip_world = fp_world + tibia_rot @ foot_tip_tibia
                    msg.foot_contact[i] = 1.0 if foot_tip_world[2] < 0.015 else 0.0
                else:
                    cur_feet.append(np.zeros(3))

            self._prev_foot_pos = cur_feet

        # 360° rangefinder distances (24 azimuths x 3 elevations = 72).
        if m.nsensor >= 72:
            msg.rangefinder[:] = d.sensordata[:72]

        self.low_state_pub.publish(msg)

    def _balance_overlay(self):
        """Return (text1, text2) showing body attitude and balance status.

        Roll/pitch are recovered from the body quaternion via projected
        gravity (same convention as the C++ HUD). "BALANCED" when both angles
        are within ±BALANCE_DEG, otherwise "TILTED".
        """
        if self._body_id < 0:
            return "attitude: n/a", ""
        qw, qx, qy, qz = self.data.xquat[self._body_id]
        gx = -2.0 * (qx * qz - qw * qy)
        gy = -2.0 * (qy * qz + qw * qx)
        gz = -(1.0 - 2.0 * (qx * qx + qy * qy))
        roll = np.degrees(np.arctan2(gy, -gz))
        pitch = np.degrees(np.arctan2(-gx, -gz))
        level = abs(roll) < self.BALANCE_DEG and abs(pitch) < self.BALANCE_DEG
        return (
            f"roll {roll:+.1f} deg  pitch {pitch:+.1f} deg",
            "BALANCED" if level else "TILTED",
        )

    def _on_key(self, keycode):
        """Cycle the camera follow mode on the '1' key.

        keycode: GLFW key code of the pressed key (digits map to ASCII).
        """
        if keycode == ord("1"):
            self.cam_mode = (self.cam_mode + 1) % 3

    def _body_pose(self):
        """Return the MP_BODY position and rotation in the world frame.

        Returns (pos, R): pos is the body origin (3-vector), R is the 3x3 xmat.
        """
        pos = self.data.xpos[self._body_id].copy()
        R = self.data.xmat[self._body_id].reshape(3, 3).copy()
        return pos, R

    @staticmethod
    def _spherical_to_world(R, azimuth, elevation):
        """Convert body-frame camera angles to world azimuth/elevation.

        R: body-to-world rotation matrix (3x3).
        azimuth, elevation: camera angles in the body frame (degrees).
        Returns (azimuth_world, elevation_world) in degrees.
        """
        ce = np.cos(np.radians(elevation))
        fwd_body = np.array([
            ce * np.cos(np.radians(azimuth)),
            ce * np.sin(np.radians(azimuth)),
            np.sin(np.radians(elevation)),
        ])
        fwd_world = R @ fwd_body
        az_world = np.degrees(np.arctan2(fwd_world[1], fwd_world[0]))
        el_world = np.degrees(np.arcsin(np.clip(fwd_world[2], -1.0, 1.0)))
        return az_world, el_world

    def _apply_camera(self, viewer):
        """Set the viewer camera for the current follow mode.

        viewer: mujoco.viewer.Handle to configure.
        """
        with viewer.lock():
            cam = viewer.cam
            if self.cam_mode == self.CAM_TRACKING:
                # Full follow: fixed offset behind/above the body, turning with
                # it. MuJoCo's native mjCAMERA_TRACKING only follows the body
                # COM at a fixed world orientation, so the heading is applied by
                # rotating a body-frame view direction into world space.
                pos, R = self._body_pose()
                az, el = self._spherical_to_world(R, 135.0, -25.0)
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.lookat[:] = pos + R @ np.array([0.0, 0.0, 0.15])
                cam.azimuth = az
                cam.elevation = el
                cam.distance = 1.2
            elif self.cam_mode == self.CAM_FOLLOW:
                # Position-only follow: fixed world orientation, look at body.
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.lookat[:] = self.data.xpos[self._body_id] + [0, 0, 0.15]
                cam.azimuth = 135.0
                cam.elevation = -25.0
                cam.distance = 1.2
            else:
                # Free mode: only force the type so the user can drag the camera.
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def simulation_loop(self):
        with mujoco.viewer.launch_passive(
            self.model, self.data,
            key_callback=self._on_key,
            show_left_ui=True, show_right_ui=True,
        ) as viewer:
            self.get_logger().info(
                "Viewer launched! Press '1' to cycle the camera mode."
            )

            # Hide the rangefinder rays (visualization only). This only turns off
            # the yellow ray drawing; the sensors keep outputting distances in
            # d.sensordata regardless of this flag.
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False

            while self.viewer_running and rclpy.ok():
                step_start = time.time()
                if self.receive_data:
                    self.data.ctrl[:] = self.cmd_buffer
                else:
                    # Backspace -> mj_resetData resets qpos to qpos0 (straight
                    # legs) and zeroes ctrl. Detect that exact reset signature and
                    # restore the crouch once. Do NOT re-assert qpos/ctrl every
                    # step, or the right-hand UI panel freezes (manual joint edits
                    # get overwritten each frame).
                    if (np.allclose(self.data.qpos[7:25], 0.0) and
                            np.allclose(self.data.ctrl, 0.0)):
                        self.data.qpos[7:25] = self.crouch_qpos
                        self.data.ctrl[:] = self.home_ctrl
                last_ctrl = Float64MultiArray()
                last_ctrl.data = self.data.ctrl
                self.last_ctrl_pub.publish(last_ctrl)
                mujoco.mj_step(self.model, self.data)

                # Apply the camera follow mode, then draw the overlay.
                self._apply_camera(viewer)
                text1, text2 = self._balance_overlay()
                viewer.set_texts([
                    (mujoco.mjtFontScale.mjFONTSCALE_150,
                     mujoco.mjtGridPos.mjGRID_TOPLEFT, text1, text2),
                    (mujoco.mjtFontScale.mjFONTSCALE_150,
                     mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                     self.CAM_LABELS[self.cam_mode], ""),
                ])
                viewer.sync()
                elapsed = time.time() - step_start
                if elapsed < 0.005:
                    time.sleep(0.005 - elapsed)
        self.get_logger().info("Simulation ended.")

    def stop(self):
        self.viewer_running = False


def main():
    
    rclpy.init()
    node = MujocoSimulator()
    sim_thread = threading.Thread(target=node.simulation_loop, daemon=True)
    sim_thread.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
