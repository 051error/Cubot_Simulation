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
    def __init__(self):
        super().__init__("mujoco_simulator")

        mj_share = get_package_share_directory("mj_sim")
        scene_path = os.path.join(mj_share, "models", "scene.xml")
        self.get_logger().info(f"Loading: {scene_path}")

        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

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

        # Leg joints: 6 legs x 3 DOF = 18
        n_leg = min(18, m.nq)
        msg.leg_pos[:n_leg]    = d.qpos[:n_leg]
        msg.leg_vel[:n_leg]    = d.qvel[:n_leg]
        msg.leg_torque[:n_leg] = d.qfrc_actuator[:n_leg]

        # Wheel joints: 4 wheels (after leg DOFs)
        ws = 18
        n_wheel = min(4, m.nq - ws)
        msg.wheel_pos[:n_wheel] = d.qpos[ws:ws + n_wheel]
        msg.wheel_vel[:n_wheel] = d.qvel[ws:ws + n_wheel]

        # IMU from MP_BODY frame
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "MP_BODY")
        if body_id >= 0:
            msg.imu_quat[:] = d.xquat[body_id]
            msg.imu_gyro[:] = d.qvel[body_id * 6 + 3 : body_id * 6 + 6]

        self.low_state_pub.publish(msg)

    def simulation_loop(self):
        with mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=True, show_right_ui=True
        ) as viewer:
            self.get_logger().info("Viewer launched!")
            viewer.cam.lookat[:] = [0, 0, 0.15]
            viewer.cam.distance = 1.2
            viewer.cam.elevation = -25
            viewer.cam.azimuth = 135

            while self.viewer_running and rclpy.ok():
                step_start = time.time()
                if self.receive_data:
                    self.data.ctrl[:] = self.cmd_buffer
                last_ctrl = Float64MultiArray()
                last_ctrl.data = self.data.ctrl
                self.last_ctrl_pub.publish(last_ctrl)
                mujoco.mj_step(self.model, self.data)
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
