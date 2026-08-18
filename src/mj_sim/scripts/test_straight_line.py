#!/usr/bin/env python3
"""Automated straight-line + anti-slip test for NORMAL mode.

Drives forward/backward via left stick Y (axes[1], <0=forward) and reports slip,
lateral drift and heading drift from /mujoco/low_state. Publishes /joy.
"""

import argparse
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from mj_sim.msg import LowState

MAX_LINEAR_SPEED = 0.05  # m/s at full stick, must match kMaxLinearSpeed


class StraightLineTest(Node):
    def __init__(self, scenarios, duration, log_path):
        super().__init__("straight_line_test")
        self._joy_pub = self.create_publisher(Joy, "/joy", 10)
        self.create_subscription(LowState, "/mujoco/low_state",
                                 self._on_low_state, 10)
        self.create_timer(0.02, self._publish_joy)  # 50 Hz command rate

        self._scenarios = scenarios  # list of (label, ly)  ly = left-stick Y
        self._duration = duration    # seconds per scenario
        self._log_path = log_path

        self._ly = 0.0               # current left-stick Y command
        self._recording = False

        # Sample buffers while a scenario is active.
        self._t = []
        self._vx = []      # body-frame forward velocity
        self._vy = []      # body-frame lateral velocity
        self._wz = []      # world-frame yaw rate
        self._foot_slip = []  # mean stance-foot horizontal slip speed

    # ── callbacks ─────────────────────────────────────────────────────────

    def _on_low_state(self, msg):
        if not self._recording:
            return
        if len(msg.body_vel) < 3 or len(msg.imu_gyro) < 3:
            return
        self._t.append(self.get_clock().now().nanoseconds / 1e9)
        self._vx.append(msg.body_vel[0])
        self._vy.append(msg.body_vel[1])
        self._wz.append(msg.imu_gyro[2])

        # Mean horizontal slip speed across the stance feet this tick. A planted
        # foot should be stationary relative to the ground, so |feet_vel_xy| is
        # the slip speed (0 while gripping, >0 while sliding).
        slip_sum = 0.0
        n_stance = 0
        if len(msg.foot_contact) >= 6 and len(msg.feet_vel) >= 18:
            for i in range(6):
                if msg.foot_contact[i] > 0.5:
                    fx = msg.feet_vel[3 * i]
                    fy = msg.feet_vel[3 * i + 1]
                    slip_sum += math.hypot(fx, fy)
                    n_stance += 1
        self._foot_slip.append(slip_sum / n_stance if n_stance else 0.0)

    def _publish_joy(self):
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.axes = [0.0] * 6
        msg.axes[1] = self._ly
        msg.buttons = [0] * 8
        self._joy_pub.publish(msg)

    # ── helpers ───────────────────────────────────────────────────────────

    def _spin_for(self, seconds):
        deadline = time.time() + seconds
        while time.time() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

    def _run_scenario(self, label, ly):
        vx_cmd = -ly * MAX_LINEAR_SPEED  # body-frame commanded forward speed
        self.get_logger().info(
            f"=== {label}: ly={ly:+.2f} (vx_cmd={vx_cmd:+.3f} m/s) "
            f"for {self._duration:.1f}s ==="
        )

        # Settle: stick centred, robot stands in crouch.
        self._ly = 0.0
        self._spin_for(1.0)

        # Drive and record.
        self._t = []
        self._vx = []
        self._vy = []
        self._wz = []
        self._foot_slip = []
        self._recording = True
        self._ly = ly
        self._spin_for(self._duration)
        self._recording = False

        # Centre the stick to stop.
        self._ly = 0.0
        self._spin_for(0.5)

        n = len(self._vx)
        if n == 0:
            self.get_logger().warn(f"{label}: no low_state samples received")
            return None

        t = np.asarray(self._t)
        vx = np.asarray(self._vx)
        vy = np.asarray(self._vy)
        wz = np.asarray(self._wz)
        slip = np.asarray(self._foot_slip)

        # Slip ratio: how much of the commanded speed the body actually achieved.
        # Signed mean keeps forward/backward sign; ratio uses magnitudes.
        vx_mean = float(vx.mean())
        slip_ratio = 1.0 - abs(vx_mean) / (abs(vx_cmd) + 1e-9)
        slip_ratio = float(np.clip(slip_ratio, 0.0, 1.0))

        # Straight-line metrics: integrate lateral velocity and yaw rate.
        lateral_drift = float(np.trapz(vy, t))          # metres (approx world)
        heading_drift = float(np.degrees(np.trapz(wz, t)))  # degrees

        return {
            "label": label,
            "ly": ly,
            "vx_cmd": vx_cmd,
            "n": int(n),
            "vx_mean": vx_mean,
            "vx_rms": float(np.sqrt(np.mean(vx ** 2))),
            "vy_mean": float(vy.mean()),
            "vy_rms": float(np.sqrt(np.mean(vy ** 2))),
            "wz_mean": float(np.degrees(wz.mean())),
            "wz_rms": float(np.degrees(np.sqrt(np.mean(wz ** 2)))),
            "slip_ratio": slip_ratio,
            "foot_slip_mean": float(slip.mean()),
            "foot_slip_max": float(slip.max()),
            "lateral_drift": lateral_drift,
            "heading_drift": heading_drift,
        }

    @staticmethod
    def _fmt(m):
        return (
            f"[{m['label']}] vx_cmd={m['vx_cmd']:+.3f} m/s vx_mean={m['vx_mean']:+.3f} "
            f"slip={m['slip_ratio'] * 100:.1f}% foot_slip mean={m['foot_slip_mean']:.3f} "
            f"max={m['foot_slip_max']:.3f} m/s | vy_mean={m['vy_mean']:+.3f} "
            f"rms={m['vy_rms']:.3f} m/s | yaw mean={m['wz_mean']:+.2f} "
            f"rms={m['wz_rms']:.2f} deg/s | lateral_drift={m['lateral_drift']:+.3f} m "
            f"heading_drift={m['heading_drift']:+.2f} deg"
        )

    # ── main flow ─────────────────────────────────────────────────────────

    def run(self):
        self.get_logger().info(
            "Starting straight-line + anti-slip test (NORMAL mode, left stick Y)."
        )
        results = []
        for label, ly in self._scenarios:
            m = self._run_scenario(label, ly)
            if m is not None:
                results.append(m)
                self.get_logger().info("RESULT " + self._fmt(m))

        self._write_log(results)
        self.get_logger().info(f"Log written to {self._log_path}")

    def _write_log(self, results):
        with open(self._log_path, "w") as f:
            f.write(f"# cubot straight-line + anti-slip test — "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# max linear speed = {MAX_LINEAR_SPEED:.3f} m/s; "
                    f"slip = 1 - |vx_mean|/|vx_cmd|\n")
            f.write("label ly vx_cmd vx_mean vx_rms vy_mean vy_rms wz_mean wz_rms "
                    "slip_ratio foot_slip_mean foot_slip_max lateral_drift heading_drift\n")
            for m in results:
                f.write(
                    f"{m['label']} {m['ly']:+.2f} {m['vx_cmd']:+.4f} {m['vx_mean']:+.4f} "
                    f"{m['vx_rms']:.4f} {m['vy_mean']:+.4f} {m['vy_rms']:.4f} "
                    f"{m['wz_mean']:+.4f} {m['wz_rms']:.4f} {m['slip_ratio']:.4f} "
                    f"{m['foot_slip_mean']:.4f} {m['foot_slip_max']:.4f} "
                    f"{m['lateral_drift']:+.4f} {m['heading_drift']:+.4f}\n"
                )


def main():
    parser = argparse.ArgumentParser(description="Straight-line + anti-slip test")
    parser.add_argument("--forward-speed", type=float, default=1.0,
                        help="left-stick Y magnitude for forward (default 1.0)")
    parser.add_argument("--backward-speed", type=float, default=1.0,
                        help="left-stick Y magnitude for backward (default 1.0)")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="seconds per scenario (default 15.0)")
    parser.add_argument("--log", type=str, default="straight_line_log.txt",
                        help="output log file (default straight_line_log.txt)")
    args = parser.parse_args()

    rclpy.init()
    node = StraightLineTest(
        scenarios=[("FORWARD", -args.forward_speed),
                   ("BACKWARD", args.backward_speed)],
        duration=args.duration,
        log_path=args.log,
    )
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
