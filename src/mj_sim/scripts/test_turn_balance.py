#!/usr/bin/env python3
"""Automated turn-mode balance test for the cubot hexapod.

Drives the robot in TURN mode (RB held + right stick X) at a FAST and a SLOW
speed, samples its attitude from /mujoco/low_state, and reports balance metrics
to the ROS log and a text file.

Control mapping (must match xbox_controller.cpp / robot_ctrl.cpp):
  Joy.axes[3]    = right stick X  -> turn direction & speed, in [-1, 1]
  Joy.buttons[5] = RB (1 = held)  -> TURN mode
  Joy.buttons[4] = LB             -> must stay 0 (LB+RB would enter RL mode)
  sign of axes[3]: >0 -> turn right, <0 -> turn left.

Balance metric (same convention as the C++ HUD and the Python overlay):
  roll  = atan2(gy, -gz),  pitch = atan2(-gx, -gz)   from imu_quat (w,x,y,z)
  "BALANCED" when |roll| < 2 deg AND |pitch| < 2 deg, else "TILTED".

NOTE: this node publishes /joy continuously. Run it with the physical joystick
      untouched (or joy_node silent) so the two /joy publishers don't fight.
"""

import argparse
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from mj_sim.msg import LowState

BALANCE_DEG = 2.0  # "balanced" attitude threshold


class TurnBalanceTest(Node):
    def __init__(self, scenarios, duration, log_path):
        super().__init__("turn_balance_test")
        self._joy_pub = self.create_publisher(Joy, "/joy", 10)
        self.create_subscription(LowState, "/mujoco/low_state",
                                 self._on_low_state, 10)
        self.create_timer(0.02, self._publish_joy)  # 50 Hz command rate

        self._scenarios = scenarios  # list of (label, rx)
        self._duration = duration    # seconds per scenario
        self._log_path = log_path

        # Current commanded state (published every timer tick).
        self._rx = 0.0
        self._rb = False
        # Sample buffers while a scenario is active.
        self._recording = False
        self._roll = []
        self._pitch = []

    # ── callbacks ─────────────────────────────────────────────────────────

    def _on_low_state(self, msg):
        if len(msg.imu_quat) < 4:
            return
        w, x, y, z = msg.imu_quat[:4]
        gx = -2.0 * (x * z - w * y)
        gy = -2.0 * (y * z + w * x)
        gz = -(1.0 - 2.0 * (x * x + y * y))
        if self._recording:
            self._roll.append(math.degrees(math.atan2(gy, -gz)))
            self._pitch.append(math.degrees(math.atan2(-gx, -gz)))

    def _publish_joy(self):
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.axes = [0.0] * 6
        msg.axes[3] = self._rx
        msg.buttons = [0] * 8
        msg.buttons[5] = 1 if self._rb else 0
        self._joy_pub.publish(msg)

    # ── helpers ───────────────────────────────────────────────────────────

    def _spin_for(self, seconds):
        """Pump ROS callbacks for `seconds` while /joy keeps publishing."""
        deadline = time.time() + seconds
        while time.time() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

    def _run_scenario(self, label, rx):
        self.get_logger().info(
            f"=== {label}: rx={rx:+.2f} for {self._duration:.1f}s ==="
        )
        # Settle: RB held, stick centred -> turn mode holds the crouch.
        self._rb = True
        self._rx = 0.0
        self._spin_for(1.0)

        # Turn and record attitude.
        self._roll = []
        self._pitch = []
        self._recording = True
        self._rx = rx
        self._spin_for(self._duration)
        self._recording = False

        # Centre the stick, then release RB.
        self._rx = 0.0
        self._spin_for(0.5)
        self._rb = False
        self._spin_for(0.2)

        roll = np.asarray(self._roll)
        pitch = np.asarray(self._pitch)
        n = roll.size
        if n == 0:
            self.get_logger().warn(f"{label}: no low_state samples received")
            return None

        abs_roll = np.abs(roll)
        abs_pitch = np.abs(pitch)
        tilt = np.hypot(roll, pitch)  # combined roll+pitch magnitude
        balanced = (abs_roll < BALANCE_DEG) & (abs_pitch < BALANCE_DEG)
        return {
            "label": label,
            "rx": rx,
            "n": int(n),
            "roll_abs_mean": float(abs_roll.mean()),
            "roll_abs_max": float(abs_roll.max()),
            "roll_rms": float(np.sqrt(np.mean(roll ** 2))),
            "pitch_abs_mean": float(abs_pitch.mean()),
            "pitch_abs_max": float(abs_pitch.max()),
            "pitch_rms": float(np.sqrt(np.mean(pitch ** 2))),
            "tilt_rms": float(np.sqrt(np.mean(tilt ** 2))),
            "balanced_ratio": float(balanced.mean()),
        }

    @staticmethod
    def _fmt(m):
        return (
            f"[{m['label']}] rx={m['rx']:+.2f} n={m['n']} "
            f"|roll| mean={m['roll_abs_mean']:.2f} max={m['roll_abs_max']:.2f} "
            f"rms={m['roll_rms']:.2f} deg |pitch| mean={m['pitch_abs_mean']:.2f} "
            f"max={m['pitch_abs_max']:.2f} rms={m['pitch_rms']:.2f} deg "
            f"tilt_rms={m['tilt_rms']:.2f} deg balanced={m['balanced_ratio'] * 100:.1f}%"
        )

    # ── main flow ─────────────────────────────────────────────────────────

    def run(self):
        self.get_logger().info(
            "Starting turn-mode balance test (RB held + right stick X)."
        )
        results = []
        for label, rx in self._scenarios:
            m = self._run_scenario(label, rx)
            if m is not None:
                results.append(m)
                self.get_logger().info("RESULT " + self._fmt(m))

        if len(results) == 2:
            a, b = results
            better = a["label"] if a["tilt_rms"] <= b["tilt_rms"] else b["label"]
            self.get_logger().info(
                f"SUMMARY: {a['label']} tilt_rms={a['tilt_rms']:.3f} deg "
                f"(balanced {a['balanced_ratio'] * 100:.1f}%) vs "
                f"{b['label']} tilt_rms={b['tilt_rms']:.3f} deg "
                f"(balanced {b['balanced_ratio'] * 100:.1f}%) -> "
                f"{better} turn is more balanced"
            )

        self._write_log(results)
        self.get_logger().info(f"Log written to {self._log_path}")

    def _write_log(self, results):
        with open(self._log_path, "w") as f:
            f.write(f"# cubot turn-mode balance test — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# balanced threshold: |roll|,|pitch| < {BALANCE_DEG:.1f} deg\n")
            f.write("label rx n_samples roll_abs_mean roll_abs_max roll_rms "
                    "pitch_abs_mean pitch_abs_max pitch_rms tilt_rms balanced_ratio\n")
            for m in results:
                f.write(
                    f"{m['label']} {m['rx']:+.2f} {m['n']} "
                    f"{m['roll_abs_mean']:.3f} {m['roll_abs_max']:.3f} {m['roll_rms']:.3f} "
                    f"{m['pitch_abs_mean']:.3f} {m['pitch_abs_max']:.3f} {m['pitch_rms']:.3f} "
                    f"{m['tilt_rms']:.3f} {m['balanced_ratio']:.4f}\n"
                )


def main():
    parser = argparse.ArgumentParser(description="Turn-mode balance test")
    parser.add_argument("--fast-rx", type=float, default=1.0,
                        help="right-stick X for the fast turn (default 1.0)")
    parser.add_argument("--slow-rx", type=float, default=0.2,
                        help="right-stick X for the slow turn (default 0.2)")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="seconds per scenario (default 30.0)")
    parser.add_argument("--log", type=str, default="turn_balance_log.txt",
                        help="output log file (default turn_balance_log.txt)")
    args = parser.parse_args()

    rclpy.init()
    node = TurnBalanceTest(
        scenarios=[("FAST", args.fast_rx), ("SLOW", args.slow_rx)],
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
