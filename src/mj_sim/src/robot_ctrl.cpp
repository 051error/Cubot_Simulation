#include "mj_sim/robot_ctrl.hpp"
#include "ament_index_cpp/get_package_prefix.hpp"
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>

// ─── TripodGait ─────────────────────────────────────────────────────────

TripodGait::TripodGait()
{
  // Groups: A={0,2,4}, B={1,3,5} — start π apart
  for (int i = 0; i < N_LEGS; ++i) {
    bool in_a = (i == 0 || i == 2 || i == 4);
    x[i] = in_a ? 1.0 : -1.0;
    y[i] = 0.0;
  }
}

void TripodGait::step(double omega, double rot_bias)
{
  if (omega < 0.01) omega = 0.01;

  // Same-group attract (+K), opposite repel (-K)
  for (int i = 0; i < N_LEGS; ++i) {
    bool in_a_i = (i == 0 || i == 2 || i == 4);

    double coupling = 0.0;
    for (int j = 0; j < N_LEGS; ++j) {
      if (i == j) continue;
      bool in_a_j = (j == 0 || j == 2 || j == 4);
      double w = (in_a_i == in_a_j) ? +K_CPG : -K_CPG;
      coupling += w * (x[j] - x[i]);
    }
    coupling /= (N_LEGS - 1);

    bool is_left = (i >= 3);
    double omega_i = omega + (is_left ? +rot_bias : -rot_bias);

    double r2 = x[i] * x[i] + y[i] * y[i];
    double dx = ALPHA * (MU - r2) * x[i] - omega_i * y[i] + coupling;
    double dy = ALPHA * (MU - r2) * y[i] + omega_i * x[i];

    x[i] += dx * DT;
    y[i] += dy * DT;
  }
}

// ─── FootTrajectory ─────────────────────────────────────────────────────

void FootTrajectory::compute(int leg, double phase_i,
                              double vx, double vy, double rot,
                              double& fx, double& fy, double& fz)
{
  double y_sign = (leg >= 3) ? 1.0 : -1.0;
  double stride = STRIDE * std::min(std::abs(vx) / 0.3 + 0.2, 1.0);

  fx = 0.0;
  fy = y_sign * Y_OFFSET;
  fz = Z_OFFSET;

  if (phase_i > 0.0) {
    // stance — push backward
    double s = (1.0 - phase_i) / 2.0;
    fx += stride * (1.0 - 2.0 * s);
  } else {
    // swing — lift + move forward
    double s = (1.0 + phase_i) / 2.0;
    fx -= stride * (1.0 - 2.0 * s);
    fz += LIFT * std::sin(M_PI * s);
  }

  fx += vy * 0.2;        // lateral
  fx += rot * 0.1;        // rotation
}

// ─── RobotController ────────────────────────────────────────────────────

// ─── Normalized joint-control table ─────────────────────────────────────
// The crouched pose is the control "zero": absolute joint angle (rad) is
//   q = JOINT_REF[j] + u[j] * CTRL_SCALE,   u[j] ∈ [-1, 1]
// so u=0 → crouch, u=±1 → ±0.6 rad around it. JOINT_REF holds the crouched
// qpos (coxa=0, thigh=-0.7593, tibia=-0.7108); the joints' `ref` stays 0.
static const double JOINT_REF[18] = {
    0.0, -0.7593, -0.7108,  // rf: coxa, thigh, tibia
    0.0, -0.7593, -0.7108,  // rm
    0.0, -0.7593, -0.7108,  // rr
    0.0, -0.7593, -0.7108,  // lf
    0.0, -0.7593, -0.7108,  // lm
    0.0, -0.7593, -0.7108,  // lr
};
static constexpr double CTRL_SCALE = 0.6;  // ±0.6 rad (±34°) around the crouch

RobotController::RobotController() : Node("robot_controller"),
    gait_(std::make_unique<TripodGait>())
{
  low_cmd_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
      "/mujoco/low_cmd", 10);

  rl_action_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/rl_action", 10,
      std::bind(&RobotController::rl_action_callback, this, std::placeholders::_1));

  timer_ = this->create_wall_timer(
      std::chrono::milliseconds(5),
      std::bind(&RobotController::timer_callback, this));

  RCLCPP_INFO(this->get_logger(), "RobotController ready. RL sub: /rl_action");
}

void RobotController::rl_action_callback(
    const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  if (msg->data.size() >= 18) {
    std::copy_n(msg->data.begin(), 18, rl_action_.begin());
    rl_active_ = true;
  }
}

void RobotController::set_gait(std::unique_ptr<GaitGenerator> new_gait)
{
  gait_ = std::move(new_gait);
  RCLCPP_INFO(this->get_logger(), "Gait switched to: %s", gait_->name());
}

void RobotController::timer_callback()
{
  double vx = 0, vy = 0, wz = 0;
  if (xbox) { vx = xbox->get_linear_x(); vy = xbox->get_linear_y(); wz = xbox->get_angular_z(); }

  auto cmd = std_msgs::msg::Float64MultiArray();
  cmd.data.resize(19, 0.0);

  double speed = std::sqrt(vx * vx + vy * vy);
  bool idle = (speed < 0.01 && std::abs(wz) < 0.01);

  // Policy mode: LB+RB held = RL (shoulder buttons). Read via XboxController
  // so the mode is driven by the same source that publishes /upper_ctrl.
  bool rl_mode = (xbox && xbox->get_policy_mode() == 1);

  // Log RL-mode transitions (enable/disable)
  if (rl_mode != rl_mode_prev_) {
    rl_mode_prev_ = rl_mode;
    RCLCPP_INFO(this->get_logger(), rl_mode
        ? "RL mode ENABLED (LB+RB held)"
        : "RL mode DISABLED (shoulder buttons released)");
  }

  // RL control only valid when the mode is on AND policy output is flowing.
  bool use_rl = (rl_mode && rl_active_);

  if (idle && !use_rl) {
    gait_->step(0.0, 0.0);
    // Zero-ctrl home stance (u=0 → crouch). Absolute target = JOINT_REF.
    std::copy_n(JOINT_REF, 18, cmd.data.begin());
  } else if (use_rl) {
    // RL mode: 18 leg joints from policy, lid stays Y-button.
    // Keep using the latest rl_action_ (rl_inference re-publishes continuously).
    std::copy_n(rl_action_.begin(), 18, cmd.data.begin());
  } else {
    // CPG fallback — also used when RL mode is on but no policy output yet.
    double omega = 2.0 * M_PI * speed / 0.15 + 4.0;
    gait_->step(omega, wz * 0.5);

    double amp = std::min(speed / 0.3, 1.0) * 0.4;
    for (int leg = 0; leg < 6; ++leg) {
      double ph = gait_->phase(leg);
      int j = leg * 3;
      // Oscillate around the crouched reference, not around fully-extended 0.
      cmd.data[j + 0] = JOINT_REF[j + 0] + amp * std::sin(ph * M_PI);
      cmd.data[j + 1] = JOINT_REF[j + 1] + ((ph > 0) ? 0.0 : amp * 0.5 * (1.0 + std::cos(ph * M_PI)));
      cmd.data[j + 2] = JOINT_REF[j + 2] + ((ph > 0) ? -amp * 0.3 * ph : amp * 0.3 * ph);
    }
  }
  cmd.data[18] = (xbox && xbox->get_button_y()) ? 0.08 : 0.0;
  low_cmd_pub_->publish(cmd);

  // Warn if RL mode requested but no /rl_action is arriving.
  if (rl_mode && !rl_active_) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "RL mode requested but no /rl_action received — is rl_inference.py running?");
  }

  if (++step_count_ % 200 == 0 && !idle) {
    const char* mode = rl_mode ? "RL" : "CPG";
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
        "%s[%s] v=%.2f amp=%.2f",
        gait_->name(), mode, speed, std::min(speed / 0.3, 1.0) * 0.4);
  }
}

// ─── main ───────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  std::string joy_path = ament_index_cpp::get_package_prefix("joy") + "/lib/joy/joy_node";
  pid_t pid = fork();
  if (pid == 0) { execl(joy_path.c_str(), "joy_node", (char*)nullptr); _exit(1); }

  auto node = std::make_shared<RobotController>();
  XboxController xbox(node.get());
  node->xbox = &xbox;
  RCLCPP_INFO(node->get_logger(), "Tripod CPG + joy_node ready.");
  rclcpp::spin(node);

  kill(pid, SIGTERM); waitpid(pid, nullptr, 0);
  rclcpp::shutdown();
  return 0;
}
