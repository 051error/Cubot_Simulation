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

// ── Turn-mode fixed CPG parameters (in-place rotation, no RL) ────────────
// The coxa swings a FIXED angle; the turn angular velocity is set by the
// cadence, so the gait frequency scales with |wz| (fixed angle × cadence).
static constexpr double TURN_AMP         = 0.25;             // fixed coxa swing angle (rad)
static constexpr double TURN_LIFT        = 0.20;             // swing foot lift height (rad, raised-cosine peak)
static constexpr double TIBIA_TUCK       = 0.15;             // swing-phase tibia fold (rad) for extra ground clearance
static constexpr double TURN_OMEGA_MIN   = 2.0 * M_PI * 0.8; // min cadence (0.8 Hz) — keep slow turns from stalling
static constexpr double TURN_OMEGA_MAX   = 2.0 * M_PI * 1.4; // max cadence (1.4 Hz)
static constexpr double TURN_DEADZONE    = 0.05;             // right-stick deadzone
static constexpr double TURN_SMOOTH_ALPHA= 0.06;             // first-order low-pass gain on wz (@200 Hz, ~80 ms)
static constexpr double TURN_DIR_SMOOTH  = 0.10;             // coxa direction ramp width through zero

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

  // ── Locomotion mode state machine: NORMAL / RL / TURN ───────────────
  //   LB+RB held → RL   (policy drives all 18 leg joints)
  //   RB only    → TURN (fixed-param CPG, in-place rotation)
  //   otherwise  → NORMAL (joystick-driven CPG walking)
  Mode mode = Mode::NORMAL;
  if (xbox) {
    bool lb = xbox->get_button_lb();
    bool rb = xbox->get_button_rb();
    if (lb && rb)      mode = Mode::RL;
    else if (rb)       mode = Mode::TURN;
  }

  // Edge-triggered mode-switch log.
  if (mode != mode_prev_) {
    const char* mname = (mode == Mode::RL) ? "RL" : (mode == Mode::TURN) ? "TURN" : "NORMAL";
    RCLCPP_INFO(this->get_logger(), "Mode: %s", mname);
    mode_prev_ = mode;
  }

  // RL control only valid when the mode is on AND policy output is flowing.
  bool use_rl = (mode == Mode::RL && rl_active_);

  switch (mode) {
    case Mode::RL:
      if (use_rl) {
        // RL mode: 18 leg joints from policy, lid stays Y-button.
        std::copy_n(rl_action_.begin(), 18, cmd.data.begin());
      } else {
        // RL requested but no policy output yet → hold crouch.
        gait_->step(0.0, 0.0);
        std::copy_n(JOINT_REF, 18, cmd.data.begin());
      }
      break;

    case Mode::TURN:
      // Independent CPG for in-place rotation (no RL).
      turn_step(cmd.data, wz);
      break;

    case Mode::NORMAL:
    default:
      if (idle) {
        gait_->step(0.0, 0.0);
        // Zero-ctrl home stance (u=0 → crouch). Absolute target = JOINT_REF.
        std::copy_n(JOINT_REF, 18, cmd.data.begin());
      } else {
        // Joystick-driven forward/lateral walking (also the pre-RL fallback).
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
      break;
  }
  cmd.data[18] = (xbox && xbox->get_button_y()) ? 0.08 : 0.0;
  low_cmd_pub_->publish(cmd);

  // Warn if RL mode requested but no /rl_action is arriving.
  if (mode == Mode::RL && !rl_active_) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "RL mode requested but no /rl_action received — is rl_inference.py running?");
  }

  if (++step_count_ % 200 == 0 && !idle) {
    const char* mname = (mode == Mode::RL) ? "RL" : (mode == Mode::TURN) ? "TURN" : "NORMAL";
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
        "%s[%s] v=%.2f wz=%.2f",
        gait_->name(), mname, speed, wz);
  }
}

// ─── Turn mode: independent CPG for in-place rotation ────────────────────
// No RL — fixed gait parameters rotate the body in place. The tripod CPG
// alternates stance/swing groups; stance and swing rotate the coxa in OPPOSITE
// directions. The coxa is a yaw joint, so a planted stance foot that sweeps its
// coxa pushes the body around yaw; the lifted swing foot sweeps the coxa back
// toward neutral so the coxa angle resets each cycle. Using the QUADRATURE
// component y (not phase x) makes the coxa sweep unidirectionally through
// stance and reverse through swing — a ratchet, not a back-and-forth (which
// produces zero net yaw, the failure of the previous front/rear version).
//
// Tripod groups {0,2,4} and {1,3,5} are π apart, so this automatically yields
// the requested pattern — e.g. for a left turn, {0,2,4} (stance) rotate the
// coxa one way while {1,3,5} (swing) rotate it the other way — and the signs
// reverse on the next half-cycle.
void RobotController::turn_step(std::vector<double>& joint_cmd, double wz)
{
  // Low-pass the stick so start/stop and speed changes ramp instead of jump.
  wz_filt_ += TURN_SMOOTH_ALPHA * (wz - wz_filt_);
  double wf = wz_filt_;
  double m  = std::abs(wf);

  if (m <= TURN_DEADZONE) {
    // Stick centred → hold the crouch.
    gait_->step(0.0, 0.0);
    std::copy_n(JOINT_REF, 18, joint_cmd.begin());
    return;
  }

  // Right stick X (wz): sign → turn direction, magnitude → turn angular rate.
  // Fixed coxa angle × cadence = angular velocity, so |wf| scales the cadence.
  // Stick left (wf<0) → turn left; stick right (wf>0) → turn right.
  double mag   = std::min(m, 1.0);
  double omega = TURN_OMEGA_MIN + (TURN_OMEGA_MAX - TURN_OMEGA_MIN) * mag;
  gait_->step(omega, 0.0);

  // Signed, full-amplitude coxa coefficient: ±1 away from zero, ramping
  // linearly through zero so direction changes don't snap.
  double k = wf / std::max(m, TURN_DIR_SMOOTH);

  for (int leg = 0; leg < 6; ++leg) {
    int j = leg * 3;
    // Quadrature y ∈ [-1,1]: monotonic through stance, reversed through swing,
    // so coxa pushes the body on the ground and resets toward neutral in air.
    double yy = gait_->orthogonal(leg);
    // Minus sign flips the coxa sweep so a rightward stick (k>0) turns the
    // body right.
    joint_cmd[j + 0] = JOINT_REF[j + 0] - k * TURN_AMP * yy;

    // Phase normalized to [-1,1] (raw x lives on the ±√MU limit cycle, not ±1).
    double ph_n = gait_->phase(leg) / std::sqrt(TripodGait::MU);

    // Swing: LIFT the foot. Verified against MuJoCo forward kinematics: the
    // foot tip rises when the thigh goes MORE NEGATIVE (rearward swing) and
    // when the tibia goes MORE POSITIVE (straighten). Raised-cosine envelope
    // keeps value AND slope zero at liftoff/touchdown (no impact spike).
    double lift = 0.0;
    double tuck = 0.0;
    if (ph_n < 0.0) {
      double s = -ph_n / 2.0;            // 0 (swing start/end) → 0.5 (swing middle)
      double env = 0.5 * (1.0 - std::cos(2.0 * M_PI * s));
      lift = -TURN_LIFT * env;    // thigh more negative → foot tip up
      tuck = +TIBIA_TUCK * env;   // tibia more positive → foot tip up & back
    }

    joint_cmd[j + 1] = JOINT_REF[j + 1] + lift;
    joint_cmd[j + 2] = JOINT_REF[j + 2] + tuck;
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
