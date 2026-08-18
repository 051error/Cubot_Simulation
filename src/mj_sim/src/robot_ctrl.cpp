#include "mj_sim/robot_ctrl.hpp"
#include "ament_index_cpp/get_package_prefix.hpp"
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <algorithm>

// ─── RobotController ────────────────────────────────────────────────────

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

  // Mode: LB+RB = RL, RB only = TURN, otherwise NORMAL.
  Mode mode = Mode::NORMAL;
  if (xbox) {
    bool lb = xbox->get_button_lb();
    bool rb = xbox->get_button_rb();
    if (lb && rb)      mode = Mode::RL;
    else if (rb)       mode = Mode::TURN;
  }

  if (mode != mode_prev_) {
    const char* mname = (mode == Mode::RL) ? "RL" : (mode == Mode::TURN) ? "TURN" : "NORMAL";
    RCLCPP_INFO(this->get_logger(), "Mode: %s", mname);
    mode_prev_ = mode;
  }

  bool use_rl = (mode == Mode::RL && rl_active_);

  switch (mode) {
    case Mode::RL:
      if (use_rl) {
        std::copy_n(rl_action_.begin(), 18, cmd.data.begin());
      } else {
        gait_->step(0.0, 0.0);
        std::copy_n(JOINT_REF, 18, cmd.data.begin());
      }
      break;

    case Mode::TURN:
      turn_.step(*gait_, wz, cmd.data.data());
      break;

    case Mode::NORMAL:
    default:
      if (idle) {
        gait_->step(0.0, 0.0);
        std::copy_n(JOINT_REF, 18, cmd.data.begin());
      } else {
        // CPG → foot target → IK. Stick up (axes[1]<0) = forward +X,
        // left (axes[0]<0) = left +Y; right stick ignored in NORMAL.
        double bvx = -vx;
        double bvy = -vy;
        double bspd = std::hypot(bvx, bvy);

        double omega = 2.0 * M_PI * bspd / 0.15 + 4.0;
        gait_->step(omega, 0.0);

        for (int leg = 0; leg < 6; ++leg) {
          double ph = gait_->phase(leg);
          double yy = gait_->orthogonal(leg);

          double fx, fy, fz;
          traj_.compute(leg, ph, yy, bvx, bvy, fx, fy, fz);

          // Body → coxa frame: c1 +X = body +Z, c1 +Z = (-c1_y_y, c1_y_x, 0).
          const LegGeometry& g = LEG_GEOM[leg];
          double dx = fx - g.coxa_x;
          double dy = fy - g.coxa_y;
          double dz = fz - g.coxa_z;
          double c1x = dz;
          double c1y = g.c1_y_x * dx + g.c1_y_y * dy;
          double c1z = -g.c1_y_y * dx + g.c1_y_x * dy;

          double ang[3];
          LegIK::solve(c1x, c1y, c1z, ang);
          int j = leg * 3;
          cmd.data[j + 0] = ang[0];
          cmd.data[j + 1] = ang[1];
          cmd.data[j + 2] = ang[2];
        }
      }
      break;
  }
  cmd.data[18] = (xbox && xbox->get_button_y()) ? 0.08 : 0.0;
  low_cmd_pub_->publish(cmd);

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
