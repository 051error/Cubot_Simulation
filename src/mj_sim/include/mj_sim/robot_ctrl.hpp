#pragma once
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "mj_sim/msg/up_cmd.hpp"
#include "mj_sim/ik_solver.hpp"
#include "mj_sim/gait.hpp"
#include <memory>
#include <string>
#include <array>
#include <vector>
#include <cmath>

// ═══════════════════════════════════════════════════════════════════════
//  Xbox controller: /joy → /upper_ctrl + getters
// ═══════════════════════════════════════════════════════════════════════

class XboxController
{
public:
  XboxController(rclcpp::Node* node);

  std::pair<float, float> get_left_stick()  const { return left_stick_; }
  std::pair<float, float> get_right_stick() const { return right_stick_; }
  std::vector<int> get_buttons() const { return buttons_; }
  bool is_pressed() const { return is_pressed_; }

  float get_linear_x()  const { return linear_x_; }
  float get_linear_y()  const { return linear_y_; }
  float get_angular_z() const { return angular_z_; }
  bool  get_button_y()  const { return ac_y; }
  bool  get_button_lb() const { return lb_; }   // left shoulder
  bool  get_button_rb() const { return rb_; }   // right shoulder
  int   get_policy_mode() const { return policy_mode; }

private:
  void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg);
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
  rclcpp::Publisher<mj_sim::msg::UpCmd>::SharedPtr upper_ctrl_pub_;

  std::pair<float, float> left_stick_, right_stick_;
  std::vector<int> buttons_;
  bool is_pressed_ = false;
  bool ac_a = false, ac_b = false, ac_x = false, ac_y = false;
  bool lb_ = false, rb_ = false;   // LB=buttons[4], RB=buttons[5]
  float linear_x_ = 0, linear_y_ = 0, angular_z_ = 0;
  int   policy_mode = 0;

  static constexpr float kMaxLinearSpeed  = 0.05f;   // m/s
  static constexpr float kMaxAngularSpeed = 1.0f;    // rad/s
};

// ═══════════════════════════════════════════════════════════════════════
//  Robot controller: mode state machine + ROS IO
// ═══════════════════════════════════════════════════════════════════════

class RobotController : public rclcpp::Node
{
public:
  RobotController();

  /// Switch gait at runtime
  void set_gait(std::unique_ptr<GaitGenerator> new_gait);

  /// Pointer to the Xbox controller (owned by main)
  XboxController* xbox = nullptr;

private:
  void timer_callback();
  void rl_action_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr low_cmd_pub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr rl_action_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::unique_ptr<GaitGenerator> gait_;
  FootTrajectory traj_;
  TurnGait turn_;
  int step_count_ = 0;

  // Mode: NORMAL = joystick CPG, RL = policy joints, TURN = in-place rotation.
  enum class Mode { NORMAL, RL, TURN };
  Mode mode_ = Mode::NORMAL;
  Mode mode_prev_ = Mode::NORMAL;

  bool rl_active_ = false;       // received at least one /rl_action message
  std::array<double, 18> rl_action_ = {};
};
