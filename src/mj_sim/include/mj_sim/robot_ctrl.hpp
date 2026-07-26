#pragma once
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "mj_sim/msg/up_cmd.hpp"


class XboxController
{
public:
  XboxController(rclcpp::Node* node);

  std::pair<float, float> get_left_stick()  const { return left_stick_; }
  std::pair<float, float> get_right_stick() const { return right_stick_; }
  std::vector<int> get_buttons() const { return buttons_; }
  bool is_pressed() const { return is_pressed_; }

  // High-level velocity commands (computed from sticks)
  float get_linear_x()  const { return linear_x_; }
  float get_linear_y()  const { return linear_y_; }
  float get_angular_z() const { return angular_z_; }

private:
  void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg);

  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
  rclcpp::Publisher<mj_sim::msg::UpCmd>::SharedPtr upper_ctrl_pub_;

  std::pair<float, float> left_stick_;
  std::pair<float, float> right_stick_;
  std::vector<int> buttons_;

  bool is_pressed_ = false;
  bool ac_a = false;
  bool ac_b = false;
  bool ac_x = false;
  bool ac_y = false;

  // Velocity commands (m/s and rad/s)
  float linear_x_  = 0.0f;
  float linear_y_  = 0.0f;
  float angular_z_ = 0.0f;
  int policy_mode = 0;

  // Max speed limits
  static constexpr float kMaxLinearSpeed  = 0.3f;   // 0.3 m/s ≈ 1 km/h
  static constexpr float kMaxAngularSpeed = 2.0f;   // 2.0 rad/s ≈ 115°/s
};


class RobotController : public rclcpp::Node
{
public:
  RobotController();
};