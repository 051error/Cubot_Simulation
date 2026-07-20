#pragma once
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

class XboxController
{
public:
  XboxController(rclcpp::Node* node);

  std::pair<float, float> get_left_stick() const { return left_stick_; }
  std::pair<float, float> get_right_stick() const { return right_stick_; }
  std::vector<int> get_buttons() const { return buttons_; }
  bool is_pressed() const { return is_pressed_; }

private:
  void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg);

  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;

  std::pair<float, float> left_stick_;
  std::pair<float, float> right_stick_;
  std::vector<int> buttons_;
  bool is_pressed_ = false;
};
