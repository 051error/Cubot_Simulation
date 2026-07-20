#include "mj_sim/low_level_ctrl.hpp"

XboxController::XboxController(rclcpp::Node* node)
{
  joy_sub_ = node->create_subscription<sensor_msgs::msg::Joy>(
      "/joy", 10, [this](const sensor_msgs::msg::Joy::SharedPtr msg) {
        joy_callback(msg);
      });

  // init with default values
  left_stick_  = {0.0f, 0.0f};
  right_stick_ = {0.0f, 0.0f};
}

void XboxController::joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg)
{
  if (msg->axes.size() >= 4) {
    left_stick_  = {msg->axes[0], msg->axes[1]};
    right_stick_ = {msg->axes[2], msg->axes[3]};
  }
  buttons_    = msg->buttons;
  is_pressed_ = (msg->buttons.size() >= 6) &&
                (msg->buttons[4] == 1 && msg->buttons[5] == 1);
}
