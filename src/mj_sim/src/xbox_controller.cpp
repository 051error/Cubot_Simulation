#include "mj_sim/robot_ctrl.hpp"
#include "mj_sim/msg/up_cmd.hpp"

XboxController::XboxController(rclcpp::Node* node)
{
  upper_ctrl_pub_ = node->create_publisher<mj_sim::msg::UpCmd>(
      "/upper_ctrl", 10);

  joy_sub_ = node->create_subscription<sensor_msgs::msg::Joy>(
      "/joy", 10, [this](const sensor_msgs::msg::Joy::SharedPtr msg) {
        joy_callback(msg);
      });

  left_stick_  = {0.0f, 0.0f};
  right_stick_ = {0.0f, 0.0f};
}

void XboxController::joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg)
{
  // Xbox: 0=LX 1=LY 2=LT 3=RX 4=RY 5=RT
  if (msg->axes.size() >= 5) {
    left_stick_  = {msg->axes[0], msg->axes[1]};
    right_stick_ = {msg->axes[3], msg->axes[4]};
  }
  buttons_ = msg->buttons;
  is_pressed_ = (msg->buttons.size() >= 6) &&
                (msg->buttons[4] == 1 && msg->buttons[5] == 1);
  
  if (is_pressed_) {
    policy_mode = 1;  // RL
  } else {
    policy_mode = 0;
  }

  // Update action flags
  ac_a = (msg->buttons.size() > 0) ? (msg->buttons[0] == 1) : false;
  ac_b = (msg->buttons.size() > 1) ? (msg->buttons[1] == 1) : false;
  ac_x = (msg->buttons.size() > 2) ? (msg->buttons[2] == 1) : false;
  ac_y = (msg->buttons.size() > 3) ? (msg->buttons[3] == 1) : false;

  // Convert stick values to velocity commands
  // Left stick Y (axes[1]) → forward speed
  // Left stick X (axes[0]) → lateral speed
  // Right stick X (axes[2]) → angular speed
  linear_x_  = msg->axes[1] * kMaxLinearSpeed;
  linear_y_  = msg->axes[0] * kMaxLinearSpeed;
  angular_z_ = msg->axes[3] * kMaxAngularSpeed;  // right stick X → rotation

  // Publish high-level velocity command
  auto cmd = mj_sim::msg::UpCmd();
  cmd.linear_x   = linear_x_;
  cmd.linear_y   = linear_y_;
  cmd.angular_z  = angular_z_;
  cmd.policy_mode = policy_mode;
  cmd.ac_a = ac_a;
  cmd.ac_b = ac_b;
  cmd.ac_x = ac_x;
  cmd.ac_y = ac_y;
  upper_ctrl_pub_->publish(cmd);
}
