#include "mj_sim/robot_ctrl.hpp"
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>

RobotController::RobotController() : Node("robot_controller")
{
  // Subscriptions, publishers etc. will be added here
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  // Launch joy_node in a child process
  pid_t pid = fork();
  if (pid == 0) {
    // Child: run joy_node
    execl("/opt/ros/jazzy/lib/joy/joy_node", "joy_node", (char*)nullptr);
    _exit(1);  // only reached if execl fails
  }

  auto node = std::make_shared<RobotController>();
  XboxController xbox(node.get());
  rclcpp::spin(node);

  // Cleanup: kill joy_node child on exit
  kill(pid, SIGTERM);
  waitpid(pid, nullptr, 0);
  rclcpp::shutdown();
  return 0;
}