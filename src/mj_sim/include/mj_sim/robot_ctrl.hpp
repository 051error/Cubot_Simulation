#pragma once
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "mj_sim/msg/up_cmd.hpp"
#include "mj_sim/ik_solver.hpp"
#include <memory>
#include <string>
#include <array>
#include <cmath>


// ═══════════════════════════════════════════════════════════════════════
//  Xbox Controller (unchanged)
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
  int   get_policy_mode() const { return policy_mode; }

private:
  void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg);
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
  rclcpp::Publisher<mj_sim::msg::UpCmd>::SharedPtr upper_ctrl_pub_;

  std::pair<float, float> left_stick_, right_stick_;
  std::vector<int> buttons_;
  bool is_pressed_ = false;
  bool ac_a = false, ac_b = false, ac_x = false, ac_y = false;
  float linear_x_ = 0, linear_y_ = 0, angular_z_ = 0;
  int   policy_mode = 0;

  static constexpr float kMaxLinearSpeed  = 0.05f;   // 5 cm/s (halved from 0.10)
  static constexpr float kMaxAngularSpeed = 1.0f;    // 1.0 rad/s (halved from 2.0)
};


// ═══════════════════════════════════════════════════════════════════════
//  Gait Generator — abstract base (swap implementations here)
// ═══════════════════════════════════════════════════════════════════════

struct GaitGenerator {
  static constexpr int N_LEGS = 6;
  virtual ~GaitGenerator() = default;

  /// Step oscillators forward.
  /// @param omega    base frequency (rad/s)
  /// @param rot_bias rotation bias breaking left/right symmetry
  virtual void step(double omega, double rot_bias) = 0;

  /// Phase signal for leg i: >0 = stance, <0 = swing
  virtual double phase(int i) const = 0;

  /// Human-readable name
  virtual const char* name() const = 0;
};


// ── Tripod Gait ────────────────────────────────────────────────────────

struct TripodGait : GaitGenerator {
  static constexpr double DT    = 0.005;   // 200 Hz
  static constexpr double ALPHA = 5.0;     // convergence
  static constexpr double MU    = 0.06;    // amplitude
  static constexpr double K_CPG = 2.0;     // coupling gain

  double x[N_LEGS] = {};
  double y[N_LEGS] = {};

  TripodGait();

  void step(double omega, double rot_bias) override;
  double phase(int i) const override { return x[i]; }
  const char* name() const override { return "Tripod"; }
};


// ═══════════════════════════════════════════════════════════════════════
//  Foot Trajectory: CPG phase → foot XYZ in coxa frame (gait-agnostic)
// ═══════════════════════════════════════════════════════════════════════

struct FootTrajectory {
  static constexpr double STRIDE   = 0.06;
  static constexpr double LIFT     = 0.03;
  static constexpr double Y_OFFSET = 0.18;   // extend further outward
  static constexpr double Z_OFFSET = -0.10;  // less depth below coxa

  void compute(int leg, double phase_i, double vx, double vy, double rot,
               double& fx, double& fy, double& fz);
};


// ═══════════════════════════════════════════════════════════════════════
//  Robot Controller
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
  int step_count_ = 0;

  bool rl_active_ = false;       // received at least one /rl_action message
  bool rl_mode_prev_ = false;    // previous RL-mode state, for edge-triggered logging
  std::array<double, 18> rl_action_ = {};
};
