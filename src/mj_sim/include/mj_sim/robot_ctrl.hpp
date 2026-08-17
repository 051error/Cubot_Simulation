#pragma once
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "mj_sim/msg/up_cmd.hpp"
#include "mj_sim/ik_solver.hpp"
#include <memory>
#include <string>
#include <array>
#include <vector>
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

  /// Quadrature signal for leg i, normalized to [-1,1]. Orthogonal to phase():
  /// it sweeps monotonically through stance and reverses through swing, which
  /// is the waveform needed for unidirectional (ratchet) in-place turning.
  virtual double orthogonal(int i) const = 0;

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
  // Phase signals are normalized by the ACTUAL oscillator radius r=√(x²+y²),
  // NOT by √MU. The linear inter-leg coupling term adds energy and pushes the
  // limit cycle radius above √MU (~0.56 instead of 0.245), so normalizing by
  // √MU would clamp/saturate and cause step jumps. x/r and y/r are the true
  // phase (cos θ) and quadrature (sin θ) regardless of radius.
  double phase(int i) const override {
    double r = std::hypot(x[i], y[i]);
    return (r > 1e-9) ? x[i] / r : 1.0;
  }
  double orthogonal(int i) const override {
    double r = std::hypot(x[i], y[i]);
    return (r > 1e-9) ? y[i] / r : 0.0;
  }
  const char* name() const override { return "Tripod"; }
};


// ═══════════════════════════════════════════════════════════════════════
//  Foot Trajectory: CPG phase → foot XYZ in BODY frame (gait-agnostic)
// ═══════════════════════════════════════════════════════════════════════
//
//  Outputs the foot target in the body frame (X=forward, Y=left, Z=up), so
//  the velocity commands map directly onto body axes.
//
//  The foot sweeps along an ARC centred on the coxa axis (radial distance held
//  at the nominal value) rather than along a body-frame straight line. This is
//  what makes locomotion coxa-dominant: a planted stance foot sweeps its coxa
//  sideways against the ground (the same ratchet mechanism as turn mode), while
//  the thigh/tibia only maintain height + lift. Sweeping a straight line instead
//  would split the motion between coxa AND thigh/tibia, so the thigh/tibia would
//  also push radially ("digging") and the body bounces at high command magnitude.
//
//  Fore/aft (vx) is produced by the coxa swing angle; lateral (vy) is produced
//  by a radial shift, because the middle legs' coxa tangent points fore/aft and
//  physically cannot step sideways — lateral motion must lengthen/shorten the
//  leg. The two terms are scaled independently by |vx|/speed and |vy|/speed so
//  a diagonal command blends them without cross-coupling.
//
//  Two CPG signals are used, each with a distinct role:
//   - phase (cos component)  → stance/swing switch (>0 stance, <0 swing) and the
//     swing LIFT envelope. It is NOT monotonic within a stance (0 → ±1 → 0), so
//     it must never drive a sweep directly.
//   - orthogonal (sin component) → the sweep waveform. It IS monotonic within a
//     stance (-1 → +1) and reverses through swing, which is the unidirectional
//     (ratchet) waveform the coxa swing needs. Using phase here instead would
//     make the coxa swing back and forth mid-stance instead of sweeping once.
//
//  vx/vy are body-frame velocities in m/s (vx forward+, vy left+).

struct FootTrajectory {
  // Amplitude is capped by two constraints, not by desired speed:
  //  - SWING_AMP too large => the stance foot sweeps an arc far longer than the
  //    body actually advances (0.149 m arc vs 0.026 m/step at full speed), so the
  //    foot slips ~80% and the leading legs lose ground contact -> weak drive.
  //  - RADIAL too large => during swing the foot shortens toward the coxa while
  //    lifting, driving the 2-link distance below the leg's reachable minimum
  //    (D=0.092 < d_min=0.097) -> 306 IK-clamped samples/cycle and a 127 deg thigh
  //    sweep (maxjump 0.32 rad) that makes the body bounce sideways.
  //  0.36 / 0.04 keep the reachable-margin positive and the thigh sweep to ~92 deg.
  static constexpr double SWING_AMP = 0.36;   // max coxa swing angle (rad) for fore/aft
  static constexpr double RADIAL    = 0.04;   // max radial shift (m) for lateral
  static constexpr double LIFT      = 0.06;   // swing lift height (m)
  static constexpr double SPEED_REF = 0.05;   // speed (m/s) for full amplitude

  void compute(int leg, double phase, double orthogonal, double vx, double vy,
               double& fx, double& fy, double& fz);
};

// Per-leg mounting geometry (body frame). Used to convert a body-frame foot
// target into the coxa frame for LegIK.
struct LegGeometry {
  double coxa_x, coxa_y, coxa_z;   // coxa (c1) origin in body frame
  double c1_y_x, c1_y_y;           // c1 +Y axis in body frame (horizontal)
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
  void turn_step(std::vector<double>& joint_cmd, double wz);

  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr low_cmd_pub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr rl_action_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::unique_ptr<GaitGenerator> gait_;
  FootTrajectory traj_;
  int step_count_ = 0;

  // ── Locomotion mode state machine ────────────────────────────────────
  //   NORMAL  joystick-driven CPG walking
  //   RL      policy drives all 18 leg joints (LB+RB held)
  //   TURN    fixed-param CPG in-place rotation (RB only held)
  enum class Mode { NORMAL, RL, TURN };
  Mode mode_ = Mode::NORMAL;
  Mode mode_prev_ = Mode::NORMAL;

  bool rl_active_ = false;       // received at least one /rl_action message
  std::array<double, 18> rl_action_ = {};
  double wz_filt_ = 0.0;         // low-passed turn command (turn_step smoothing)
};
