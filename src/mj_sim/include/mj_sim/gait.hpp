#pragma once
#include <cmath>

// ═══════════════════════════════════════════════════════════════════════
//  Gait generation: CPG phase → foot target / joint angles.
//  Pure C++ (no ROS), used by RobotController for NORMAL and TURN modes.
// ═══════════════════════════════════════════════════════════════════════

// ── Gait generator (abstract) ────────────────────────────────────────────
struct GaitGenerator {
  static constexpr int N_LEGS = 6;
  virtual ~GaitGenerator() = default;

  virtual void step(double omega, double rot_bias) = 0;
  virtual double phase(int i) const = 0;       // >0 stance, <0 swing
  virtual double orthogonal(int i) const = 0;  // monotonic sweep, [-1,1]
  virtual const char* name() const = 0;
};

// ── Tripod gait (Hopf oscillator) ────────────────────────────────────────
struct TripodGait : GaitGenerator {
  static constexpr double DT    = 0.005;
  static constexpr double ALPHA = 5.0;
  static constexpr double MU    = 0.06;
  static constexpr double K_CPG = 2.0;

  double x[N_LEGS] = {};
  double y[N_LEGS] = {};

  TripodGait();
  void step(double omega, double rot_bias) override;
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

// ── Foot trajectory: CPG phase → foot target in body frame ──────────────
// Sweeps a straight line along the body axes (X=forward, Y=left, Z=up),
// opposite to vx/vy — no coxa arc, so no sideways leak on corner legs.
struct FootTrajectory {
  static constexpr double STRIDE_X  = 0.05;   // fore/aft sweep half-length (m)
  static constexpr double STRIDE_Y  = 0.04;   // lateral sweep half-length (m)
  static constexpr double LIFT      = 0.06;   // swing lift height (m)
  static constexpr double SPEED_REF = 0.05;   // speed (m/s) for full amplitude

  void compute(int leg, double phase, double orthogonal, double vx, double vy,
               double& fx, double& fy, double& fz);
};

// ── Per-leg mounting geometry (body frame) ──────────────────────────────
struct LegGeometry {
  double coxa_x, coxa_y, coxa_z;  // coxa origin in body frame
  double c1_y_x, c1_y_y;          // c1 +Y axis in body frame
};

// Calibrated via MuJoCo FK (leg order: rf, rm, rr, lf, lm, lr).
inline constexpr LegGeometry LEG_GEOM[6] = {
    { 0.1248,  -0.06164,  0.001116, -0.707107,  0.707107 },  // rf
    { 0.0,     -0.1034,   0.001116,  0.0,       1.0      },  // rm
    {-0.1248,  -0.06164,  0.001116,  0.707107,  0.707107 },  // rr
    { 0.1248,   0.06164,  0.001116, -0.707107, -0.707107 },  // lf
    { 0.0,      0.1034,   0.001116,  0.0,      -1.0      },  // lm
    {-0.1248,   0.06164,  0.001116,  0.707107, -0.707107 },  // lr
};

// Crouched reference qpos (coxa=0, thigh=-0.7593, tibia=-0.7108 ×6).
inline constexpr double JOINT_REF[18] = {
    0.0, -0.7593, -0.7108,  // rf
    0.0, -0.7593, -0.7108,  // rm
    0.0, -0.7593, -0.7108,  // rr
    0.0, -0.7593, -0.7108,  // lf
    0.0, -0.7593, -0.7108,  // lm
    0.0, -0.7593, -0.7108,  // lr
};

// ── Turn gait: fixed-param CPG for in-place rotation ─────────────────────
class TurnGait {
public:
  // Compute 18 leg joint targets for in-place rotation.
  // gait  CPG to advance; wz: turn rate (rad/s, sign=direction); cmd: output[18]
  void step(GaitGenerator& gait, double wz, double* cmd);

private:
  static constexpr double AMP          = 0.25;             // coxa swing (rad)
  static constexpr double LIFT         = 0.20;             // swing lift (rad)
  static constexpr double TIBIA_TUCK   = 0.15;             // swing tibia fold (rad)
  static constexpr double OMEGA_MIN    = 2.0 * M_PI * 0.8; // min cadence (rad/s)
  static constexpr double OMEGA_MAX    = 2.0 * M_PI * 1.4; // max cadence (rad/s)
  static constexpr double DEADZONE     = 0.05;
  static constexpr double SMOOTH_ALPHA = 0.06;             // wz low-pass gain
  static constexpr double DIR_SMOOTH   = 0.10;             // direction ramp width
  double wz_filt_ = 0.0;
};
