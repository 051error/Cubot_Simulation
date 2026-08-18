#include "mj_sim/gait.hpp"
#include <algorithm>

// ── Nominal foot position (body frame) at crouch ────────────────────────
// IK maps these back to JOINT_REF when the robot stands still.
static const double NOMINAL_FOOT[6][3] = {
    { 0.2289, -0.1679, -0.1238 },  // rf
    {-0.0016, -0.2522, -0.1238 },  // rm
    {-0.2311, -0.1657, -0.1238 },  // rr
    { 0.2311,  0.1657, -0.1238 },  // lf
    { 0.0016,  0.2522, -0.1238 },  // lm
    {-0.2289,  0.1679, -0.1238 },  // lr
};

// ── TripodGait ───────────────────────────────────────────────────────────

TripodGait::TripodGait()
{
  // Groups A={0,2,4}, B={1,3,5}, π apart.
  for (int i = 0; i < N_LEGS; ++i) {
    x[i] = (i == 0 || i == 2 || i == 4) ? 1.0 : -1.0;
    y[i] = 0.0;
  }
}

void TripodGait::step(double omega, double rot_bias)
{
  // Hopf oscillator: same-group attract, opposite-group repel.
  if (omega < 0.01) omega = 0.01;

  for (int i = 0; i < N_LEGS; ++i) {
    bool in_a_i = (i == 0 || i == 2 || i == 4);

    double coupling = 0.0;
    for (int j = 0; j < N_LEGS; ++j) {
      if (i == j) continue;
      bool in_a_j = (j == 0 || j == 2 || j == 4);
      double w = (in_a_i == in_a_j) ? +K_CPG : -K_CPG;
      coupling += w * (x[j] - x[i]);
    }
    coupling /= (N_LEGS - 1);

    double omega_i = omega + ((i >= 3) ? +rot_bias : -rot_bias);
    double r2 = x[i] * x[i] + y[i] * y[i];
    double dx = ALPHA * (MU - r2) * x[i] - omega_i * y[i] + coupling;
    double dy = ALPHA * (MU - r2) * y[i] + omega_i * x[i];

    x[i] += dx * DT;
    y[i] += dy * DT;
  }
}

// ── FootTrajectory ───────────────────────────────────────────────────────

void FootTrajectory::compute(int leg, double phase, double orthogonal,
                              double vx, double vy,
                              double& fx, double& fy, double& fz)
{
  // Compute foot target in body frame from CPG phase.
  // leg  0..5; phase: >0 stance, <0 swing; orthogonal: sweep [-1,1]; vx,vy: velocity
  const double* nom = NOMINAL_FOOT[leg];
  const double speed = std::hypot(vx, vy);

  const double ph = std::clamp(phase, -1.0, 1.0);
  const double yy = std::clamp(orthogonal, -1.0, 1.0);

  // Straight sweep opposite to velocity; yy back-sweeps once per stance.
  double sx = 0.0, sy = 0.0;
  if (speed > 1e-6) {
    const double g = std::min(speed / SPEED_REF, 1.0);
    sx = -STRIDE_X * (vx / speed) * g * yy;
    sy = -STRIDE_Y * (vy / speed) * g * yy;
  }

  fx = nom[0] + sx;
  fy = nom[1] + sy;
  fz = nom[2];

  if (ph < 0.0) {
    fz += LIFT * 0.5 * (1.0 - std::cos(M_PI * ph));  // raised-cosine lift
  }
}

// ── TurnGait ─────────────────────────────────────────────────────────────

void TurnGait::step(GaitGenerator& gait, double wz, double* cmd)
{
  // Compute 18 leg joint targets for in-place rotation.
  // gait: CPG to advance; wz: turn rate (sign=direction); cmd: output[18]
  wz_filt_ += SMOOTH_ALPHA * (wz - wz_filt_);
  double wf = wz_filt_;
  double m  = std::abs(wf);

  if (m <= DEADZONE) {
    gait.step(0.0, 0.0);
    std::copy_n(JOINT_REF, 18, cmd);
    return;
  }

  double mag   = std::min(m, 1.0);
  double omega = OMEGA_MIN + (OMEGA_MAX - OMEGA_MIN) * mag;
  gait.step(omega, 0.0);

  // Signed coxa coefficient, ramped through zero for smooth direction change.
  double k = wf / std::max(m, DIR_SMOOTH);

  for (int leg = 0; leg < 6; ++leg) {
    int j = leg * 3;
    double yy = gait.orthogonal(leg);
    cmd[j + 0] = JOINT_REF[j + 0] - k * AMP * yy;

    double ph_n = gait.phase(leg);
    double lift = 0.0, tuck = 0.0;
    if (ph_n < 0.0) {
      double s = -ph_n / 2.0;
      double env = 0.5 * (1.0 - std::cos(2.0 * M_PI * s));
      lift = -LIFT * env;        // thigh more negative → foot up
      tuck = +TIBIA_TUCK * env;  // tibia more positive → foot up & back
    }
    cmd[j + 1] = JOINT_REF[j + 1] + lift;
    cmd[j + 2] = JOINT_REF[j + 2] + tuck;
  }
}
