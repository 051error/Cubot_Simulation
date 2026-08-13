#include "mj_sim/ik_solver.hpp"
#include <algorithm>

bool LegIK::solve(double x, double y, double z, double angles[3])
{
  // ── Step 1: Coxa angle θ0 ────────────────────────────────────────────
  // Leg extends along -Y at θ0=0. After coxa rotation θ0 around Z:
  //   foot_xy = R * (-sin θ0, -cos θ0)
  // Solving for θ0:  θ0 = atan2(-x, -y)
  angles[0] = std::atan2(-x, -y);

  // ── Step 2: Subtract coxa link, project onto leg direction ───────────
  // Coxa end at: (-L0·sin θ0, -L0·cos θ0)
  // Remaining vector from coxa end to foot target:
  const double c0 = std::cos(angles[0]);
  const double s0 = std::sin(angles[0]);
  const double dx = x + L0 * s0;   // x - (-L0·s0)
  const double dy = y + L0 * c0;   // y - (-L0·c0)
  // Radial distance along leg direction (-s0, -c0):
  const double R = -(dx * s0 + dy * c0);

  // ── Step 3: 2-link IK (femur L1 + tibia L2) in R-Z plane ───────────
  const double d = std::sqrt(R * R + z * z);
  const double d_max = L1 + L2;
  const double d_min = std::abs(L1 - L2);
  double d_use = std::clamp(d, d_min + 0.001, d_max - 0.001);

  // Law of cosines: tibia angle θ2 (elbow down convention)
  const double cos_val = (L1 * L1 + L2 * L2 - d_use * d_use) / (2.0 * L1 * L2);
  angles[2] = M_PI - std::acos(std::clamp(cos_val, -1.0, 1.0));

  // Femur angle θ1 = α - β
  const double alpha = std::atan2(z, R);            // target elevation
  const double beta  = std::atan2(L2 * std::sin(angles[2]),
                                   L1 + L2 * std::cos(angles[2]));  // knee bend offset
  angles[1] = alpha - beta;

  return d >= d_min && d <= d_max;
}
