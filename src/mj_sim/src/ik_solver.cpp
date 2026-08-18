#include "mj_sim/ik_solver.hpp"
#include <algorithm>

bool LegIK::solve(double x, double y, double z, double angles[3])
{
  // Solve IK for one leg.
  // x,y,z: foot target in coxa frame; angles: output {coxa, thigh, tibia}
  angles[0] = std::atan2(-z, -y);   // coxa yaw from horizontal direction

  // Planar 2-link (thigh L1 + tibia L2) in the X-Y plane.
  const double r  = std::hypot(y, z);
  const double dX = x;
  const double dY = L0 - r;
  const double D  = std::hypot(dX, dY);

  const double d_max = L1 + L2;
  const double d_min = std::abs(L1 - L2);
  const bool reachable = (D >= d_min && D <= d_max);

  const double cos_q2 = std::clamp(
      (D * D - L1 * L1 - L2 * L2) / (2.0 * L1 * L2), -1.0, 1.0);
  const double q2 = -std::acos(cos_q2);   // negative = knee bent back

  const double q1 = std::atan2(dY, dX)
                  - std::atan2(L2 * std::sin(q2), L1 + L2 * std::cos(q2));

  // Map planar angles back to MuJoCo qpos.
  angles[1] = A10 - q1;
  angles[2] = q2 + OFFSET;

  return reachable;
}
