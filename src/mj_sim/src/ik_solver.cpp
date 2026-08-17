#include "mj_sim/ik_solver.hpp"
#include <algorithm>

bool LegIK::solve(double x, double y, double z, double angles[3])
{
  // ── Step 1: coxa yaw θ0 ──────────────────────────────────────────────
  // The leg lies in the c1 X-Y plane at θ0=0 and extends along -Y. Rotating
  // the coxa by θ0 around +X sweeps that plane through the horizontal Y-Z
  // directions. The horizontal direction of the foot (y, z) is therefore set
  // by the coxa alone:
  //     -cos(θ0) = y / r,  -sin(θ0) = z / r   (r = hypot(y, z))
  //   => θ0 = atan2(-z, -y)
  angles[0] = std::atan2(-z, -y);

  // ── Step 2: planar 2-link (thigh L1 + tibia L2) ──────────────────────
  // After removing the coxa yaw, the foot sits at radial distance r in the
  // X-Y plane. The thigh joint is offset from the coxa axis by L0 along -Y,
  // so relative to the thigh joint the target is:
  //     dX = x            (vertical, +up)
  //     dY = L0 - r       (radial: r is positive outward, L0 inward)
  const double r  = std::hypot(y, z);
  const double dX = x;
  const double dY = L0 - r;
  const double D  = std::hypot(dX, dY);

  const double d_max = L1 + L2;
  const double d_min = std::abs(L1 - L2);
  const bool reachable = (D >= d_min && D <= d_max);

  // Law of cosines for the knee bend (q2 = tibia angle relative to thigh,
  // elbow-down convention -> negative). Clamp so a slightly-out-of-range
  // target degrades to a fully-straight / fully-folded leg instead of NaN.
  const double cos_q2 = std::clamp(
      (D * D - L1 * L1 - L2 * L2) / (2.0 * L1 * L2), -1.0, 1.0);
  const double q2 = -std::acos(cos_q2);   // negative = knee bent back

  // Thigh angle relative to +X (vertical-up). Standard 2-link solution in the
  // (dX=vertical, dY=radial) plane:
  //     q1 = atan2(dY, dX) - atan2(L2·sin q2, L1 + L2·cos q2)
  const double q1 = std::atan2(dY, dX)
                  - std::atan2(L2 * std::sin(q2), L1 + L2 * std::cos(q2));

  // ── Step 3: map planar angles back to MuJoCo joint qpos ──────────────
  // Forward model (FK-verified): thigh direction α1 = A10 - θ1, so
  //     θ1 = A10 - q1
  // and tibia direction α2 = α1 - OFFSET + θ2, so
  //     θ2 = q2 + OFFSET
  angles[1] = A10 - q1;
  angles[2] = q2 + OFFSET;

  return reachable;
}
