#pragma once
#include <cmath>

// Inverse kinematics for the 3-DOF leg (coxa frame).
// +X up, +Y outward, +Z fore/aft; leg extends along -Y at coxa 0.
struct LegIK {
  static constexpr double L0 = 0.054;        // coxa link length (m)
  static constexpr double L1 = 0.0661;       // femur link length (m)
  static constexpr double L2 = 0.1632;       // tibia link, joint→foot tip (m)
  static constexpr double A10 = -1.791926;   // thigh dir at thigh=0 (rad)
  static constexpr double OFFSET = 1.163566; // thigh↔tibia straight angle (rad)

  // Solve IK for one leg.
  // x,y,z   foot target in coxa frame
  // angles  output {coxa, thigh, tibia} qpos (rad)
  // returns true if reachable (clamped otherwise)
  static bool solve(double x, double y, double z, double angles[3]);
};
