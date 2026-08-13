#pragma once
#include <cmath>

/// Inverse Kinematics for PhantomX 3-DOF leg.
///
/// Coordinate convention (coxa frame):
///   Z = coxa rotation axis (body up, after c1 quaternion maps joint X→body Z)
///   Leg extends along -Y at coxa angle 0.
///   Femur + Tibia form a 2-link arm in the RZ plane (R = radial distance in XY).
///
/// Link lengths calibrated from cubot.xml MJCF model geometry:
///   L0 = |c1→c2|       = 0.054   (c2 pos: 0, -0.054, 0 in c1 frame)
///   L1 = |thigh→tibia|  = 0.066   (tibia pos: 0, -0.0645, -0.0145 from thigh)
///   L2 = |tibia→foot_tip| ≈ 0.125 (estimated from tibia geom extent; needs
///                                   experimental verification via FK at joints=0)
struct LegIK {
  static constexpr double L0 = 0.054;    // coxa link length (m)
  static constexpr double L1 = 0.066;    // femur link length (m)
  static constexpr double L2 = 0.125;    // tibia link length, joint→foot tip (m)

  /// Solve IK for a single leg.
  /// @param x,y,z   foot target in coxa frame (see convention above)
  /// @param angles  output: {coxa, femur, tibia} in radians
  /// @return true if reachable, false if out of range (angles still set)
  static bool solve(double x, double y, double z, double angles[3]);
};
