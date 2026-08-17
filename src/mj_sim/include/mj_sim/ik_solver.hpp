#pragma once
#include <cmath>

/// Inverse Kinematics for the cubot 3-DOF leg.
///
/// Coordinate convention (coxa frame "c1", calibrated via MuJoCo forward
/// kinematics — NOT the PhantomX convention this file used to assume):
///   +X = up (vertical, the coxa yaw axis, body +Z after c1 quaternion)
///   +Y = horizontal "outward" (the c1 frame's own +Y axis)
///   +Z = horizontal fore/aft  (the c1 frame's own +Z axis)
/// The leg extends along -Y at coxa angle 0.
///
/// Thigh and tibia are parallel pitch joints (both lie along the c1 frame's
/// ±Z axis), so thigh+tibia form a planar 2-link arm in the c1 X-Y plane and
/// the coxa rotates that plane around +X. This is the standard
/// yaw-coxa / pitch-femur / pitch-tibia hexapod layout.
///
/// Link geometry calibrated from cubot.xml + MuJoCo FK (all 6 legs identical):
///   L0 = |coxa axis -> thigh axis|  = 0.054   (c2 pos: 0, -0.054, 0 in c1)
///   L1 = |thigh axis -> tibia axis| = 0.0661  (tibia pos from thigh, FK)
///   L2 = |tibia axis -> foot tip|   = 0.1632  (FK at qpos=0)
///   A10    = thigh segment direction at thigh=0 (relative +X, c1 X-Y plane)
///   OFFSET = thigh<->tibia angle when both are 0 (straight leg)
struct LegIK {
  static constexpr double L0 = 0.054;      // coxa link length (m)
  static constexpr double L1 = 0.0661;     // femur link length (m)
  static constexpr double L2 = 0.1632;     // tibia link length, joint->foot tip (m)

  // Thigh segment direction at thigh=0, measured in the c1 X-Y plane via FK:
  //   thigh joint (0,-0.054,0) -> tibia joint (-0.0145,-0.1185,0)
  static constexpr double A10 = -1.791926;  // rad

  // Thigh<->tibia straight angle (both joints zero). Tibia at tibia=0 points
  // A20 = -2.955492; offset keeps the knee bent correctly for a hexapod.
  static constexpr double OFFSET = 1.163566;  // rad (A10 - A20)

  /// Solve IK for a single leg.
  /// @param x,y,z   foot target in the coxa frame (see convention above)
  /// @param angles  output: {coxa, thigh, tibia} absolute qpos in radians
  /// @return true if reachable, false if out of range (angles still set, clamped)
  static bool solve(double x, double y, double z, double angles[3]);
};
