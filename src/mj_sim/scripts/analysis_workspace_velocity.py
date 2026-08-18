#!/usr/bin/env python3
"""Foot workspace analysis and CPG max velocity calculation.

Task 1: sample joint range via FK to characterize the foot workspace.
Task 2: compute max foot velocity during a CPG cycle.
"""

import numpy as np
import sys
import os

# Add leg_ik.py to path if not already importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from leg_ik import _leg_fk_batch, _leg_fk

# ──────────────────────────────────────────────────────────────────────────────
# Common parameters
# ──────────────────────────────────────────────────────────────────────────────
JOINT_MAX = 2.6179939  # rad (~150 deg), from cubot.xml
JOINT_RES = 0.05       # rad — finer than IK grid for workspace analysis

# c1_rest frame convention:
#   X = vertical (positive up)
#   Y = forward (outward from hip)
#   Z = lateral


def task1_workspace_analysis():
    """Task 1: Compute foot reachable workspace in c1_rest frame.

    Samples joints over the full [-JOINT_MAX, JOINT_MAX] range with JOINT_RES
    resolution, runs vectorized FK, and reports per-axis statistics.
    """
    print("=" * 72)
    print("TASK 1: FOOT WORKSPACE ANALYSIS (c1_rest frame)")
    print("=" * 72)

    # ── Build joint sampling grid ─────────────────────────────────────────
    jv = np.arange(-JOINT_MAX, JOINT_MAX + JOINT_RES / 2, JOINT_RES)
    n = len(jv)
    total = n ** 3
    print(f"\nJoint range: [{jv[0]:.4f}, {jv[-1]:.4f}] rad  "
          f"({np.degrees(jv[0]):.1f}° to {np.degrees(jv[-1]):.1f}°)")
    print(f"Resolution:  {JOINT_RES:.3f} rad  ({np.degrees(JOINT_RES):.1f}°)")
    print(f"Grid size:   {n}³ = {total:,} points")

    T0, T1, T2 = np.meshgrid(jv, jv, jv, indexing="ij")
    t0f, t1f, t2f = T0.ravel(), T1.ravel(), T2.ravel()

    # ── Run FK in chunks (avoid OOM) ──────────────────────────────────────
    chunk_size = 200_000
    num_chunks = int(np.ceil(total / chunk_size))
    results = []
    print(f"\nComputing FK in {num_chunks} chunk(s) of ~{chunk_size:,} points...")

    for c in range(num_chunks):
        s = c * chunk_size
        e = min(s + chunk_size, total)
        p = _leg_fk_batch(t0f[s:e], t1f[s:e], t2f[s:e])
        results.append(p)
        if num_chunks > 1:
            print(f"  Chunk {c + 1}/{num_chunks}: {e - s:,} points done")

    pts = np.concatenate(results, axis=0)
    print(f"\nTotal FK evaluations: {len(pts):,}")

    # ── Per-axis statistics ───────────────────────────────────────────────
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    print(f"\n{'─' * 60}")
    print(f"{'Axis':>6} | {'Min (m)':>10} | {'Max (m)':>10} | {'Range (m)':>10} | "
          f"{'Mean (m)':>10} | {'Median (m)':>10}")
    print(f"{'─' * 60}")
    for label, arr in [("X (vert)", x), ("Y (fwd)", y), ("Z (lat)", z)]:
        print(f"{label:>10} | {arr.min():10.4f} | {arr.max():10.4f} | "
              f"{np.ptp(arr):10.4f} | {arr.mean():10.4f} | "
              f"{np.median(arr):10.4f}")

    # ── Workspace extent ──────────────────────────────────────────────────
    bb_min = pts.min(axis=0)
    bb_max = pts.max(axis=0)
    bb_size = bb_max - bb_min

    print(f"\n{'─' * 60}")
    print("Bounding box (c1_rest):")
    print(f"  X: [{bb_min[0]:.4f}, {bb_max[0]:.4f}] m  (vertical)")
    print(f"  Y: [{bb_min[1]:.4f}, {bb_max[1]:.4f}] m  (forward)")
    print(f"  Z: [{bb_min[2]:.4f}, {bb_max[2]:.4f}] m  (lateral)")
    print(f"  Volume: {bb_size[0] * bb_size[1] * bb_size[2]:.6f} m³")

    # ── Radial distance from c1 origin ────────────────────────────────────
    r = np.sqrt(x**2 + y**2 + z**2)
    print(f"\nRadial distance from c1 origin:")
    print(f"  Min:  {r.min():.4f} m")
    print(f"  Max:  {r.max():.4f} m")
    print(f"  Mean: {r.mean():.4f} m")

    # ── Reachable area at stance height ───────────────────────────────────
    stance_h = 0.15  # default stance height (from CPG)
    tol = 0.005      # tolerance for slice
    near_stance = pts[(x >= -stance_h - tol) & (x <= -stance_h + tol)]
    if len(near_stance) > 0:
        print(f"\nReachable envelope at stance height (X ≈ {-stance_h:.2f}m ± {tol*1000:.0f}mm):")
        print(f"  Points in slice: {len(near_stance):,}")
        print(f"  Y range: [{near_stance[:,1].min():.4f}, {near_stance[:,1].max():.4f}] m")
        print(f"  Z range: [{near_stance[:,2].min():.4f}, {near_stance[:,2].max():.4f}] m")

    # ── Nominal foot position from CPG ────────────────────────────────────
    foot_y_neutral = -0.02
    foot_z_lat = 0.0
    foot_x_vert = -stance_h
    dist_to_nominal = np.sqrt((x - foot_x_vert)**2 + (y - foot_y_neutral)**2 + (z - foot_z_lat)**2)
    min_dist_idx = np.argmin(dist_to_nominal)
    print(f"\nClosest FK point to nominal CPG foot position "
          f"({foot_x_vert:.3f}, {foot_y_neutral:.3f}, {foot_z_lat:.3f}):")
    print(f"  Position: ({pts[min_dist_idx, 0]:.4f}, {pts[min_dist_idx, 1]:.4f}, {pts[min_dist_idx, 2]:.4f}) m")
    print(f"  Distance: {dist_to_nominal[min_dist_idx]:.4f} m")
    print(f"  Joints θ0,θ1,θ2: ({t0f[min_dist_idx]:.4f}, {t1f[min_dist_idx]:.4f}, {t2f[min_dist_idx]:.4f}) rad")
    noms = _leg_fk_batch(t0f[min_dist_idx:min_dist_idx+1],
                         t1f[min_dist_idx:min_dist_idx+1],
                         t2f[min_dist_idx:min_dist_idx+1])
    print(f"  FK verify: ({noms[0, 0]:.4f}, {noms[0, 1]:.4f}, {noms[0, 2]:.4f}) m")

    return pts


def task2_velocity_analysis():
    """Task 2: Max foot velocity during CPG gait cycle.

    Uses the Hopf-oscillator dynamics from hexapod_cpg.py to compute
    foot velocity in c1_rest frame (Y and Z components).
    """
    print(f"\n\n{'=' * 72}")
    print("TASK 2: MAX FOOT VELOCITY DURING CPG GAIT CYCLE")
    print("=" * 72)

    # ── CPG parameters (best values from hexapod_cpg.py) ──────────────────
    step_amp = 0.07    # m
    lift_h   = 0.04    # m
    stance_h = 0.15    # m
    freq     = 1.5     # Hz
    alpha    = 50.0    # Hopf convergence rate
    dt_cpg   = 0.02    # s (50 Hz CPG rate)

    omega = 2.0 * np.pi * freq  # angular frequency (rad/s)
    T     = 1.0 / freq          # period (s)

    print(f"\nCPG Parameters:")
    print(f"  step_amp = {step_amp:.3f} m     (Y-direction oscillation amplitude)")
    print(f"  lift_h   = {lift_h:.3f} m     (Z-direction swing lift height)")
    print(f"  stance_h = {stance_h:.3f} m     (Z-direction stance depth)")
    print(f"  freq     = {freq:.2f} Hz     (walking frequency)")
    print(f"  omega    = {omega:.4f} rad/s  (angular frequency)")
    print(f"  Period T = {T:.4f} s")
    print(f"  CPG dt   = {dt_cpg:.3f} s      (50 Hz)")

    # ── Simulate one full cycle of a single Hopf oscillator ───────────────
    # Start at (x=1, y=0) = swing start (Group 0), no coupling
    n_steps = int(5 * T / dt_cpg)  # simulate 5 cycles, enough to converge
    t = np.arange(n_steps) * dt_cpg
    x_hist = np.zeros(n_steps)
    y_hist = np.zeros(n_steps)

    x_hist[0] = 1.0
    y_hist[0] = 0.0

    for i in range(1, n_steps):
        r2 = x_hist[i-1]**2 + y_hist[i-1]**2
        dx = alpha * (1.0 - r2) * x_hist[i-1] - omega * y_hist[i-1]
        dy = alpha * (1.0 - r2) * y_hist[i-1] + omega * x_hist[i-1]
        x_hist[i] = x_hist[i-1] + dx * dt_cpg
        y_hist[i] = y_hist[i-1] + dy * dt_cpg

    # ── Use the last cycle (steady-state on limit cycle) for analysis ─────
    # Find zero-crossings of y (stance→swing transitions)
    last_cycle_start = int((n_steps * 3 // 5))  # skip first 3 cycles
    yy = y_hist[last_cycle_start:]
    xx = x_hist[last_cycle_start:]
    tt = t[last_cycle_start:]

    # ── Foot trajectory mapping ───────────────────────────────────────────
    # Y: foot_y = FOOT_Y_NEUTRAL + step_amp * x
    # Z: foot_z = -stance_h + lift_h * max(0, y)
    foot_y = -0.02 + step_amp * xx
    foot_z = -stance_h + lift_h * np.maximum(0, yy)

    # ── Velocity via finite differences ───────────────────────────────────
    dt_actual = tt[1] - tt[0]  # should be dt_cpg
    vel_y = np.gradient(foot_y, dt_actual)
    vel_z = np.gradient(foot_z, dt_actual)
    vel_mag = np.sqrt(vel_y**2 + vel_z**2)

    # ── Separate stance and swing phases ──────────────────────────────────
    stance_mask = yy < 0
    swing_mask  = yy >= 0

    print(f"\n{'─' * 70}")
    print("Hopf Oscillator Analysis (steady-state, last ~2 cycles):")
    print(f"{'─' * 70}")

    # ── Y-direction (forward) velocity ────────────────────────────────────
    # Analytically: d(foot_y)/dt = step_amp * dx/dt
    # On limit cycle (r≈1): dx/dt ≈ -omega * y
    # So max |d(foot_y)/dt| = step_amp * omega  (when |y|=1)
    v_y_max_theory = step_amp * omega  # theoretical max

    print(f"\n  Y-direction (forward) foot velocity:")
    print(f"    Theoretical max |v_y| = step_amp * ω = {step_amp:.3f} * {omega:.3f} = {v_y_max_theory:.4f} m/s")
    print(f"    Simulated  max |v_y| = {abs(vel_y).max():.4f} m/s")
    print(f"    Simulated  |v_y| during stance: max = {abs(vel_y[stance_mask]).max():.4f} m/s, "
          f"mean = {abs(vel_y[stance_mask]).mean():.4f} m/s")
    print(f"    Simulated  |v_y| during swing:  max = {abs(vel_y[swing_mask]).max():.4f} m/s, "
          f"mean = {abs(vel_y[swing_mask]).mean():.4f} m/s")

    # ── Z-direction (vertical) velocity ───────────────────────────────────
    # On limit cycle: dz/dt = lift_h * dy/dt = lift_h * omega * x  (swing only)
    v_z_max_theory = lift_h * omega  # when |x|=1 during swing

    print(f"\n  Z-direction (vertical/lift) foot velocity:")
    print(f"    Theoretical max |v_z| = lift_h * ω = {lift_h:.3f} * {omega:.3f} = {v_z_max_theory:.4f} m/s  (swing only, at |x|=1)")
    print(f"    Simulated  max |v_z| = {abs(vel_z).max():.4f} m/s")
    print(f"    Simulated  |v_z| during stance: max = {abs(vel_z[stance_mask]).max():.4f} m/s, "
          f"mean = {abs(vel_z[stance_mask]).mean():.4f} m/s")
    print(f"    Simulated  |v_z| during swing:  max = {abs(vel_z[swing_mask]).max():.4f} m/s, "
          f"mean = {abs(vel_z[swing_mask]).mean():.4f} m/s")

    # ── Combined velocity ─────────────────────────────────────────────────
    print(f"\n  Total velocity magnitude |v| = sqrt(v_y² + v_z²):")
    print(f"    Theoretical max |v| = sqrt({v_y_max_theory:.4f}² + {v_z_max_theory:.4f}²) "
          f"= {np.sqrt(v_y_max_theory**2 + v_z_max_theory**2):.4f} m/s")
    print(f"    Simulated  max |v| = {vel_mag.max():.4f} m/s")
    print(f"    Simulated  |v| during stance: max = {vel_mag[stance_mask].max():.4f} m/s, "
          f"mean = {vel_mag[stance_mask].mean():.4f} m/s")
    print(f"    Simulated  |v| during swing:  max = {vel_mag[swing_mask].max():.4f} m/s, "
          f"mean = {vel_mag[swing_mask].mean():.4f} m/s")

    # ── Phase breakdown ───────────────────────────────────────────────────
    # stance: y < 0  → x goes from -1 to +1
    # swing:  y >= 0 → x goes from +1 to -1
    stance_duration = stance_mask.mean() * T
    swing_duration  = swing_mask.mean() * T

    print(f"\n  Gait phase timing (per leg, per cycle):")
    print(f"    Stance duration: ~{stance_duration:.4f} s  (y < 0, x: -1 → +1)")
    print(f"    Swing  duration: ~{swing_duration:.4f} s  (y ≥ 0, x: +1 → -1)")
    print(f"    Duty factor:     ~{stance_duration / T:.2f}")

    # ── First-principles max velocity check ───────────────────────────────
    # During stance: x traverses [-1, +1] in T/2
    # Δfoot_y = step_amp * 2.0
    # v_y_avg_stance = step_amp * 2.0 / (T/2) = step_amp * 4.0 / T
    v_y_avg_stance_analytic = step_amp * 4.0 / T
    print(f"\n  First-principles check (stance phase):")
    print(f"    Δfoot_y during stance = step_amp * 2 = {step_amp * 2:.4f} m")
    print(f"    Δt stance = T/2 = {T/2:.4f} s")
    print(f"    v_y_avg (stance) = Δy / Δt = {v_y_avg_stance_analytic:.4f} m/s")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  SUMMARY: MAX FOOT VELOCITIES")
    print(f"{'─' * 70}")
    print(f"  {'Component':<25} {'Max (m/s)':>12} {'Phase':>12} {'Method':>18}")
    print(f"  {'─' * 67}")
    print(f"  {'Y (forward, stance)':<25} {v_y_max_theory:12.4f} {'stance':>12} {'analytic/theory':>18}")
    print(f"  {'Y (forward, swing)':<25} {v_y_max_theory:12.4f} {'swing':>12} {'analytic/theory':>18}")
    print(f"  {'Z (vertical/lift)':<25} {v_z_max_theory:12.4f} {'swing':>12} {'analytic/theory':>18}")
    print(f"  {'Z (vertical/lift)':<25} {0.0:12.4f} {'stance':>12} {'foot on ground':>18}")
    print(f"  {'Combined |v|':<25} {np.sqrt(v_y_max_theory**2 + v_z_max_theory**2):12.4f} {'both':>12} {'sqrt(v_y²+v_z²)':>18}")


if __name__ == "__main__":
    task1_workspace_analysis()
    task2_velocity_analysis()
