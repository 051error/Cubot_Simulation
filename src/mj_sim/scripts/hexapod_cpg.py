"""Hopf-oscillator CPG for hexapod foot trajectory generation — Coxa-driven gait.

Implements the CPG architecture from arXiv:2310.07744 ("Terrain-adaptive CPGs
with RL for Hexapod Locomotion"), Section III-A, modified to use coxa (hip)
rotation as the primary forward propulsion mechanism.

Each leg has a Hopf oscillator (x_i, y_i) converging to a unit limit cycle.
Inter-leg diffusive coupling enforces the tripod gait pattern.

Gait cycle (counterclockwise Hopf oscillator, stance = y < 0, swing = y > 0):
  STANCE (y<0, x: -1→+1):
    - Coxa sweeps unidirectionally: foot_z goes 0 → +coxa_amp
    - Foot planted on ground: foot_x fixed at -stance_h, foot_y at neutral + drift
    - Body is pushed forward by coxa rotation through ground friction
  SWING (y>0, x: +1→-1):
    - Coxa resets: foot_z returns amp → 0
    - Foot lifts (foot_x = -stance_h + lift) and reaches forward (foot_y extends)
    - Ready for next stance

Foot positions are in c1_rest frame: X=vertical(up), Y=forward, Z=lateral.
IK solver converts foot positions → 3 joint angles per leg.

Usage:
    from hexapod_cpg import HexapodCPG
    cpg = HexapodCPG(n_legs=6, dt=0.02)
    foot_positions = cpg.step(action)  # action from RL policy
"""

import numpy as np


class HexapodCPG:
    """Hopf-oscillator CPG — coxa-driven gait with tripod coordination.

    Each leg has a Hopf oscillator (x_i, y_i).  During stance (y<0) the
    coxa joint sweeps unidirectionally through foot_z, pushing the body
    forward against the planted foot.  During swing (y>0) the foot lifts,
    reaches forward, and the coxa resets to neutral.

    Tripod groups: {0,2,4} vs {1,3,5} — anti-phase coordination ensures
    3 legs in stance at any time for static stability.
    """

    # Tripod groups: {0,2,4} vs {1,3,5}
    TRIPOD_GROUP = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
    ALPHA = 50.0             # Hopf convergence rate
    FOOT_Y_NEUTRAL = -0.02   # Neutral foot Y in c1_rest (slightly behind hip)
    COXA_Y_COUPLING = 0.7    # d(foot_y)/d(foot_z) for pure coxa rotation (from FK)

    def __init__(self, n_legs=6, dt=0.02):
        self.n_legs = n_legs
        self.dt = dt
        self.x = np.zeros(n_legs)
        self.y = np.zeros(n_legs)
        self.reset()

    def reset(self):
        for i in range(self.n_legs):
            if self.TRIPOD_GROUP[i] == 0:
                self.x[i] = 1.0
                self.y[i] = 0.0
            else:
                self.x[i] = -1.0
                self.y[i] = 0.0

    def get_osc_state(self):
        out = np.empty(self.n_legs * 2, dtype=np.float32)
        for i in range(self.n_legs):
            out[i * 2] = self.x[i]
            out[i * 2 + 1] = self.y[i]
        return out

    def step(self, action):
        """Advance CPG and return foot positions in c1_rest frame.

        Args:
            action: CPG foot-trajectory params in [-1, 1]^8:
              [0] coxa_amp:   [0.005, 0.025] m  — coxa rotation amplitude (foot_z sweep)
              [1] lift_h:     [0.020, 0.060] m  — foot clearance during swing
              [2] stance_h:   [0.140, 0.170] m  — body height from coxa
              [3] freq:       [1.0,  2.0  ] Hz — walking frequency
              [4] turn_bias:  [-0.010, 0.010] m — turn asymmetry
              [5] step_reach: [0.000, 0.060] m  — forward reach during swing
              [6] coupling:   [2.0,  4.0  ]    — inter-leg coupling strength
              [7] swing_asym: [-0.020, 0.020] m — swing height asymmetry

        Returns:
            foot_positions: (18,) float32 — [z0,y0,x0, z1,y1,x1, ...] in c1_rest
            where z=lateral(coxa-driven), y=forward, x=vertical
        """
        n_joints = self.n_legs * 3

        # Decode action
        coxa_amp   = 0.005 + 0.020 * action[0]
        lift_h     = 0.04  + 0.020 * action[1]
        stance_h   = 0.155 + 0.015 * action[2]
        freq       = 1.5   + 0.5   * action[3]
        turn_bias  = 0.010 * action[4]
        step_reach = 0.03  + 0.030 * action[5]
        coupling   = 3.0   + 1.0   * action[6]
        swing_asym = 0.020 * action[7]

        omega = 2.0 * np.pi * freq

        # Coupling forces
        cx = np.zeros(self.n_legs)
        cy = np.zeros(self.n_legs)
        for i in range(self.n_legs):
            for j in range(self.n_legs):
                if i == j:
                    continue
                if self.TRIPOD_GROUP[i] == self.TRIPOD_GROUP[j]:
                    cx[i] += coupling * (self.x[j] - self.x[i])
                    cy[i] += coupling * (self.y[j] - self.y[i])
                else:
                    cx[i] += coupling * (-self.x[j] - self.x[i])
                    cy[i] += coupling * (-self.y[j] - self.y[i])
            cx[i] /= 5.0
            cy[i] /= 5.0

        # Hopf oscillator step
        for i in range(self.n_legs):
            r2 = self.x[i]**2 + self.y[i]**2
            dx = self.ALPHA * (1.0 - r2) * self.x[i] - omega * self.y[i] + cx[i]
            dy = self.ALPHA * (1.0 - r2) * self.y[i] + omega * self.x[i] + cy[i]
            self.x[i] += dx * self.dt
            self.y[i] += dy * self.dt

        # Foot positions
        foot_positions = np.empty(n_joints, dtype=np.float32)

        for i in range(self.n_legs):
            j = i * 3
            # side_sign based on actual left/right, NOT tripod group.
            # XML body order: rf=0, rm=1, rr=2 (RIGHT), lf=3, lm=4, lr=5 (LEFT)
            # RIGHT=-1: right coxa sweeps foot in -Z → body pushed FORWARD
            # LEFT=+1:  left coxa sweeps foot in +Z → body pushed FORWARD
            side_sign = -1.0 if i < 3 else 1.0
            eff_amp = max(0.0, coxa_amp + turn_bias * side_sign)

            if self.y[i] > 0.0:
                # SWING: coxa resets, foot lifts and reaches forward
                coxa_z = eff_amp * max(0.0, self.x[i])
                foot_z = coxa_z * side_sign
                lift = self.y[i]
                foot_x = -stance_h + lift_h * lift + swing_asym * lift * side_sign
                stance_drift = self.COXA_Y_COUPLING * eff_amp
                foot_y = (self.FOOT_Y_NEUTRAL
                          + step_reach * lift
                          + stance_drift * max(0.0, self.x[i]))
            else:
                # STANCE: coxa sweeps unidirectionally, foot on ground
                push = (1.0 + self.x[i]) / 2.0
                foot_z = eff_amp * push * side_sign
                foot_x = -stance_h
                foot_y = (self.FOOT_Y_NEUTRAL
                          + self.COXA_Y_COUPLING * eff_amp * push)

            # c1_rest frame: Z=lateral, Y=forward, X=vertical
            foot_positions[j + 0] = foot_z
            foot_positions[j + 1] = foot_y
            foot_positions[j + 2] = foot_x

        return foot_positions
