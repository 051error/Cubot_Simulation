#!/usr/bin/env python3
"""CPG+RL training for hexapod walking — residual joint-angle policy.

Pipeline: RL policy (18D residual) + C++ tripod CPG target → IK → MuJoCo.
"""

import os, sys, argparse, glob, json, shutil
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from torch.utils.tensorboard import SummaryWriter

# ─── Config ──────────────────────────────────────────────────────────────
MODEL_XML = os.path.join(os.path.dirname(__file__), "..", "models", "scene.xml")
LOG_DIR      = os.environ.get(
    "CUBOT_RL_LOG_DIR", os.path.join(os.path.dirname(__file__), "..", "rl_logs"))
EVAL_LOG_DIR = os.path.join(LOG_DIR, "eval")
CKPT_DIR     = os.environ.get(
    "CUBOT_RL_CKPT_DIR", os.path.join(os.path.dirname(__file__), "..", "rl_checkpoints"))
BEST_PATH    = os.path.join(CKPT_DIR, "best_model.zip")
LATEST_PATH  = os.path.join(CKPT_DIR, "latest_model.zip")
BASELINE_PATH = os.path.join(os.path.dirname(__file__), "cpg_baseline.json")

N_LEGS  = 6
N_JOINTS = N_LEGS * 3          # 18
CTRL_DT  = 0.02                 # 50Hz RL control
SIM_STEPS_PER_CTRL = 4          # 0.02s / 0.005s -> CPG integrates at 200Hz
ACTION_SCALE = 0.20             # action in [-1,1] -> joint-angle increment (rad)
LEARNING_RATE = 7e-5            # PPO learning rate for fresh and resumed runs

# Foot tip in tibia local frame (mesh center - half_extent in Z)
FOOT_TIP_TIBIA = np.array([0.00162782, 0.16052104, 0.02951023])
EPISODE_STEPS = 200             # 4s per episode (fixed-length dense reward)

# ─── Terrain curriculum: reset() picks one scene per episode ─────────────
# Train and evaluate exclusively on hills. Pass --fixed_terrain to pin every
# episode to the flat scene for debugging / ablations.
TERRAIN_SCENES = ["scene.xml", "hill.xml"]  # relative to models/
TERRAIN_PROBS  = [0.0, 1.0]                 # flat / hill
CMD_TAU = 0.5      # command low-pass time constant (s)
CMD_EPS = 0.005    # command-to-target closeness threshold (m/s)

# ─── Reward config: motion is positive; all stability/action/stall terms are
#     bounded non-negative costs before their group weights are applied.
# Stability costs are normalized to [0, 1] and aggregated before weighting.
# The action term remains deliberately small: residuals are needed to adapt the
# nominal CPG to rough ground, while action-rate still discourages chattering.
REWARD_TERM_KEYS = [
    "motion", "vel", "progress", "gravity", "ang_vel", "height_vz_hf",
    "vertical_accel", "roll_pitch_accel", "lateral_accel", "slip", "action",
    "action_rate", "stability", "anti_stall", "total_unscaled", "total_scaled",
]
EPISODE_METRIC_RMS_KEYS = {
    "vz_hf", "vertical_accel", "roll_pitch_accel", "lateral_accel",
}

REWARD_SIGMA = {
    "vel":        0.05,              # velocity tracking scale (m/s)
}

# Nominal zero-residual CPG baseline tolerances. These are the values inside
# which normal CPG gait motion is not penalized; they should be recalibrated by
# the zero-action baseline script before comparing final training runs.
CPG_BASELINE = {
    "vz_hf_deadzone": 0.015,          # m/s
    "vz_hf_scale": 0.04,              # m/s
    "vertical_accel_deadzone": 0.50,  # m/s^2
    "vertical_accel_scale": 1.50,     # m/s^2
    "roll_pitch_accel_deadzone": 5.0, # rad/s^2
    "roll_pitch_accel_scale": 15.0,   # rad/s^2
    "lateral_accel_deadzone": 0.50,   # m/s^2
    "lateral_accel_scale": 1.50,      # m/s^2
    "slip_deadzone": 0.08,            # m/s (slip08 experiment)
    "slip_scale": 0.12,               # m/s (roughly baseline p95 excess)
    "tilt_deadzone": np.deg2rad(1.5), # rad
    "tilt_scale": np.deg2rad(4.0),    # rad
    "ang_vel_deadzone": 0.05,         # rad/s
    "ang_vel_scale": 0.30,            # rad/s
}

# Terrain-aware velocity reference: slow the reference on rough ground so the
# policy is not pushed to hold flat-ground speed over obstacles it cannot.
TERRAIN_REF_DECAY = 1.0
TERRAIN_REF_SIGMA = 0.03             # roughness scale (m)
TERRAIN_REF_MIN_SCALE = 0.6          # never brake below 60% of CPG reference

# Reward groups: motion remains about twice the stability-group magnitude, while
# anti-stall keeps a stationary solution below a moving but rough solution.
W_MOTION = 1.0
W_STABILITY = 0.5
W_STALL = 0.4
STALL_CMD_DEADZONE = 0.012            # m/s
STALL_MIN_PROGRESS = 0.25             # fraction of CPG baseline speed
STALL_EMA_TAU = 0.35                  # seconds
STALL_GRACE_STEPS = 20                # reset/command-change grace at 50 Hz

STABILITY_BENCHMARK_COMMANDS = {
    "forward": [0.05, 0.0, 0.0],
    "backward": [-0.05, 0.0, 0.0],
    "left": [0.0, 0.05, 0.0],
    "right": [0.0, -0.05, 0.0],
}
STABILITY_BENCHMARK_SEEDS = (0, 1, 2)

STABILITY_WEIGHTS = {
    "gravity": 0.10,
    "ang_vel": 0.10,
    "height": 0.10,
    "vertical_accel": 0.25,
    "roll_pitch_accel": 0.25,
    "lateral_accel": 0.10,
    "slip": 0.10,
}

# Keep the scaled reward in a PPO-friendly range after the new bounded costs.
REWARD_SCALE = 0.8
SIM_DT = 0.005

# Observation layout (must match rl_inference.py exactly):
#   projected_gravity(3) + body_vel(3) + joint_pos(18) + joint_vel(18)
#   + foot_contact(6) + cmd(3) + prev_action(18) + q_target(18)
#   + height_map(72) = 159
OBS_DIM = 3 + 3 + 18 + 18 + 6 + 3 + 18 + 18 + 72  # 159

from cpg_gait import (
    TripodGait, FootTrajectory, compute_joint_targets, JOINT_REF,
    STRIDE_X, STRIDE_Y, SPEED_REF,
)
from height_map import rangefinder_to_height_map, NO_GROUND_HEIGHT


def _load_baseline_profile():
    """Load an optional frozen CPG baseline profile from disk.

    returns  profile dictionary, or an empty dictionary when no profile exists
    """
    try:
        with open(BASELINE_PATH) as baseline_file:
            profile = json.load(baseline_file)
        return profile if isinstance(profile, dict) else {}
    except (OSError, ValueError):
        return {}


BASELINE_PROFILE = _load_baseline_profile()
if BASELINE_PROFILE.get("thresholds"):
    CPG_BASELINE.update({
        key: float(value)
        for key, value in BASELINE_PROFILE["thresholds"].items()
        if key in CPG_BASELINE
    })


# ══════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════

def quat_to_projected_gravity(quat):
    """Convert body quaternion [w,x,y,z] to projected gravity (body-frame)."""
    w, x, y, z = quat
    return np.array([
        -2.0 * (x * z - w * y),
        -2.0 * (y * z + w * x),
        -(1.0 - 2.0 * (x * x + y * y)),
    ], dtype=np.float32)


def smooth_penalty(value, deadzone, scale):
    """Return a bounded, smooth penalty after an absolute-value deadzone.

    value     scalar or array whose magnitude is penalized
    deadzone  zero-cost magnitude threshold in the same units
    scale     post-deadzone magnitude that sets the exponential rise
    """
    excess = np.maximum(np.abs(value) - deadzone, 0.0)
    return 1.0 - np.exp(-np.square(excess / max(scale, 1e-9)))


def impact_penalty(value, deadzone, scale):
    """Return a bounded penalty that retains gradient for severe impacts.

    value     scalar or array whose magnitude is penalized
    deadzone  zero-cost magnitude threshold in the same units
    scale     post-deadzone exponential rise scale
    """
    excess = np.maximum(np.abs(value) - deadzone, 0.0)
    return 1.0 - np.exp(-excess / max(scale, 1e-9))


def rms(values):
    """Return the root-mean-square of an array, or zero for an empty array.

    values  scalar samples collected during the current control step
    """
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


# ══════════════════════════════════════════════════════════════════════════
# Environment
# ══════════════════════════════════════════════════════════════════════════

class HexapodEnv(gym.Env):
    """Hexapod walking with CPG+IK (159D obs, 18D residual-joint action)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, fixed_terrain=False, fixed_command=None,
                 height_map_ablation=False):
        super().__init__()
        self.fixed_terrain = fixed_terrain
        self.fixed_command = None if fixed_command is None else np.asarray(
            fixed_command, dtype=np.float32)
        self.height_map_ablation = height_map_ablation

        self.foot_names = [
            "tibia_rf", "tibia_rm", "tibia_rr",
            "tibia_lf", "tibia_lm", "tibia_lr",
        ]

        # Preload the terrain scenes (all include cubot.xml, so body/foot ids
        # are identical across them) so reset() can switch scenes without
        # re-parsing XML every episode.
        models_dir = os.path.dirname(MODEL_XML)
        self._terrain_models = [
            mujoco.MjModel.from_xml_path(os.path.join(models_dir, s))
            for s in TERRAIN_SCENES]
        self._terrain_datas = [mujoco.MjData(m) for m in self._terrain_models]
        self._terrain_body_ids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "MP_BODY")
            for m in self._terrain_models]
        self._terrain_foot_ids = [
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in self.foot_names]
            for m in self._terrain_models]
        self._terrain_geom_ids = []
        self._terrain_geom_to_leg = []
        for model, foot_ids in zip(self._terrain_models, self._terrain_foot_ids):
            geom_to_leg = {}
            foot_body_to_leg = {bid: i for i, bid in enumerate(foot_ids) if bid >= 0}
            for gid, bid in enumerate(model.geom_bodyid):
                if bid in foot_body_to_leg:
                    geom_to_leg[gid] = foot_body_to_leg[bid]
            self._terrain_geom_ids.append(
                [gid for gid, bid in enumerate(model.geom_bodyid) if bid == 0])
            self._terrain_geom_to_leg.append(geom_to_leg)

        # Active model/data, swapped by reset(); start on the flat scene.
        self._terrain_idx = 0
        self.model = self._terrain_models[0]
        self.data = self._terrain_datas[0]
        self.body_id = self._terrain_body_ids[0]
        self.foot_ids = self._terrain_foot_ids[0]
        self.terrain_geom_ids = set(self._terrain_geom_ids[0])
        self.geom_to_leg = self._terrain_geom_to_leg[0]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_JOINTS,), dtype=np.float32)

        self.render_mode = render_mode
        self.viewer = None

        # CPG layer — C++ tripod CPG + foot trajectory (shared with NORMAL mode)
        self.cpg = TripodGait()
        self.traj = FootTrajectory()
        self._q_target = JOINT_REF.copy()

        # Episode state
        self.step_count = 0
        self.cmd = np.zeros(3)

        # Reward state
        self._prev_action = None
        self._prev_body_lin = None     # body linear velocity at previous step (accel term)
        self._prev_body_ang = None     # body angular velocity at previous step (accel term)
        self._prev_body_z = None       # body z at previous step (height-rate term)
        self._prev_foot_tip = None     # foot-tip world positions at previous step (slip term)
        self._prev_body_vz = None
        self._vz_trend = 0.0
        self._v_parallel_ema = 0.0
        self._stall_grace = STALL_GRACE_STEPS
        self._prev_contacts = np.zeros(N_LEGS, dtype=bool)
        self._vz_initialized = False
        self._prev_cmd_dir = np.zeros(2, dtype=np.float64)
        self._last_metrics = {}
        self._reset_episode_metrics()

        # Stability metric (mean body tilt from vertical, radians)
        self._tilt_sum = 0.0
        self._tilt_count = 0

        # Per-episode weighted-reward-term accumulator (for TensorBoard logging)
        self._reward_terms = {k: 0.0 for k in REWARD_TERM_KEYS}

        # IK failure counter (for debugging)
        self._ik_fails = 0

    def _reset_episode_metrics(self):
        """Reset the control-step metrics accumulated for one episode."""
        self._episode_metric_sums = {}
        self._episode_metric_sq_sums = {}
        self._episode_metric_count = 0
        self._episode_tilt_max = 0.0
        self._episode_stall_run = 0
        self._episode_stall_max_steps = 0

    def _accumulate_episode_metrics(self, metrics, tilt):
        """Accumulate one control-step diagnostics sample for episode reporting.

        metrics  current control-step diagnostics dictionary; tilt  body tilt in radians
        """
        for key, value in metrics.items():
            value = float(value)
            self._episode_metric_sums[key] = self._episode_metric_sums.get(key, 0.0) + value
            if key in EPISODE_METRIC_RMS_KEYS:
                self._episode_metric_sq_sums[key] = (
                    self._episode_metric_sq_sums.get(key, 0.0) + value ** 2)
        self._episode_metric_count += 1
        self._episode_tilt_max = max(self._episode_tilt_max, float(tilt))
        if metrics["stall"] > 0.0:
            self._episode_stall_run += 1
            self._episode_stall_max_steps = max(
                self._episode_stall_max_steps, self._episode_stall_run)
        else:
            self._episode_stall_run = 0

    def _sample_target(self):
        """Sample a random command target (bidirectional [0.02, 0.05] m/s, wz=0)."""
        sign = lambda: 1 if self.np_random.random() < 0.5 else -1
        # Command range aligned with the halved /upper_ctrl limits: linear
        # ±0.05 m/s. No angular command — rotation is delegated to the dedicated
        # TURN mode, so the yaw command is always zero.
        return np.array([
            self.np_random.uniform(0.02, 0.05) * sign(),  # vx
            self.np_random.uniform(0.02, 0.05) * sign(),  # vy
            0.0,                                           # wz
        ])

    def _update_cmd(self):
        """Smoothly approach the random target via a first-order low-pass filter."""
        alpha = 1.0 - np.exp(-CTRL_DT / CMD_TAU)
        self.cmd[:2] += alpha * (self._cmd_target[:2] - self.cmd[:2])
        if np.linalg.norm(self.cmd[:2] - self._cmd_target[:2]) < CMD_EPS:
            self._cmd_target = self._sample_target()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Pick a terrain scene per episode. With fixed_terrain the flat scene
        # (idx 0) is used every episode; otherwise sample by TERRAIN_PROBS.
        if self.fixed_terrain:
            idx = 0
        else:
            idx = self.np_random.choice(len(TERRAIN_SCENES), p=TERRAIN_PROBS)
        self._terrain_idx = int(idx)
        self.model = self._terrain_models[idx]
        self.data = self._terrain_datas[idx]
        self.body_id = self._terrain_body_ids[idx]
        self.foot_ids = self._terrain_foot_ids[idx]
        self.terrain_geom_ids = set(self._terrain_geom_ids[idx])
        self.geom_to_leg = self._terrain_geom_to_leg[idx]

        mujoco.mj_resetData(self.model, self.data)
        # Start from the crouched "home" keyframe (if defined) instead of the
        # fully-extended default pose. Keeps foot tips on the ground.
        if self.model.nkey > 0:
            self.data.qpos[:] = self.model.key_qpos[0]
            mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self._prev_action = None
        self._prev_body_lin = None
        self._prev_body_ang = None
        self._prev_body_z = None
        self._prev_foot_tip = None
        self._prev_body_vz = None
        self._vz_trend = 0.0
        self._v_parallel_ema = 0.0
        self._stall_grace = STALL_GRACE_STEPS
        self._prev_contacts = np.zeros(N_LEGS, dtype=bool)
        self._vz_initialized = False
        self._prev_cmd_dir = np.zeros(2, dtype=np.float64)
        self._last_metrics = {}
        self._reset_episode_metrics()
        self._tilt_sum = 0.0
        self._tilt_count = 0
        self._reward_terms = {k: 0.0 for k in REWARD_TERM_KEYS}
        self.cpg.reset()
        self._q_target = JOINT_REF.copy()

        # Smooth random command: start from rest, approach a random target.
        if self.fixed_command is None:
            self.cmd = np.zeros(3, dtype=np.float32)
            self._cmd_target = self._sample_target()
        else:
            self.cmd = self.fixed_command.copy()
            self._cmd_target = self.fixed_command.copy()
        return self._get_obs(), {}

    def step(self, action):
        # ── CPG target joints + RL residual -> joint angles -> simulation ──
        # The C++ tripod CPG runs at 200Hz; RL supplies 18 joint residuals at 50Hz.
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        self._current_action = action.copy()

        if self.fixed_command is None:
            self._update_cmd()   # smooth the random command toward its target

        substep_samples = []
        for _ in range(SIM_STEPS_PER_CTRL):
            q_target = compute_joint_targets(self.cpg, self.traj,
                                             self.cmd[0], self.cmd[1])
            q = q_target + action * ACTION_SCALE
            self.data.ctrl[:18] = np.clip(q, -2.5, 2.5)
            mujoco.mj_step(self.model, self.data)
            substep_samples.append(self._sample_dynamics())
        self._q_target = q_target

        self.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(substep_samples) / REWARD_SCALE
        self._reward_terms["total_scaled"] = (
            self._reward_terms.get("total_scaled", 0.0) + reward)

        terminated = False
        truncated = self.step_count >= EPISODE_STEPS
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        """Build the 159D observation."""
        d = self.data; m = self.model

        quat = d.xquat[self.body_id].copy() if self.body_id >= 0 else np.array([1., 0., 0., 0.])
        projected_gravity = quat_to_projected_gravity(quat)

        body_ang, body_lin = self._body_velocity(local=True)
        body_vel = np.array([body_lin[0], body_lin[1], body_ang[2]], dtype=np.float32)

        joint_pos = d.qpos[7:25].copy()
        joint_vel = d.qvel[6:24].copy()

        contacts, _, _ = self._contact_state()
        contacts = contacts.astype(np.float32)

        prev_action = self._prev_action if self._prev_action is not None \
                      else np.zeros(N_JOINTS, dtype=np.float32)

        # 360° rangefinder -> robot-centric local height map (72 ground heights
        # = 24 azimuths x [near, mid, far]), appended to the observation.
        ranges = d.sensordata[:72] if m.nsensor >= 72 else np.full(72, -1.0)
        height_map = rangefinder_to_height_map(ranges)
        if self.height_map_ablation:
            height_map.fill(0.0)

        obs = np.concatenate([
            projected_gravity,                                # 3
            body_vel / np.array([0.3, 0.3, 2.0]),             # 3
            joint_pos / 1.0,                                  # 18
            joint_vel / 5.0,                                  # 18
            contacts,                                         # 6
            self.cmd.copy() / np.array([0.05, 0.05, 1.0]),    # 3
            prev_action,                                      # 18
            self._q_target / 1.0,                             # 18
            height_map / 0.5,                                 # 72
        ]).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    def _body_velocity(self, local=True):
        """Return MP_BODY angular and linear velocity from MuJoCo.

        local  true for body-local orientation, false for world orientation
        """
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.body_id,
            velocity, int(local))
        return velocity[:3].copy(), velocity[3:].copy()

    def _point_velocity(self, body_id, point):
        """Return the world velocity of a world-space point on a body.

        body_id  MuJoCo body id; point  world-space contact position
        """
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body_id,
            velocity, 0)
        angular, linear = velocity[:3], velocity[3:]
        return linear + np.cross(angular, point - self.data.xpos[body_id])

    def _contact_state(self):
        """Return force-bearing contacts and force-weighted tangent slip.

        returns  contact flags (6,), mean tangent slip speed, bearing foot count
        """
        normal_force = np.zeros(N_LEGS, dtype=np.float64)
        slip_force = np.zeros(N_LEGS, dtype=np.float64)
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1, geom2 = int(contact.geom[0]), int(contact.geom[1])
            if geom1 in self.geom_to_leg and geom2 in self.terrain_geom_ids:
                leg, foot_geom, terrain_geom = self.geom_to_leg[geom1], geom1, geom2
            elif geom2 in self.geom_to_leg and geom1 in self.terrain_geom_ids:
                leg, foot_geom, terrain_geom = self.geom_to_leg[geom2], geom2, geom1
            else:
                continue
            if contact.efc_address < 0:
                continue

            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_id, force)
            fn = max(float(force[0]), 0.0)
            if fn < 1.0:
                continue

            point = np.asarray(contact.pos, dtype=np.float64)
            foot_body = int(self.model.geom_bodyid[foot_geom])
            terrain_body = int(self.model.geom_bodyid[terrain_geom])
            relative_velocity = (
                self._point_velocity(foot_body, point)
                - self._point_velocity(terrain_body, point))
            normal = np.asarray(contact.frame[:3], dtype=np.float64)
            tangent = relative_velocity - np.dot(relative_velocity, normal) * normal
            slip_speed = float(np.linalg.norm(tangent))
            normal_force[leg] += fn
            slip_force[leg] += fn * slip_speed

        contacts = normal_force > 0.0
        if np.any(contacts):
            per_leg_slip = np.divide(
                slip_force, normal_force, out=np.zeros_like(slip_force),
                where=normal_force > 0.0)
            mean_slip = float(np.average(
                per_leg_slip[contacts], weights=normal_force[contacts]))
        else:
            mean_slip = 0.0
        return contacts, mean_slip, int(np.count_nonzero(contacts))

    def _sample_dynamics(self):
        """Sample body dynamics and contact slip after one MuJoCo substep.

        returns  dictionary consumed by the 50 Hz reward aggregation
        """
        body_ang, body_lin = self._body_velocity(local=True)
        _, world_lin = self._body_velocity(local=False)
        contacts, slip_speed, contact_count = self._contact_state()

        if self._prev_body_lin is None:
            body_accel = np.zeros(3, dtype=np.float64)
            angular_accel = np.zeros(3, dtype=np.float64)
            vertical_accel = 0.0
        else:
            body_accel = (body_lin - self._prev_body_lin) / SIM_DT
            angular_accel = (body_ang - self._prev_body_ang) / SIM_DT
            vertical_accel = (world_lin[2] - self._prev_body_vz) / SIM_DT
        self._prev_body_lin = body_lin.copy()
        self._prev_body_ang = body_ang.copy()
        self._prev_body_vz = float(world_lin[2])

        trend_alpha = 1.0 - np.exp(-SIM_DT / 0.35)
        if not self._vz_initialized:
            self._vz_trend = float(world_lin[2])
            self._vz_initialized = True
        else:
            self._vz_trend += trend_alpha * (world_lin[2] - self._vz_trend)
        return {
            "body_ang": body_ang,
            "body_lin": body_lin,
            "body_accel": body_accel,
            "angular_accel": angular_accel,
            "world_vz": float(world_lin[2]),
            "vz_hf": float(world_lin[2] - self._vz_trend),
            "vertical_accel": float(vertical_accel),
            "slip_speed": slip_speed,
            "contacts": contacts,
            "contact_count": contact_count,
        }

    # ─── Reward (bounded motion and stability groups) ─────────────────────

    def _compute_reward(self, samples):
        """Compute bounded motion, stability, anti-stall, and action terms.

        samples  200 Hz dynamics samples collected within this control step
        """
        d = self.data; m = self.model
        quat = d.xquat[self.body_id] if self.body_id >= 0 else np.array([1., 0., 0., 0.])
        action = self._current_action
        body_lin = np.mean([s["body_lin"] for s in samples], axis=0)
        body_ang = np.mean([s["body_ang"] for s in samples], axis=0)

        cmd_xy = np.asarray(self.cmd[:2], dtype=np.float64)
        cmd_speed = float(np.linalg.norm(cmd_xy))
        if cmd_speed > 1e-6:
            cmd_dir = cmd_xy / cmd_speed
            lateral_dir = np.array([-cmd_dir[1], cmd_dir[0]])
            omega = 2.0 * np.pi * cmd_speed / 0.15 + 4.0
            gain = min(cmd_speed / SPEED_REF, 1.0)
            cpg_ref = 4.0 * gain * omega / (2.0 * np.pi)
            stride_ref = np.array([STRIDE_X * cmd_dir[0], STRIDE_Y * cmd_dir[1]])
            cpg_ref_velocity = cpg_ref * stride_ref
        else:
            cmd_dir = np.zeros(2)
            lateral_dir = np.zeros(2)
            cpg_ref_velocity = np.zeros(2)

        ranges = d.sensordata[:72] if m.nsensor >= 72 else np.full(72, -1.0)
        height_map = rangefinder_to_height_map(ranges)
        valid_height = height_map[height_map > NO_GROUND_HEIGHT + 0.5]
        roughness = float(np.std(valid_height)) if valid_height.size > 4 else 0.0
        ref_scale = 1.0 / (
            1.0 + TERRAIN_REF_DECAY * (roughness / TERRAIN_REF_SIGMA) ** 2)
        ref_scale = float(np.clip(ref_scale, TERRAIN_REF_MIN_SCALE, 1.0))
        cpg_ref_velocity *= ref_scale
        ref_speed = max(float(np.linalg.norm(cpg_ref_velocity)), 1e-6)

        velocity_error = body_lin[:2] - cpg_ref_velocity
        r_vel = float(np.exp(-np.sum(velocity_error ** 2) / REWARD_SIGMA["vel"] ** 2))
        v_parallel = float(np.dot(body_lin[:2], cmd_dir)) if cmd_speed > 1e-6 else 0.0
        progress_ratio = max(v_parallel, 0.0) / ref_speed if cmd_speed > 1e-6 else 0.0
        r_progress = float(1.0 - np.exp(-progress_ratio ** 2))
        r_motion = 0.7 * r_vel + 0.3 * r_progress

        gravity = quat_to_projected_gravity(quat)
        tilt = float(np.arccos(np.clip(-gravity[2], -1.0, 1.0)))
        c_gravity = float(smooth_penalty(
            tilt, CPG_BASELINE["tilt_deadzone"], CPG_BASELINE["tilt_scale"]))
        roll_pitch_speed = float(np.linalg.norm(body_ang[:2]))
        c_ang_vel = float(smooth_penalty(
            roll_pitch_speed, CPG_BASELINE["ang_vel_deadzone"],
            CPG_BASELINE["ang_vel_scale"]))

        vz_hf = rms([s["vz_hf"] for s in samples])
        vertical_accel = rms([s["vertical_accel"] for s in samples])
        roll_pitch_accel = rms([
            np.linalg.norm(s["angular_accel"][:2]) for s in samples])
        if cmd_speed > STALL_CMD_DEADZONE:
            lateral_accel = rms([
                np.dot(s["body_accel"][:2], lateral_dir) for s in samples])
        else:
            lateral_accel = 0.0
        slip_speed = float(np.mean([s["slip_speed"] for s in samples]))

        c_height = float(smooth_penalty(
            vz_hf, CPG_BASELINE["vz_hf_deadzone"], CPG_BASELINE["vz_hf_scale"]))
        c_vertical_accel = float(impact_penalty(
            vertical_accel, CPG_BASELINE["vertical_accel_deadzone"],
            CPG_BASELINE["vertical_accel_scale"]))
        c_roll_pitch_accel = float(impact_penalty(
            roll_pitch_accel, CPG_BASELINE["roll_pitch_accel_deadzone"],
            CPG_BASELINE["roll_pitch_accel_scale"]))
        c_lateral_accel = float(smooth_penalty(
            lateral_accel, CPG_BASELINE["lateral_accel_deadzone"],
            CPG_BASELINE["lateral_accel_scale"]))
        c_slip = float(smooth_penalty(
            slip_speed, CPG_BASELINE["slip_deadzone"], CPG_BASELINE["slip_scale"]))

        c_action = float(np.mean(np.square(action)))
        if self._prev_action is not None:
            c_action_rate = float(np.mean(np.square(action - self._prev_action)) / 4.0)
        else:
            c_action_rate = 0.0
        self._prev_action = action.copy()

        ema_alpha = 1.0 - np.exp(-CTRL_DT / STALL_EMA_TAU)
        direction_changed = (
            cmd_speed > STALL_CMD_DEADZONE
            and np.linalg.norm(self._prev_cmd_dir) > 0.0
            and np.dot(cmd_dir, self._prev_cmd_dir) < 0.5)
        if direction_changed:
            self._v_parallel_ema = 0.0
            self._stall_grace = STALL_GRACE_STEPS
        if cmd_speed > STALL_CMD_DEADZONE:
            self._prev_cmd_dir = cmd_dir.copy()
        self._v_parallel_ema += ema_alpha * (v_parallel - self._v_parallel_ema)
        if self._stall_grace > 0:
            self._stall_grace -= 1
            c_stall = 0.0
        elif cmd_speed > STALL_CMD_DEADZONE:
            stall_target = STALL_MIN_PROGRESS * ref_speed
            deficit = max(stall_target - self._v_parallel_ema, 0.0)
            c_stall = float(1.0 - np.exp(-(deficit / max(stall_target, 1e-6)) ** 2))
        else:
            c_stall = 0.0

        stability_costs = {
            "gravity": c_gravity,
            "ang_vel": c_ang_vel,
            "height": c_height,
            "vertical_accel": c_vertical_accel,
            "roll_pitch_accel": c_roll_pitch_accel,
            "lateral_accel": c_lateral_accel,
            "slip": c_slip,
        }
        c_stability = float(sum(
            STABILITY_WEIGHTS[key] * value
            for key, value in stability_costs.items()))
        action_cost = 0.075 * c_action + 0.075 * c_action_rate
        total = (W_MOTION * r_motion - W_STABILITY * c_stability
                 - W_STALL * c_stall - action_cost)

        terms = {
            "motion": W_MOTION * r_motion,
            "vel": r_vel,
            "progress": r_progress,
            "gravity": -c_gravity,
            "ang_vel": -c_ang_vel,
            "height_vz_hf": -c_height,
            "vertical_accel": -c_vertical_accel,
            "roll_pitch_accel": -c_roll_pitch_accel,
            "lateral_accel": -c_lateral_accel,
            "slip": -c_slip,
            "action": -0.075 * c_action,
            "action_rate": -0.075 * c_action_rate,
            "stability": -W_STABILITY * c_stability,
            "anti_stall": -W_STALL * c_stall,
            "total_unscaled": total,
        }
        for key, value in terms.items():
            self._reward_terms[key] += float(value)

        self._last_metrics = {
            "roughness": roughness, "ref_scale": ref_scale,
            "ref_speed": ref_speed, "v_parallel": v_parallel,
            "progress_ratio": progress_ratio,
            "vz_hf": vz_hf, "vertical_accel": vertical_accel,
            "roll_pitch_accel": roll_pitch_accel,
            "lateral_accel": lateral_accel, "slip_speed": slip_speed,
            "contact_count": float(np.mean([s["contact_count"] for s in samples])),
            "stall": float(c_stall > 0.0),
        }
        self._accumulate_episode_metrics(self._last_metrics, tilt)
        self._tilt_sum += tilt
        self._tilt_count += 1
        return total

    def _is_terminated(self):
        # Unused in fixed-length mode (step() always returns terminated=False);
        # kept for debugging / fallback to early-termination training.
        d = self.data; m = self.model
        quat = d.xquat[self.body_id] if self.body_id >= 0 else np.array([1., 0., 0., 0.])
        w, x, y, zz = quat
        upright = 1.0 - 2.0 * (x * x + y * y)
        body_z = d.xpos[self.body_id][2] if self.body_id >= 0 else 1.0
        return upright < 0.3 or body_z < 0.05

    @property
    def mean_tilt(self):
        # Mean body tilt angle from vertical (radians) over the episode.
        if self._tilt_count == 0:
            return 0.0
        return self._tilt_sum / self._tilt_count

    @property
    def reward_terms(self):
        # Per-episode sum of each weighted reward term (for TensorBoard).
        return dict(self._reward_terms)

    @property
    def episode_metrics(self):
        """Return whole-episode dynamics aligned with the episode reward."""
        count = self._episode_metric_count
        if count == 0:
            return {}
        metrics = {}
        for key, total in self._episode_metric_sums.items():
            if key in EPISODE_METRIC_RMS_KEYS:
                metrics[f"{key}_rms"] = float(np.sqrt(
                    self._episode_metric_sq_sums[key] / count))
            elif key != "stall":
                metrics[f"{key}_mean"] = float(total / count)
        metrics["tilt_mean_deg"] = float(np.degrees(self.mean_tilt))
        metrics["tilt_max_deg"] = float(np.degrees(self._episode_tilt_max))
        metrics["stall_fraction"] = float(
            self._episode_metric_sums.get("stall", 0.0) / count)
        metrics["stall_max_steps"] = float(self._episode_stall_max_steps)
        return metrics

    @property
    def reward_metrics(self):
        """Return the latest unscaled dynamics metrics for diagnostics.

        returns  copy of the latest per-control-step metric dictionary
        """
        return dict(self._last_metrics)

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# ══════════════════════════════════════════════════════════════════════════
# Callbacks
# ══════════════════════════════════════════════════════════════════════════

class EvalAndSaveCallback(BaseCallback):
    def __init__(self, eval_env, stability_envs, save_freq, n_eval_episodes=5, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.stability_envs = stability_envs
        self.save_freq = save_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.writer = None

    def _on_training_start(self):
        # Fixed directory (no timestamp) so a resume appends to the same
        # tfevents stream. The x-axis below uses the global num_timesteps, so
        # eval curves continue from the last step instead of restarting at 0.
        self.writer = SummaryWriter(log_dir=EVAL_LOG_DIR)

    def _run_stability_benchmark(self):
        """Run fixed-command, fixed-seed hill episodes for each cardinal direction."""
        results = {}
        for direction, env in self.stability_envs.items():
            rewards = []
            metrics = {}
            for seed in STABILITY_BENCHMARK_SEEDS:
                obs, _ = env.reset(seed=seed)
                terminated, truncated = False, False
                episode_reward = 0.0
                while not (terminated or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                rewards.append(episode_reward)
                for key, value in env.unwrapped.episode_metrics.items():
                    metrics.setdefault(key, []).append(value)
            results[direction] = {"reward": rewards, **metrics}
        return results

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            step = self.model.num_timesteps   # global counter, survives resume
            rewards, tilts = [], []
            term_means = {k: [] for k in REWARD_TERM_KEYS}
            episode_metrics = {}
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                ep_reward = 0.0
                terminated, truncated = False, False
                while not (terminated or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, r, terminated, truncated, _ = self.eval_env.step(action)
                    ep_reward += r
                rewards.append(ep_reward)
                tilts.append(self.eval_env.unwrapped.mean_tilt)
                for k, v in self.eval_env.unwrapped.reward_terms.items():
                    term_means[k].append(v)
                for k, v in self.eval_env.unwrapped.episode_metrics.items():
                    episode_metrics.setdefault(k, []).append(v)

            mean_r = np.mean(rewards)
            mean_tilt_deg = np.degrees(np.mean(tilts))

            benchmark_results = self._run_stability_benchmark()
            tilt_summary = ", ".join(
                f"{direction[0].upper()}={np.mean(values['tilt_max_deg']):.1f}"
                for direction, values in benchmark_results.items())
            print(f"[{step:>8d} steps]  mean_reward={mean_r:+.3f}  "
                  f"tilt={mean_tilt_deg:.2f} deg  best={self.best_mean_reward:+.3f}  "
                  f"benchmark_tilt_max({tilt_summary})")

            os.makedirs(CKPT_DIR, exist_ok=True)
            self.model.save(LATEST_PATH)

            if mean_r > self.best_mean_reward:
                self.best_mean_reward = mean_r
                self.model.save(BEST_PATH)
                print(f"  >>> New BEST model saved: reward={mean_r:+.3f}")

            if self.writer:
                self.writer.add_scalar("eval/mean_reward", mean_r, step)
                self.writer.add_scalar("eval/mean_tilt", mean_tilt_deg, step)
                self.writer.add_scalar("eval/best_reward", self.best_mean_reward, step)
                for k, vals in term_means.items():
                    self.writer.add_scalar(f"eval/term_{k}", np.mean(vals), step)
                for k, vals in episode_metrics.items():
                    if vals:
                        self.writer.add_scalar(
                            f"eval/episode_{k}", np.mean(vals), step)
                    self.writer.add_scalar(f"eval/episode_{k}_std", np.std(vals), step)
                for direction, metrics in benchmark_results.items():
                    for metric, values in metrics.items():
                        self.writer.add_scalar(f"eval/stability/{direction}/{metric}", np.mean(values), step)
                        self.writer.add_scalar(f"eval/stability/{direction}/{metric}_std", np.std(values), step)

        return True

    def _on_training_end(self):
        for env in (self.eval_env, *self.stability_envs.values()):
            env.close()
        if self.writer:
            self.writer.close()


# ══════════════════════════════════════════════════════════════════════════
# Checkpoint & main
# ══════════════════════════════════════════════════════════════════════════

def find_latest_checkpoint():
    os.makedirs(CKPT_DIR, exist_ok=True)
    if os.path.exists(LATEST_PATH):
        return LATEST_PATH
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "model_*_steps.zip")))
    return ckpts[-1] if ckpts else None


def _make_env(fixed_terrain=False, fixed_command=None, height_map_ablation=False):
    return Monitor(HexapodEnv(
        fixed_terrain=fixed_terrain, fixed_command=fixed_command,
        height_map_ablation=height_map_ablation))


def run_height_map_ablation(model_path):
    """Compare a trained policy with its terrain-map observation set to zero.

    model_path  PPO checkpoint evaluated on fixed cardinal hill commands
    """
    model = PPO.load(model_path)
    results = {"model_path": os.path.abspath(model_path),
               "seeds": list(STABILITY_BENCHMARK_SEEDS), "conditions": {}}
    for condition, ablate in (("terrain_map", False), ("zero_map", True)):
        condition_results = {}
        for direction, command in STABILITY_BENCHMARK_COMMANDS.items():
            env = _make_env(fixed_command=command, height_map_ablation=ablate)
            rewards = []
            metrics = {}
            for seed in STABILITY_BENCHMARK_SEEDS:
                obs, _ = env.reset(seed=seed)
                terminated, truncated = False, False
                episode_reward = 0.0
                while not (terminated or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                rewards.append(float(episode_reward))
                for key, value in env.unwrapped.episode_metrics.items():
                    metrics.setdefault(key, []).append(float(value))
            condition_results[direction] = {"reward": rewards, **metrics}
            env.close()
        results["conditions"][condition] = condition_results
    return results


def run_cpg_baseline(seeds=3):
    """Measure zero-residual CPG dynamics on hill terrain and save a profile.

    seeds  number of deterministic reset seeds per cardinal command
    """
    commands = {
        "forward": [0.05, 0.0, 0.0],
        "backward": [-0.05, 0.0, 0.0],
        "left": [0.0, 0.05, 0.0],
        "right": [0.0, -0.05, 0.0],
    }
    zero_action = np.zeros(N_JOINTS, dtype=np.float32)
    profile = {"version": 1, "terrain": "hill.xml", "seeds": seeds,
               "commands": {}}
    for label, command in commands.items():
        returns = []
        progress_speeds = []
        metrics = {key: [] for key in (
            "height_vz_hf", "vertical_accel", "roll_pitch_accel",
            "lateral_accel", "slip", "progress")}
        for seed in range(seeds):
            env = HexapodEnv(fixed_command=command)
            env.reset(seed=seed)
            episode_return = 0.0
            episode_progress = []
            for _ in range(EPISODE_STEPS):
                _, reward, _, _, _ = env.step(zero_action)
                episode_return += reward
                episode_progress.append(env.reward_metrics.get("v_parallel", 0.0))
            returns.append(episode_return)
            progress_speeds.append(float(np.mean(episode_progress[20:])))
            terms = env.reward_terms
            for key in metrics:
                metrics[key].append(terms[key] / EPISODE_STEPS)
            env.close()
        profile["commands"][label] = {}
        profile["commands"][label]["command"] = command
        profile["commands"][label]["return_mean"] = float(np.mean(returns))
        profile["commands"][label]["return_std"] = float(np.std(returns))
        profile["commands"][label]["progress_speed"] = float(np.mean(progress_speeds))
        profile["commands"][label]["terms"] = {}
        for key, values in metrics.items():
            profile["commands"][label]["terms"][key] = {
                "mean": float(np.mean(values)),
                "p95": float(np.percentile(values, 95)),
            }
        print(f"[baseline {label:8s}] return={np.mean(returns):+.3f} "
              f"+/-{np.std(returns):.3f}")
        for key, values in metrics.items():
            print(f"  {key:20s} mean={np.mean(values):+.5f} "
                  f"p95={np.percentile(values, 95):+.5f}")
    with open(BASELINE_PATH, "w") as baseline_file:
        json.dump(profile, baseline_file, indent=2)
    print(f"Baseline profile: {BASELINE_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=5_000_000)
    parser.add_argument("--batch_size",  type=int, default=256)
    parser.add_argument("--n_envs",      type=int, default=4)
    parser.add_argument("--save_freq",   type=int, default=5_000)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--fixed_terrain", action="store_true",
                        help="Pin every episode to the flat scene (disable terrain randomization)")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--height_map_ablation", action="store_true")
    parser.add_argument("--model", type=str, default=BEST_PATH)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.baseline:
        run_cpg_baseline()
        return
    if args.height_map_ablation:
        results = run_height_map_ablation(args.model)
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(args.model)), "height_map_ablation.json")
        with open(output_path, "w") as output_file:
            json.dump(results, output_file, indent=2)
        print(f"Height-map ablation: {output_path}")
        for direction in STABILITY_BENCHMARK_COMMANDS:
            normal = results["conditions"]["terrain_map"][direction]
            ablated = results["conditions"]["zero_map"][direction]
            print(f"[{direction:8s}] reward {np.mean(normal['reward']):+.2f} -> "
                  f"{np.mean(ablated['reward']):+.2f}; vertical "
                  f"{np.mean(normal['vertical_accel_rms']):.3f} -> "
                  f"{np.mean(ablated['vertical_accel_rms']):.3f}; tilt_max "
                  f"{np.mean(normal['tilt_max_deg']):.2f} -> "
                  f"{np.mean(ablated['tilt_max_deg']):.2f}")
        return

    os.makedirs(LOG_DIR, exist_ok=True)

    n_envs = args.n_envs
    print(f"CPG+RL (residual policy): {n_envs} parallel envs on {os.cpu_count()} CPUs")
    print(f"  RL → 18 joint-angle increments (±{ACTION_SCALE} rad) → CPG target + residual")
    print(f"  terrain: {TERRAIN_SCENES} (probs {TERRAIN_PROBS})"
          f"{' (fixed flat)' if args.fixed_terrain else ''}")
    env = DummyVecEnv([lambda: _make_env(args.fixed_terrain) for _ in range(n_envs)])
    # NOTE: SubprocVecEnv hangs with EGL backend; use DummyVecEnv instead
    eval_env = _make_env(args.fixed_terrain)

    # log_std_init=-2.0 (std~0.135): the residual action is small (ACTION_SCALE
    # 0.2 rad), so a large initial std (0.37 at -1.0) makes rollouts unstable and
    # forces ~1M steps of std shrinkage before sampled rollouts turn positive.
    stability_envs = {
        direction: _make_env(fixed_command=command)
        for direction, command in STABILITY_BENCHMARK_COMMANDS.items()
    }

    policy_kwargs = dict(net_arch=dict(pi=[128, 64], vf=[128, 64]),
                         log_std_init=-2.0)

    ckpt_path = None if args.fresh else find_latest_checkpoint() if args.resume else None
    if ckpt_path is None and os.path.isdir(EVAL_LOG_DIR):
        # Fresh run: drop stale eval events so the curve starts clean at step 0
        # instead of mixing with a previous run's points.
        shutil.rmtree(EVAL_LOG_DIR)
    if ckpt_path:
        print(f"[Resume] Loading checkpoint: {ckpt_path}")
        model = PPO.load(ckpt_path, env=env, tensorboard_log=LOG_DIR)
        # PPO.load restores the old optimizer schedule; explicitly apply the
        # current learning rate so this configuration also takes effect on resume.
        model.learning_rate = LEARNING_RATE
        model.lr_schedule = lambda _: LEARNING_RATE
        model.batch_size = args.batch_size
        for param_group in model.policy.optimizer.param_groups:
            param_group["lr"] = LEARNING_RATE
    else:
        model = PPO("MlpPolicy", env, verbose=0,
                    n_steps=4096, batch_size=args.batch_size,
                    learning_rate=LEARNING_RATE, ent_coef=0.0,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=LOG_DIR)

    eval_cb = EvalAndSaveCallback(
        eval_env, stability_envs, save_freq=args.save_freq, n_eval_episodes=10)

    # learn() treats total_timesteps as "steps to run THIS call", not a global
    # target. On resume, subtract what's already done so we stop at the requested
    # total instead of overshooting by a full total_steps on each resume.
    remaining = args.total_steps - model.num_timesteps
    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=eval_cb,
                    reset_num_timesteps=(ckpt_path is None), progress_bar=True)
    else:
        print(f"[Done] num_timesteps {model.num_timesteps} already >= total_steps "
              f"{args.total_steps}; skipping training.")

    final = os.path.join(CKPT_DIR, "final_model.zip")
    model.save(final)
    print(f"Final model: {final}")
    print(f"Best model: {BEST_PATH}" if os.path.exists(BEST_PATH) else "No best model saved")


if __name__ == "__main__":
    main()
