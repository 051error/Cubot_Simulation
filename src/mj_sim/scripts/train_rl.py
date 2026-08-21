#!/usr/bin/env python3
"""CPG+RL training for hexapod walking — residual joint-angle policy.

Pipeline: RL policy (18D residual) + C++ tripod CPG target → IK → MuJoCo.
"""

import os, sys, time, argparse, glob
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
LOG_DIR    = os.path.join(os.path.dirname(__file__), "..", "rl_logs")
CKPT_DIR   = os.path.join(os.path.dirname(__file__), "..", "rl_checkpoints")
BEST_PATH  = os.path.join(CKPT_DIR, "best_model.zip")
LATEST_PATH = os.path.join(CKPT_DIR, "latest_model.zip")

N_LEGS  = 6
N_JOINTS = N_LEGS * 3          # 18
CTRL_DT  = 0.02                 # 50Hz RL control
SIM_STEPS_PER_CTRL = 4          # 0.02s / 0.005s -> CPG integrates at 200Hz
ACTION_SCALE = 0.2              # action in [-1,1] -> joint-angle increment (rad)

# Foot tip in tibia local frame (mesh center - half_extent in Z)
FOOT_TIP_TIBIA = np.array([0.00162782, 0.16052104, 0.02951023])
EPISODE_STEPS = 200             # 4s per episode (fixed-length sparse reward)

# ─── Terrain curriculum: reset() picks one scene per episode ─────────────
TERRAIN_SCENES = ["scene.xml", "hill.xml", "stairs.xml"]  # relative to models/
TERRAIN_PROBS  = [0.3, 0.4, 0.3]          # flat / hill / stairs
CMD_TAU = 0.5      # command low-pass time constant (s)
CMD_EPS = 0.005    # command-to-target closeness threshold (m/s)

# ─── Reward config: every term is normalized to [0,1] (positive) or [-1,0]
#     (penalty) before weighting, so the weights are directly comparable.
REWARD_WEIGHTS = {
    "vel":         1.0,   # 1. velocity tracking
    "gravity":     2.0,   # 2. IMU gravity deviation (main stability)
    "ang_vel":     1.5,   # 3. angular-velocity stability
    "height":      0.5,   # 4. body height stability (bobbing)
    "accel":       0.5,   # 5. body acceleration penalty (linear + angular)
    "slip":        0.5,   # 6. stance-foot slip penalty
    "action":      0.05,  # 7. residual magnitude penalty
    "action_rate": 0.02,  # 8. action-change penalty
}
REWARD_SIGMA = {
    "vel":        0.05,              # velocity error scale (m/s)
    "gravity":    np.sqrt(0.2),      # gravity-deviation scale (dimensionless)
    "ang_vel":    0.5,               # angular-velocity scale (rad/s)
    "height_vel": 0.05,              # body-height rate scale (m/s)
    "accel_lin":  2.0,               # linear-acceleration scale (m/s^2)
    "accel_ang":  2.0,               # angular-acceleration scale (rad/s^2)
    "slip":       0.1,               # foot slip-speed scale (m/s)
}

# Observation layout (must match rl_inference.py exactly):
#   projected_gravity(3) + body_vel(3) + joint_pos(18) + joint_vel(18)
#   + foot_contact(6) + cmd(3) + prev_action(18) + q_target(18)
#   + height_map(72) = 159
OBS_DIM = 3 + 3 + 18 + 18 + 6 + 3 + 18 + 18 + 72  # 159

from cpg_gait import TripodGait, FootTrajectory, compute_joint_targets, JOINT_REF
from height_map import rangefinder_to_height_map


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


def quat_to_rotation_matrix(quat):
    """Body→world rotation matrix from quaternion."""
    w, x, y, z = quat
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


# ══════════════════════════════════════════════════════════════════════════
# Environment
# ══════════════════════════════════════════════════════════════════════════

class HexapodEnv(gym.Env):
    """Hexapod walking with CPG+IK (159D obs, 18D residual-joint action)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()

        self.foot_names = [
            "tibia_rf", "tibia_rm", "tibia_rr",
            "tibia_lf", "tibia_lm", "tibia_lr",
        ]

        # Preload the three terrain scenes (all include cubot.xml, so body/foot
        # ids are identical across them) so reset() can switch scenes without
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

        # Active model/data, swapped by reset(); start on the flat scene.
        self._terrain_idx = 0
        self.model = self._terrain_models[0]
        self.data = self._terrain_datas[0]
        self.body_id = self._terrain_body_ids[0]
        self.foot_ids = self._terrain_foot_ids[0]

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
        self._ep_reward_sum = 0.0   # accumulated reward for fixed-length episode

        # Stability metric (mean body tilt from vertical, radians)
        self._tilt_sum = 0.0
        self._tilt_count = 0

        # IK failure counter (for debugging)
        self._ik_fails = 0

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

        # Pick a terrain scene per episode (flat 0.3 / hill 0.4 / stairs 0.3).
        idx = self.np_random.choice(len(TERRAIN_SCENES), p=TERRAIN_PROBS)
        self._terrain_idx = int(idx)
        self.model = self._terrain_models[idx]
        self.data = self._terrain_datas[idx]
        self.body_id = self._terrain_body_ids[idx]
        self.foot_ids = self._terrain_foot_ids[idx]

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
        self._tilt_sum = 0.0
        self._tilt_count = 0
        self._ep_reward_sum = 0.0
        self.cpg.reset()
        self._q_target = JOINT_REF.copy()

        # Smooth random command: start from rest, approach a random target.
        self.cmd = np.zeros(3, dtype=np.float32)
        self._cmd_target = self._sample_target()
        return self._get_obs(), {}

    def step(self, action):
        # ── CPG target joints + RL residual -> joint angles -> simulation ──
        # The C++ tripod CPG + foot trajectory + analytic IK (same as NORMAL
        # mode) produce a target q at 200Hz. The policy adds a per-joint
        # residual (action in [-1,1] -> +-ACTION_SCALE rad) on top of it.
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        self._current_action = action.copy()

        self._update_cmd()   # smooth the random command toward its target

        for _ in range(SIM_STEPS_PER_CTRL):
            q_target = compute_joint_targets(self.cpg, self.traj,
                                             self.cmd[0], self.cmd[1])
            q = q_target + action * ACTION_SCALE
            self.data.ctrl[:18] = np.clip(q, -2.5, 2.5)
            mujoco.mj_step(self.model, self.data)
        self._q_target = q_target

        self.step_count += 1

        obs = self._get_obs()
        step_reward = self._compute_reward()
        self._ep_reward_sum += step_reward

        # Fixed-length episode: no early termination. The episode's cumulative
        # reward is delivered as a single sparse signal on the final step.
        terminated = False
        truncated = self.step_count >= EPISODE_STEPS
        reward = self._ep_reward_sum if truncated else 0.0
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        """Build the 159D observation."""
        d = self.data; m = self.model

        quat = d.xquat[self.body_id].copy() if self.body_id >= 0 else np.array([1., 0., 0., 0.])
        projected_gravity = quat_to_projected_gravity(quat)

        R = quat_to_rotation_matrix(quat)
        world_vel = d.qvel[0:6] if m.nv >= 6 else np.zeros(6)
        body_lin = R @ world_vel[0:3]
        body_ang = R @ world_vel[3:6]
        body_vel = np.array([body_lin[0], body_lin[1], body_ang[2]], dtype=np.float32)

        joint_pos = d.qpos[7:25].copy()
        joint_vel = d.qvel[6:24].copy()

        contacts = np.empty(6, dtype=np.float32)
        for i, fid in enumerate(self.foot_ids):
            if fid >= 0:
                tibia_rot = d.xmat[fid].reshape(3, 3)
                foot_tip_world = d.xpos[fid] + tibia_rot @ FOOT_TIP_TIBIA
                contacts[i] = 1.0 if foot_tip_world[2] < 0.015 else 0.0
            else:
                contacts[i] = 0.0

        prev_action = self._prev_action if self._prev_action is not None \
                      else np.zeros(N_JOINTS, dtype=np.float32)

        # 360° rangefinder -> robot-centric local height map (72 ground heights
        # = 24 azimuths x [near, mid, far]), appended to the observation.
        ranges = d.sensordata[:72] if m.nsensor >= 72 else np.full(72, -1.0)
        height_map = rangefinder_to_height_map(ranges)

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

    # ─── Reward (8 normalized terms, weighted) ──────────────────────────

    def _compute_reward(self):
        """8-term reward; each term normalized to [0,1] or [-1,0] before weighting."""
        d = self.data; m = self.model

        quat = d.xquat[self.body_id] if self.body_id >= 0 else np.array([1., 0., 0., 0.])
        R = quat_to_rotation_matrix(quat)
        world_vel = d.qvel[0:6] if m.nv >= 6 else np.zeros(6)
        body_lin = R @ world_vel[0:3]
        body_ang = R @ world_vel[3:6]

        cmd_vx, cmd_vy, _ = self.cmd
        dt = CTRL_DT
        action = self._current_action

        # 1. Velocity tracking — Gaussian kernel over the 2D horizontal velocity
        #    error (yaw cmd is zero). 1.0 = exact match, ->0 far off.
        err_v = np.array([body_lin[0] - cmd_vx, body_lin[1] - cmd_vy])
        r_vel = np.exp(-np.sum(err_v ** 2) / REWARD_SIGMA["vel"] ** 2)

        # 2. IMU gravity deviation (main stability) — Gaussian kernel over the
        #    squared distance from upright [0,0,-1]. 1.0 = perfectly upright.
        gb = quat_to_projected_gravity(quat)
        dev_g = np.sum((gb - np.array([0.0, 0.0, -1.0])) ** 2)
        r_gravity = np.exp(-dev_g / REWARD_SIGMA["gravity"] ** 2)

        # 3. Angular-velocity stability — Gaussian kernel over the full 3-axis
        #    body angular velocity. 1.0 = no rotation.
        r_ang_vel = np.exp(-np.sum(body_ang ** 2) / REWARD_SIGMA["ang_vel"] ** 2)

        # 4. Body height stability — minus squared body-height rate (dh/dt),
        #    penalizing vertical bobbing rather than absolute height.
        body_z = d.xpos[self.body_id][2] if self.body_id >= 0 else 0.124
        if self._prev_body_z is not None:
            dh = (body_z - self._prev_body_z) / dt
            r_height = -np.clip(dh ** 2 / REWARD_SIGMA["height_vel"] ** 2, 0.0, 1.0)
        else:
            r_height = 0.0
        self._prev_body_z = body_z

        # 5. Body acceleration penalty — normalized squared linear + angular
        #    acceleration from a finite difference of body velocity.
        if self._prev_body_lin is not None:
            a_lin = (body_lin - self._prev_body_lin) / dt
            a_ang = (body_ang - self._prev_body_ang) / dt
            r_accel = -np.clip(
                np.sum(a_lin ** 2) / REWARD_SIGMA["accel_lin"] ** 2 +
                np.sum(a_ang ** 2) / REWARD_SIGMA["accel_ang"] ** 2,
                0.0, 1.0)
        else:
            r_accel = 0.0
        self._prev_body_lin = body_lin.copy()
        self._prev_body_ang = body_ang.copy()

        # 6. Stance-foot slip penalty — squared horizontal velocity of feet that
        #    are in contact (a planted stance foot should not slide sideways).
        foot_tip = np.zeros((N_LEGS, 3), dtype=np.float32)
        contact = np.zeros(N_LEGS, dtype=bool)
        for i, fid in enumerate(self.foot_ids):
            if fid >= 0:
                tibia_rot = d.xmat[fid].reshape(3, 3)
                foot_tip[i] = d.xpos[fid] + tibia_rot @ FOOT_TIP_TIBIA
                contact[i] = foot_tip[i][2] < 0.015
        if self._prev_foot_tip is not None:
            v_foot = (foot_tip - self._prev_foot_tip) / dt
            slip_sq = np.sum(v_foot[:, 0:2] ** 2, axis=1)
            r_slip = -np.clip(np.sum(contact * slip_sq) /
                              (REWARD_SIGMA["slip"] ** 2 * N_LEGS), 0.0, 1.0)
        else:
            r_slip = 0.0
        self._prev_foot_tip = foot_tip.copy()

        # 7. Residual magnitude penalty — squared action averaged over joints.
        #    Keeps the RL residual as small as possible around the CPG target.
        r_action = -np.clip(np.sum(action ** 2) / N_JOINTS, 0.0, 1.0)

        # 8. Action-change penalty — squared per-step delta, averaged and scaled
        #    by the [-2,2] delta range (2x the action range) so it lands in [-1,0].
        if self._prev_action is not None:
            dact = action - self._prev_action
            r_action_rate = -np.clip(np.sum(dact ** 2) / (4.0 * N_JOINTS), 0.0, 1.0)
        else:
            r_action_rate = 0.0
        self._prev_action = action.copy()

        # Stability metric: body tilt angle from vertical (world -z).
        tilt = np.arccos(np.clip(-gb[2], -1.0, 1.0))
        self._tilt_sum += tilt
        self._tilt_count += 1

        # 9. Weighted sum of the normalized terms.
        total = (
            REWARD_WEIGHTS["vel"] * r_vel +
            REWARD_WEIGHTS["gravity"] * r_gravity +
            REWARD_WEIGHTS["ang_vel"] * r_ang_vel +
            REWARD_WEIGHTS["height"] * r_height +
            REWARD_WEIGHTS["accel"] * r_accel +
            REWARD_WEIGHTS["slip"] * r_slip +
            REWARD_WEIGHTS["action"] * r_action +
            REWARD_WEIGHTS["action_rate"] * r_action_rate
        )
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
    def __init__(self, eval_env, save_freq, n_eval_episodes=5, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_freq = save_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.writer = None

    def _on_training_start(self):
        self.writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, f"ppo_{int(time.time())}"))

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            rewards, tilts = [], []
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

            mean_r = np.mean(rewards)
            mean_tilt_deg = np.degrees(np.mean(tilts))

            print(f"[{self.n_calls:>8d} steps]  mean_reward={mean_r:+.3f}  "
                  f"tilt={mean_tilt_deg:.2f} deg  best={self.best_mean_reward:+.3f}")

            os.makedirs(CKPT_DIR, exist_ok=True)
            self.model.save(LATEST_PATH)

            if mean_r > self.best_mean_reward:
                self.best_mean_reward = mean_r
                self.model.save(BEST_PATH)
                print(f"  >>> New BEST model saved: reward={mean_r:+.3f}")

            if self.writer:
                self.writer.add_scalar("eval/mean_reward", mean_r, self.n_calls)
                self.writer.add_scalar("eval/mean_tilt", mean_tilt_deg, self.n_calls)
                self.writer.add_scalar("eval/best_reward", self.best_mean_reward, self.n_calls)

        return True

    def _on_training_end(self):
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


def _make_env():
    return Monitor(HexapodEnv())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=5_000_000)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--n_envs",      type=int, default=4)
    parser.add_argument("--save_freq",   type=int, default=5_000)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    n_envs = args.n_envs
    print(f"CPG+RL (residual policy): {n_envs} parallel envs on {os.cpu_count()} CPUs")
    print(f"  RL → 18 joint-angle increments (±{ACTION_SCALE} rad) → CPG target + residual")
    print(f"  terrain: {TERRAIN_SCENES} (probs {TERRAIN_PROBS})")
    env = DummyVecEnv([lambda: _make_env() for _ in range(n_envs)])
    # NOTE: SubprocVecEnv hangs with EGL backend; use DummyVecEnv instead
    eval_env = _make_env()

    policy_kwargs = dict(net_arch=dict(pi=[128, 64], vf=[128, 64]))

    ckpt_path = find_latest_checkpoint() if args.resume else None
    if ckpt_path:
        print(f"[Resume] Loading checkpoint: {ckpt_path}")
        model = PPO.load(ckpt_path, env=env, tensorboard_log=LOG_DIR)
    else:
        model = PPO("MlpPolicy", env, verbose=0,
                    n_steps=4096, batch_size=args.batch_size,
                    learning_rate=3e-4, ent_coef=0.01,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=LOG_DIR)

    eval_cb = EvalAndSaveCallback(eval_env, save_freq=args.save_freq, n_eval_episodes=3)

    model.learn(total_timesteps=args.total_steps, callback=eval_cb,
                reset_num_timesteps=(ckpt_path is None), progress_bar=True)

    final = os.path.join(CKPT_DIR, "final_model.zip")
    model.save(final)
    print(f"Final model: {final}")
    print(f"Best model: {BEST_PATH}" if os.path.exists(BEST_PATH) else "No best model saved")


if __name__ == "__main__":
    main()
