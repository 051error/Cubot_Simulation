#!/usr/bin/env python3
"""CPG+RL training for hexapod walking — aligned with arXiv:2310.07744.

Architecture (matches "Terrain-adaptive CPGs with RL for Hexapod Locomotion"):
  RL policy → CPG foot-trajectory params (8D) → Hopf oscillators
  → foot positions in coxa frame (6×3D) → IK solver → joint angles (18D)
  → MuJoCo simulation

Reference: arXiv:2310.07744, Table I (reward), Fig.2 (CPG architecture),
           Section III-B (observation), Eq.5 (action space).
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
CTRL_DT  = 0.02                 # 50Hz control (paper: 200Hz policy, 1000Hz CPG)
SIM_STEPS_PER_CTRL = 10

# Foot tip in tibia local frame (mesh center - half_extent in Z)
FOOT_TIP_TIBIA = np.array([0.00162782, 0.16052104, 0.02951023])
EPISODE_STEPS = 200             # 4s per episode (fixed-length sparse reward)

# CPG action space: RL outputs 8 foot-trajectory params (paper Eq.5)
N_CPG_PARAMS = 8

# Observation layout (must match rl_inference.py exactly):
#   projected_gravity(3) + body_vel(3) + feet_pos(18) + feet_vel(18)
#   + foot_contact(6) + cmd(3) + prev_action(8) + osc_state(12) = 71
OBS_DIM = 3 + 3 + 18 + 18 + 6 + 3 + N_CPG_PARAMS + 12  # 71

from leg_ik import LegIK
from hexapod_cpg import HexapodCPG


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
    """Hexapod walking with CPG+IK, aligned with arXiv:2310.07744.

    Observation (71D):
      [0:3]    projected_gravity  (body-frame gravity direction)
      [3:6]    body_vel           (vx, vy, wz in body frame)
      [6:24]   feet_pos           (6 feet × xyz in body frame)
      [24:42]  feet_vel           (6 feet × vxyz in body frame)
      [42:48]  foot_contact       (1=touching, 0=air)
      [48:51]  cmd                (joystick velocity command)
      [51:59]  prev_action        (previous CPG params, for smoothing)
      [59:71]  osc_state          (Hopf oscillator x,y for 6 legs)

    Action (8D): CPG foot-trajectory parameters (paper Eq.5).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(MODEL_XML)
        self.data   = mujoco.MjData(self.model)

        self.foot_names = [
            "tibia_rf", "tibia_rm", "tibia_rr",
            "tibia_lf", "tibia_lm", "tibia_lr",
        ]
        self.foot_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
                         for n in self.foot_names]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_CPG_PARAMS,), dtype=np.float32)

        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "MP_BODY")
        self.render_mode = render_mode
        self.viewer = None

        # CPG layer — foot trajectory generator
        self.cpg = HexapodCPG(n_legs=N_LEGS, dt=CTRL_DT)

        # Episode state
        self.step_count = 0
        self.cmd = np.zeros(3)
        self._prev_foot_pos = None

        # Reward state
        self._prev_action = None
        self._prev_joint_vel = None
        self._feet_air_time = np.zeros(N_LEGS, dtype=np.float32)
        self._feet_contact_prev = np.zeros(N_LEGS, dtype=bool)
        self._ep_reward_sum = 0.0   # accumulated reward for fixed-length episode

        # Success metric
        self._vel_error_sum = 0.0
        self._vel_error_count = 0

        # IK failure counter (for debugging)
        self._ik_fails = 0

    def _sample_cmd(self):
        sign = lambda: 1 if np.random.random() < 0.5 else -1
        # Command range aligned with the halved /upper_ctrl limits:
        # linear ±0.05 m/s. No angular command — rotation is delegated to the
        # dedicated TURN mode, so the yaw command is always zero.
        self.cmd = np.array([
            np.random.uniform(0.02, 0.05) * sign(), # vx: bidirectional [2, 5] cm/s
            np.random.uniform(0.02, 0.05) * sign(), # vy: bidirectional [2, 5] cm/s
            0.0,                                    # wz: rotation handled by TURN mode
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # Start from the crouched "home" keyframe (if defined) instead of the
        # fully-extended default pose. Keeps foot tips on the ground.
        if self.model.nkey > 0:
            self.data.qpos[:] = self.model.key_qpos[0]
            mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self._prev_foot_pos = None
        self._prev_action = None
        self._prev_joint_vel = None
        self._feet_air_time[:] = 0.0
        self._feet_contact_prev[:] = False
        self._vel_error_sum = 0.0
        self._vel_error_count = 0
        self._ep_reward_sum = 0.0
        self.cpg.reset()
        LegIK.clear_cache()  # reset warm starts for new episode

        self._sample_cmd()
        return self._get_obs(), {}

    def step(self, action):
        # ── CPG → foot positions → IK → joint angles → simulation ──────
        # Coxa-driven gait: coxa_amp is primary speed control.
        # No forward bias — RL learns full speed range from command alone.
        # action[0] ∈ [-1,1] → coxa_amp ∈ [0.005, 0.025]m → speed ∈ [0, ~11] cm/s.
        foot_targets = self.cpg.step(action)

        # Solve IK for each leg independently.
        # CPG outputs in "logical" frame: [x=lateral, y=forward, z=vertical]
        # c1_rest frame: X=vertical, Y=forward, Z=lateral
        # Remapping: target_c1 = [cpg_z, cpg_y, cpg_x]
        joint_targets = np.empty(N_JOINTS, dtype=np.float32)
        for i in range(N_LEGS):
            j = i * 3
            angles = LegIK.solve(
                foot_targets[j + 2],  # c1_rest X = vertical (CPG z)
                foot_targets[j + 1],  # c1_rest Y = forward  (CPG y)
                foot_targets[j + 0],  # c1_rest Z = lateral  (CPG x)
                leg_idx=i,
            )
            joint_targets[j:j+3] = angles

        self.data.ctrl[:18] = np.clip(joint_targets, -2.5, 2.5)

        for _ in range(SIM_STEPS_PER_CTRL):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        # Fixed-length episode: the velocity command is sampled once in reset()
        # and held constant for the whole episode (no mid-episode resampling).
        self._current_action = action.copy()  # store actual CPG action applied

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
        """Build observation (71D) aligned with paper Section III-B.

        Layout: projected_gravity(3) + body_vel(3) + feet_pos(18)
                + feet_vel(18) + foot_contact(6) + cmd(3)
                + prev_action(8) + osc_state(12)
        """
        d = self.data; m = self.model

        quat = d.xquat[self.body_id].copy() if self.body_id >= 0 else np.array([1., 0., 0., 0.])
        projected_gravity = quat_to_projected_gravity(quat)

        R = quat_to_rotation_matrix(quat)
        world_vel = d.qvel[0:6] if m.nv >= 6 else np.zeros(6)
        body_lin = R @ world_vel[0:3]
        body_ang = R @ world_vel[3:6]
        body_vel = np.array([body_lin[0], body_lin[1], body_ang[2]], dtype=np.float32)

        body_pos = d.xpos[self.body_id].copy() if self.body_id >= 0 else np.zeros(3)
        feet_pos = np.empty(18, dtype=np.float32)
        feet_vel = np.empty(18, dtype=np.float32)
        contacts = np.empty(6, dtype=np.float32)

        prev_feet = self._prev_foot_pos
        dt = CTRL_DT

        for i, fid in enumerate(self.foot_ids):
            j = i * 3
            if fid >= 0:
                fp_world = d.xpos[fid].copy()
                fp_body = R @ (fp_world - body_pos)
                feet_pos[j:j+3] = fp_body

                if prev_feet is not None:
                    fv_world = (fp_world - prev_feet[i]) / dt
                else:
                    fv_world = np.zeros(3)
                feet_vel[j:j+3] = R @ fv_world

                tibia_rot = d.xmat[fid].reshape(3, 3)
                foot_tip_world = fp_world + tibia_rot @ FOOT_TIP_TIBIA
                contacts[i] = 1.0 if foot_tip_world[2] < 0.015 else 0.0
            else:
                feet_pos[j:j+3] = 0.0
                feet_vel[j:j+3] = 0.0
                contacts[i] = 0.0

        self._prev_foot_pos = [d.xpos[fid].copy() if fid >= 0 else np.zeros(3)
                               for fid in self.foot_ids]

        prev_action = self._prev_action if self._prev_action is not None \
                      else np.zeros(N_CPG_PARAMS, dtype=np.float32)

        osc_state = self.cpg.get_osc_state()

        obs = np.concatenate([
            projected_gravity,
            body_vel / np.array([0.3, 0.3, 2.0]),
            feet_pos / 0.2,
            feet_vel / 0.5,
            contacts,
            self.cmd.copy() / np.array([0.05, 0.05, 1.0]),  # normalized to /upper_ctrl limits
            prev_action,
            osc_state,                                      # in [-1, 1]
        ]).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    # ─── Reward (paper Table I) ─────────────────────────────────────────

    def _compute_reward(self):
        """11-term reward from CPG+RL paper Table I, with cmd-relative σ².

        Paper uses all terms with dt scaling (dt=0.005). Our dt=0.02 so
        we omit dt scaling to keep reasonable reward magnitudes.
        Penalty weights reduced 10× (joint pos, torque) to compensate for
        CPG-driven joint motion vs paper's hip-only penalty.
        """
        d = self.data; m = self.model

        quat = d.xquat[self.body_id] if self.body_id >= 0 else np.array([1., 0., 0., 0.])
        R = quat_to_rotation_matrix(quat)
        world_vel = d.qvel[0:6] if m.nv >= 6 else np.zeros(6)
        body_lin = R @ world_vel[0:3]
        body_ang = R @ world_vel[3:6]

        cmd_vx, cmd_vy, cmd_wz = self.cmd
        dt = CTRL_DT

        # 1a. Forward velocity tracking — tight sigma to force speed modulation.
        #    sigma floor 0.001 → sigma=3.2cm/s. Robot speed range is 3-11 cm/s,
        #    so the policy must learn to modulate speed to match cmd, not just
        #    walk at a single fixed speed.
        err_vx_sq = (body_lin[0] - cmd_vx)**2
        sigma_sq_vx = max(cmd_vx**2 * 0.2, 0.001)
        r_lin_vel_x = np.exp(-err_vx_sq / sigma_sq_vx) * 3.0

        # 1b. Lateral velocity tracking (slightly wider — robot has inherent drift)
        err_vy_sq = (body_lin[1] - cmd_vy)**2
        sigma_sq_vy = max(max(abs(cmd_vy), 0.02)**2 * 0.2, 0.002)
        r_lin_vel_y = np.exp(-err_vy_sq / sigma_sq_vy) * 1.0

        # 2. Forward velocity shaping — small bonus for moving in cmd direction.
        #    Capped at cmd_vx and reduced weight so it doesn't dominate tracking.
        if abs(cmd_vx) > 0.02:
            actual = body_lin[0] * np.sign(cmd_vx)
            r_forward = np.clip(actual, 0.0, abs(cmd_vx)) * 2.0
        else:
            r_forward = 0.0

        # 2. Yaw suppression (cmd_wz is always 0 — rotation is TURN mode's job,
        #    so this term only penalizes unwanted yaw drift).
        err_wz_sq = body_ang[2]**2
        sigma_sq_ang = 0.05
        r_ang_vel = np.exp(-err_wz_sq / sigma_sq_ang) * 1.0

        # 3. Linear velocity Z penalty  (paper: −1dt)
        r_lin_vel_z = -(body_lin[2]**2) * 1.0

        # 4. Angular velocity XY penalty  (paper: −0.05dt)
        r_ang_vel_xy = -(body_ang[0]**2 + body_ang[1]**2) * 0.05

        # 5. Joint position penalty  (paper: −0.5dt hip-only; we: −0.005 all 18)
        joint_pos = d.qpos[7:25]
        r_joint_pos = -np.sum(np.square(joint_pos)) * 0.005

        # 6. Joint velocity penalty  (paper: −0.001dt)
        joint_vel = d.qvel[6:24]
        r_joint_vel = -np.sum(np.square(joint_vel)) * 0.001

        # 7. Joint acceleration penalty  (paper: −2.5e−7dt)
        if self._prev_joint_vel is not None:
            joint_acc = (joint_vel - self._prev_joint_vel) / CTRL_DT
            r_joint_acc = -np.sum(np.square(joint_acc)) * 2.5e-7
        else:
            r_joint_acc = 0.0
        self._prev_joint_vel = joint_vel.copy()

        # 8. Action rate penalty  (paper: −0.01dt; we: −0.002)
        action = self._current_action
        if self._prev_action is not None:
            r_action_rate = -np.sum(np.square(action - self._prev_action)) * 0.002
        else:
            r_action_rate = 0.0
        self._prev_action = action.copy()

        # 9. Torque penalty  (paper: −1e−4dt; we: −1e−5)
        r_torque = -np.sum(np.square(d.qfrc_actuator[:18])) * 1e-5

        # 10. Collision penalty  (paper: −1dt)
        body_z = d.xpos[self.body_id][2] if self.body_id >= 0 else 1.0
        r_collision = -1.0 if body_z < 0.05 else 0.0

        # 11. Feet air time  (paper: +1dt)
        r_feet_air = 0.0
        cmd_mag = abs(cmd_vx) + abs(cmd_vy) + abs(cmd_wz)
        if cmd_mag > 0.02:   # always true: vx,vy sampled in [0.02, 0.05]
            for i, fid in enumerate(self.foot_ids):
                if fid >= 0:
                    tibia_rot = d.xmat[fid].reshape(3, 3)
                    foot_tip_world = d.xpos[fid] + tibia_rot @ FOOT_TIP_TIBIA
                    h = foot_tip_world[2]
                    in_contact = h < 0.015
                    was_in_contact = self._feet_contact_prev[i]
                    if in_contact and not was_in_contact:
                        r_feet_air += (self._feet_air_time[i] - 0.5)
                        self._feet_air_time[i] = 0.0
                    elif not in_contact:
                        self._feet_air_time[i] += dt
                    self._feet_contact_prev[i] = in_contact
        r_feet_air *= 1.0

        # Tracking error for success metric
        err_vx = abs(body_lin[0] - cmd_vx)
        err_vy = abs(body_lin[1] - cmd_vy)
        err_wz = abs(body_ang[2] - cmd_wz)
        norm_err = (err_vx + err_vy + err_wz) / max(cmd_mag, 0.15)
        self._vel_error_sum += norm_err
        self._vel_error_count += 1

        total = (r_lin_vel_x + r_lin_vel_y + r_forward + r_ang_vel + r_lin_vel_z + r_ang_vel_xy +
                 r_joint_pos + r_joint_vel + r_joint_acc + r_action_rate +
                 r_torque + r_feet_air + r_collision)
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
    def tracking_error(self):
        if self._vel_error_count == 0:
            return float('inf')
        return self._vel_error_sum / self._vel_error_count

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
    def __init__(self, eval_env, save_freq, n_eval_episodes=5, verbose=1, log_freq=5000):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_freq = save_freq
        self.log_freq = log_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.writer = None

    def _on_training_start(self):
        self.writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, f"ppo_{int(time.time())}"))

    def _on_step(self):
        # Progress log every log_freq steps (lightweight, no eval)
        if self.n_calls % self.log_freq == 0:
            buf = getattr(self.model, 'ep_info_buffer', None)
            if buf is not None and len(buf) > 0:
                # Convert deque to list for slicing
                buf_list = list(buf)
                recent_n = min(20, len(buf_list))
                recent = buf_list[-recent_n:]
                avg_rew = np.mean([e['r'] for e in recent])
                avg_len = np.mean([e['l'] for e in recent])
                print(f"[{self.n_calls:>8d}] train_ep_rew(20ep)={avg_rew:+.3f}  "
                      f"ep_len={avg_len:.0f}")
                if self.writer:
                    self.writer.add_scalar("train/ep_rew_mean", avg_rew, self.n_calls)

        if self.n_calls % self.save_freq == 0:
            rewards, successes, tracking_errors = [], 0, []
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                ep_reward = 0.0
                terminated, truncated = False, False
                while not (terminated or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, r, terminated, truncated, _ = self.eval_env.step(action)
                    ep_reward += r
                rewards.append(ep_reward)
                vel_err = self.eval_env.unwrapped.tracking_error
                tracking_errors.append(vel_err)
                if not terminated and vel_err < 0.6:
                    successes += 1

            mean_r = np.mean(rewards)
            mean_track_err = np.mean(tracking_errors)
            success_rate = successes / self.n_eval_episodes

            print(f"[{self.n_calls:>8d} steps]  mean_reward={mean_r:+.3f}  "
                  f"success={success_rate:.1%}  "
                  f"track_err={mean_track_err:.3f}  best={self.best_mean_reward:+.3f}")

            os.makedirs(CKPT_DIR, exist_ok=True)
            self.model.save(LATEST_PATH)

            if mean_r > self.best_mean_reward:
                self.best_mean_reward = mean_r
                self.model.save(BEST_PATH)
                print(f"  >>> New BEST model saved: reward={mean_r:+.3f}")

            if self.writer:
                self.writer.add_scalar("eval/mean_reward", mean_r, self.n_calls)
                self.writer.add_scalar("eval/success_rate", success_rate, self.n_calls)
                self.writer.add_scalar("eval/mean_tracking_error", mean_track_err, self.n_calls)
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
    print(f"CPG+RL (paper-aligned): {n_envs} parallel envs on {os.cpu_count()} CPUs")
    print(f"  RL → {N_CPG_PARAMS} foot-trajectory params → Hopf CPG → IK → 18 joints")
    env = DummyVecEnv([_make_env for _ in range(n_envs)])
    # NOTE: SubprocVecEnv hangs with EGL backend; use DummyVecEnv instead
    eval_env = _make_env()

    policy_kwargs = dict(net_arch=dict(pi=[128, 64], vf=[128, 64]))

    ckpt_path = find_latest_checkpoint() if args.resume else None
    if ckpt_path:
        print(f"[Resume] Loading checkpoint: {ckpt_path}")
        model = PPO.load(ckpt_path, env=env, tensorboard_log=LOG_DIR)
    else:
        model = PPO("MlpPolicy", env, verbose=1,
                    n_steps=4096, batch_size=args.batch_size,
                    learning_rate=3e-4, ent_coef=0.01,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=LOG_DIR)

    eval_cb = EvalAndSaveCallback(eval_env, save_freq=args.save_freq, n_eval_episodes=3)

    model.learn(total_timesteps=args.total_steps, callback=eval_cb,
                reset_num_timesteps=(ckpt_path is None))

    final = os.path.join(CKPT_DIR, "final_model.zip")
    model.save(final)
    print(f"Final model: {final}")
    print(f"Best model: {BEST_PATH}" if os.path.exists(BEST_PATH) else "No best model saved")


if __name__ == "__main__":
    main()
