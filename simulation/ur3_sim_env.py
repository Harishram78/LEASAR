"""
LEASAR -- UR3 PyBullet Simulation Environment
==============================================
6-DOF UR3 robot simulation with:
  - Realistic joint dynamics and sensor noise
  - Synthetic fault injection (joint_drift, torque_spike, sensor_noise, collision)
  - Observation space matching real UR3 sensor streams (24-dim)
  - Gymnasium/Gym-compatible interface for SAC/PPO/DQN training
  - Sim-to-real transfer: same obs vector as pipeline/data_preprocessing.py

Author  : Harish Ramachandran
Inst    : SRM Institute of Science and Technology, Chennai
License : Apache 2.0
"""
import numpy as np
import pybullet as pb
import pybullet_data
from typing import Optional, Tuple, Dict

# Gymnasium/Gym dual compatibility -----------------------------------------
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    _GYMNASIUM = False

# -- Constants ----------------------------------------------------------------
UR3_JOINT_LIMITS_LOW  = np.array([-2 * np.pi] * 6, dtype=np.float32)
UR3_JOINT_LIMITS_HIGH = np.array([ 2 * np.pi] * 6, dtype=np.float32)
UR3_TORQUE_LIMITS     = np.array([56.0, 56.0, 28.0, 12.0, 12.0, 12.0])  # Nm
UR3_VELOCITY_LIMITS   = np.array([3.14] * 6)  # rad/s

FAULT_TYPES = ["none", "joint_drift", "torque_spike", "sensor_noise", "collision"]

# Observation: 6 joint pos + 6 joint vel + 6 torque + 3 TCP xyz + 3 TCP rpy = 24
OBS_DIM = 24
ACT_DIM = 6  # delta joint positions


class UR3SimEnv(gym.Env):
    """
    PyBullet-based UR3 6-DOF simulation environment for LEASAR.

    Supports synthetic fault injection for training anomaly detectors
    without requiring physical robot access (sim-to-real transfer).

    Observation vector (24-dim) is identical to real UR3 sensor stream:
        [joint_pos x6, joint_vel x6, joint_torque x6, TCP_xyz x3, TCP_rpy x3]

    Action space: Box(-0.05, 0.05, shape=(6,)) -- delta joint positions (rad)
    Compatible with SACAgent(state_dim=24, action_dim=6) in rl/sac_agent.py.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        fault_prob: float = 0.15,
        noise_std: float = 0.01,
        max_steps: int = 500,
        urdf_path: Optional[str] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.fault_prob = fault_prob
        self.noise_std = noise_std
        self.max_steps = max_steps
        self.urdf_path = urdf_path or "ur3/ur3.urdf"

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-0.05, high=0.05, shape=(ACT_DIM,), dtype=np.float32
        )

        self._physics_client = None
        self._robot_id = None
        self._joint_indices = list(range(6))
        self._step_count = 0
        self._active_fault = "none"
        self._fault_magnitude = 0.0
        self._connect()

    # -- Physics ---------------------------------------------------------------
    def _connect(self):
        use_gui = self.render_mode == "human"
        mode = pb.GUI if use_gui else pb.DIRECT
        self._physics_client = pb.connect(mode)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, -9.81)

    def _load_robot(self):
        pb.resetSimulation()
        pb.setGravity(0, 0, -9.81)
        pb.loadURDF("plane.urdf")
        try:
            robot_id = pb.loadURDF(
                self.urdf_path,
                basePosition=[0, 0, 0],
                useFixedBase=True,
            )
        except Exception:
            # Fallback to KUKA iiwa if UR3 URDF not present locally
            robot_id = pb.loadURDF(
                "kuka_iiwa/model.urdf",
                basePosition=[0, 0, 0],
                useFixedBase=True,
            )
        return robot_id

    # -- Reset -----------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """Reset environment. Returns (obs, info) for gymnasium; obs for gym."""
        if seed is not None:
            np.random.seed(seed)
        if self._robot_id is not None:
            pb.removeBody(self._robot_id)
        self._robot_id = self._load_robot()
        self._step_count = 0
        self._active_fault = "none"
        self._fault_magnitude = 0.0

        home_pos = np.random.uniform(-0.3, 0.3, size=6).astype(np.float32)
        for i, idx in enumerate(self._joint_indices):
            pb.resetJointState(self._robot_id, idx, home_pos[i])
        pb.stepSimulation()

        obs = self._get_obs()
        info = {"fault": "none", "fault_magnitude": 0.0}
        if _GYMNASIUM:
            return obs, info
        return obs  # legacy gym compatibility

    # -- Step ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        """
        Step environment. Returns gymnasium tuple:
            (obs, reward, terminated, truncated, info)
        Falls back to old gym (obs, reward, done, info) if gym<0.26.
        """
        self._step_count += 1

        if np.random.rand() < self.fault_prob and self._active_fault == "none":
            self._inject_fault()

        current_pos = self._get_joint_positions()
        target_pos = np.clip(
            current_pos + action,
            UR3_JOINT_LIMITS_LOW,
            UR3_JOINT_LIMITS_HIGH,
        )
        for i, idx in enumerate(self._joint_indices):
            pb.setJointMotorControl2(
                self._robot_id,
                idx,
                controlMode=pb.POSITION_CONTROL,
                targetPosition=float(target_pos[i]),
                maxVelocity=float(UR3_VELOCITY_LIMITS[i]),
            )
        pb.stepSimulation()

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        terminated = False  # no terminal success state in this task
        truncated = self._step_count >= self.max_steps
        info = {
            "fault": self._active_fault,
            "fault_magnitude": self._fault_magnitude,
            "step": self._step_count,
        }
        if _GYMNASIUM:
            return obs, reward, terminated, truncated, info
        # Legacy gym: combine terminated and truncated into done
        return obs, reward, terminated or truncated, info

    # -- Observation -----------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        joint_pos, joint_vel, joint_torque = self._get_joint_states()
        tcp_pos, tcp_rpy = self._get_tcp_pose()
        noise = np.random.normal(0, self.noise_std, size=OBS_DIM).astype(np.float32)
        obs = np.concatenate([
            joint_pos, joint_vel, joint_torque, tcp_pos, tcp_rpy
        ]).astype(np.float32) + noise
        return self._apply_fault_to_obs(obs)

    def _get_joint_states(self):
        states = [pb.getJointState(self._robot_id, i) for i in self._joint_indices]
        pos    = np.array([s[0] for s in states], dtype=np.float32)
        vel    = np.array([s[1] for s in states], dtype=np.float32)
        torque = np.array([s[3] for s in states], dtype=np.float32)
        return pos, vel, torque

    def _get_joint_positions(self) -> np.ndarray:
        pos, _, _ = self._get_joint_states()
        return pos

    def _get_tcp_pose(self):
        link_state = pb.getLinkState(
            self._robot_id,
            self._joint_indices[-1],
            computeForwardKinematics=True,
        )
        pos = np.array(link_state[4], dtype=np.float32)
        rpy = np.array(pb.getEulerFromQuaternion(link_state[5]), dtype=np.float32)
        return pos, rpy

    # -- Fault injection -------------------------------------------------------
    def _inject_fault(self):
        self._active_fault = np.random.choice(FAULT_TYPES[1:])
        self._fault_magnitude = float(np.random.uniform(0.1, 1.0))

    def _apply_fault_to_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.copy()
        m = self._fault_magnitude
        if self._active_fault == "joint_drift":
            idx = np.random.randint(0, 6)
            obs[idx] += m * 0.5
        elif self._active_fault == "torque_spike":
            idx = np.random.randint(0, 6)
            obs[12 + idx] += m * UR3_TORQUE_LIMITS[idx]
        elif self._active_fault == "sensor_noise":
            obs += np.random.normal(0, m * 0.5, size=OBS_DIM).astype(np.float32)
        elif self._active_fault == "collision":
            obs[18:21] += np.random.uniform(-m, m, size=3).astype(np.float32)
            obs[12:18] += m * UR3_TORQUE_LIMITS * 0.8
        return obs

    # -- Reward ----------------------------------------------------------------
    def _compute_reward(self, obs: np.ndarray) -> float:
        torques      = obs[12:18]
        torque_norm  = float(np.linalg.norm(torques) / np.linalg.norm(UR3_TORQUE_LIMITS))
        anomaly_pen  = -self._fault_magnitude if self._active_fault != "none" else 0.0
        energy_pen   = -0.01 * torque_norm
        return anomaly_pen + energy_pen

    # -- Render / Close --------------------------------------------------------
    def render(self):
        """Rendering is handled by PyBullet GUI mode (render_mode='human')."""
        pass

    def close(self):
        if self._physics_client is not None:
            pb.disconnect(self._physics_client)
            self._physics_client = None


# -- Factory ------------------------------------------------------------------
def make_ur3_env(render: bool = False, fault_prob: float = 0.15) -> UR3SimEnv:
    """Create a LEASAR UR3 simulation environment."""
    render_mode = "human" if render else None
    return UR3SimEnv(render_mode=render_mode, fault_prob=fault_prob)


# -- Demo ---------------------------------------------------------------------
if __name__ == "__main__":
    print("LEASAR UR3 Simulation -- Quick Demo")
    env = make_ur3_env(render=False, fault_prob=0.3)

    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    print(f"Obs shape : {obs.shape}")
    assert obs.shape == (OBS_DIM,), f"Expected (24,), got {obs.shape}"

    total = 0.0
    for step in range(200):
        action = env.action_space.sample()
        result = env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        total += reward
        if info["fault"] != "none":
            print(f"  Step {step:3d} | FAULT: {info['fault']:15s} "
                  f"| mag={info['fault_magnitude']:.2f} | r={reward:.3f}")
        if done:
            break

    print(f"Total reward: {total:.2f}")
    env.close()
    print("Demo PASSED.")
