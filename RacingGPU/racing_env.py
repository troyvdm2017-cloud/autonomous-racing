import gymnasium as gym
from gymnasium import spaces
import numpy as np


class RacetrackEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        simulator,
        sensor_opening_angle=30,
        sensor_pixels=60,
        max_distance=450,
        max_steps=1000,
    ):
        super().__init__()

        self.sim = simulator
        self.sensor_opening_angle = sensor_opening_angle
        self.sensor_pixels = sensor_pixels
        self.max_distance = max_distance
        self.max_steps = max_steps

        self.step_count = 0

        # === Observation space ===
        # Sensor readings ∈ [0, 1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(sensor_pixels,),
            dtype=np.float32,
        )

        # === Action space ===
        # throttle, steering
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        obs = self.sim.reset()
        self.step_count = 0

        return obs.astype(np.float32), {}

    def step(self, action):
        self.step_count += 1

        throttle = float(action[0])
        steering = float(action[1])

        # === Action scaling ===
        # throttle ∈ [-1, 1] → Zielgeschwindigkeit
        target_velocity = (
            (throttle + 1.0) / 2.0 * self.sim.max_velocity
        )

        # steering ∈ [-1, 1] → Lenkwinkel pro Step
        steering_angle = steering * 5.0  # degrees per step (tune!)

        obs, reward, done = self.sim.step(
            velocity=target_velocity,
            steering_angle=steering_angle,
        )

        # === Reward shaping ===
        reward = float(reward)

        # encourage movement
        reward += 0.01 * self.sim.velocity

        # penalty for standing still
        if self.sim.velocity < 1.0:
            reward -= 0.1

        truncated = self.step_count >= self.max_steps

        info = {
            "velocity": self.sim.velocity,
            "x": self.sim.x,
            "y": self.sim.y,
            "angle": self.sim.angle,
        }

        return obs.astype(np.float32), reward, done, truncated, info

    def render(self):
        # Rendering läuft bei dir über LiveVisualizer
        pass

    def close(self):
        pass