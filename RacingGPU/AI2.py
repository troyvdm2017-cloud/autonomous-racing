import torch
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from racing_env2 import RacetrackEnv
from RacingSimulator import Racetrack, Simulator


def make_env():
    def _init():
        print(f"Env gestartet in Prozess PID={os.getpid()}")
        track = [
            Racetrack("track1.json", 500, 100, 0),
            Racetrack("track2.json", 500, 140, 0),
            Racetrack("track1.json", 500, 1060, 0),
            Racetrack("track2.json", 400, 480, 0)
        ]
        sim = Simulator(track)
        env = RacetrackEnv(sim)
        return Monitor(env)
    return _init


if __name__ == "__main__":
    num_env = 8
    vec_env = SubprocVecEnv([make_env() for _ in range(num_env)])



    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[128, 128, 64],  # Policy-Netz
                vf=[256, 256, 256]  # Value-Netz
            )
        ),
        learning_rate = 5e-4,
        n_steps=4096,   # pro Env
        batch_size=4096,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef = 0.002,
        clip_range=0.2,
        verbose=1,
        device="cuda",

    )

    model.learn(total_timesteps=10_000_000)
    model.save("ppo_racer_10m_30gneurewards4tracks")