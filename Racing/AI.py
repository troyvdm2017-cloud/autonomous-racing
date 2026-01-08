from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from racing_env import RacetrackEnv
from RacingSimulator import Racetrack, Simulator


def make_env():
    track = Racetrack("track1.json", 500, 100, 0)
    sim = Simulator([track])
    env = RacetrackEnv(sim)
    return Monitor(env)


env = DummyVecEnv([make_env])

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    device= "cpu",
)

model.learn(total_timesteps=5_000_000)
model.save("ppo_racer_500k")