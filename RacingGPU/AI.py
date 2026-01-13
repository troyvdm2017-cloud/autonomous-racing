from stable_baselines3 import PPO , SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from racing_env import RacetrackEnv
from RacingSimulator import Racetrack, Simulator


def make_env():
    track = [Racetrack("track2.json", 500, 100, 0),
    Racetrack("track2.json", 500 , 50 , 0)]
    sim = Simulator([track])
    env = RacetrackEnv(sim)
    return Monitor(env)


env = DummyVecEnv([make_env])

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=4096,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    device= "cuda",
)
model.learn(total_timesteps=500_000)
model.save("ppo_racer_500k_track2")