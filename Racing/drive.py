import time
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from RacingSimulator import Racetrack, Simulator, LiveVisualizer
from racing_env import RacetrackEnv


MODEL_PATH = "ppo_racer_500k"


def main():
    track = Racetrack("track1.json", 500, 100, 0)
    sim = Simulator([track])
    env = RacetrackEnv(sim)

    model = PPO.load(MODEL_PATH)

    obs, _ = env.reset()
    viz = LiveVisualizer(sim)

    done = truncated = False

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        viz.update(obs, reward, done)
        plt.pause(0.001)   # <<< WICHTIG
        time.sleep(0.0003)

    print("🏁 Run finished")
    plt.show()  # <<< BLOCKIERT & HÄLT DAS FENSTER OFFEN


if __name__ == "__main__":
    main()