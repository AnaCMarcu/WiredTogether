import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common import logger
import gymnasium as gym
import craftium

def ppo_train():
    env = gym.make("Craftium/ChopTree-v0")

    model = PPO("CnnPolicy", env, verbose=1)

    new_logger = logger.configure("logs-ppo-agent", ["stdout", "csv"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=1_000_000)

    env.close()

def init_test():
    # Create environment with video recording
    env = gym.make("Craftium/ChopTree-v0", render_mode="rgb_array")
    env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True)
    observation, info = env.reset()

    for t in range(20):
        action = env.action_space.sample()

        # plot the observation
        plt.clf()
        plt.imshow(np.transpose(observation, (1, 0, 2)))
        plt.pause(0.02)  # wait for 0.02 seconds

        observation, reward, terminated, truncated, _info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

if __name__ == "__main__":
    init_test()