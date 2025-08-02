import gymnasium as gym
import cv2
import numpy as np
from collections import deque
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import imageio
import os


def preprocess_frame(frame, resolution=(84, 84), grayscale=True):
    if grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, resolution)
    frame = frame.astype(np.uint8)
    if grayscale:
        frame = np.expand_dims(frame, axis=-1)
    return frame


class FrameStack:
    def __init__(self, k, resolution=(84, 84), grayscale=True):
        self.k = k
        self.frames = deque(maxlen=k)
        self.resolution = resolution
        self.grayscale = grayscale

    def reset(self, obs):
        frame = preprocess_frame(obs, self.resolution, self.grayscale)
        for _ in range(self.k):
            self.frames.append(np.copy(frame))
        return np.concatenate(self.frames, axis=-1)
 
    def step(self, obs):
        frame = preprocess_frame(obs, self.resolution, self.grayscale)
        self.frames.append(frame)
        return np.concatenate(self.frames, axis=-1)


class PreprocessedCarRacing(gym.Wrapper):
    def __init__(self, env, frame_stack=4, resolution=(84, 84), grayscale=True):
        super().__init__(env)
        self.frame_stack = FrameStack(frame_stack, resolution, grayscale)
        channels = frame_stack if grayscale else 3 * frame_stack
        self.observation_space = spaces.Box(
            low=0, high=255 , shape=(channels, resolution[0], resolution[1]), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        stacked = self.frame_stack.reset(obs)
        stacked = np.transpose(stacked, (2, 0, 1))  # [H, W, C] -> [C, H, W]
        return stacked, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        stacked = self.frame_stack.step(obs)
        stacked = np.transpose(stacked, (2, 0, 1))  # [H, W, C] -> [C, H, W]
        return stacked, reward, terminated, truncated, info


def make_env():
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    env = PreprocessedCarRacing(env, frame_stack=4)
    return env


def evaluate_and_record(model_path, output_video_path="ppo_carracing_output.mp4", fps=30):
    env = make_env()
    model = PPO.load(model_path)
    obs, _ = env.reset()
    done = False
    frames = []
    while not done:
        action, _ = model.predict(obs)
        obs, __, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frames.append(env.render())
    imageio.mimsave(output_video_path, frames, fps=fps)


def train_ppo_agent(total_timesteps=100_000, model_save_path="ppo_carracing"):
    env = DummyVecEnv([make_env])
    model = PPO("CnnPolicy", env, verbose=1)
    checkpoint_interval = 2000
    for step in range(0, total_timesteps, checkpoint_interval):
        model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False)
        model.save(model_save_path)
        video_path = f"ppo_carracing_output_step_{step + checkpoint_interval}.mp4"
        evaluate_and_record(model_path=model_save_path, output_video_path=video_path)


if __name__ == "__main__":
    train_ppo_agent(total_timesteps=100_000, model_save_path="ppo_carracing")  