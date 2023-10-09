
import torch

import gymnasium as gym
import panda_gym

env = gym.make('PandaReach-v3', render_mode = "human")

observation, info = env.reset()

def testEnv():	
    for _ in range(1000):
        action = env.action_space.sample() # random action
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

if __name__ == "__main__":
	print("cuda is available", torch.cuda.is_available())
	print("hello world")
	testEnv()
	print("passed Env Testing")



