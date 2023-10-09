import torch
import gymnasium as gym
import panda_gym
import time
from stable_baselines3 import SAC
from stable_baselines3 import DDPG, HerReplayBuffer
#from stable_baselines3.common.envs import VecNormalize




if __name__ == "__main__":
    total_timesteps = 100000
    env = gym.make('PandaReach-v3') #render_mode="human")
    model = SAC.load("sac_pandareach{}".format(total_timesteps), env = env, print_system_info = False)

    vec_env = model.get_env()
    obs = vec_env.reset()
    episode_num = 100
    episode_return = []
    
    for i in range(episode_num):
        time.sleep(0.1)
        episode_reward = 0
        time.sleep(0.01)
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
        while not done:
            action, _state = model.predict(obs, deterministic = True)
            obs, reward, done, info = vec_env.step(action)
            episode_reward += reward
        episode_return.append(episode_reward)
    print(episode_return)
    print("average return", sum(episode_return)/episode_num)
            