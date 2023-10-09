import torch
import gymnasium as gym
import panda_gym
import time
from stable_baselines3 import SAC, TD3
from stable_baselines3 import DDPG, HerReplayBuffer
#from stable_baselines3.common.envs import VecNormalize


def check_device():
    assert torch.cuda.is_available, "cuda not available"


def train(model, total_timesteps, policy_name):
    print("Let's start training ------")
    begin_time = time.time()

    model.learn(total_timesteps)
    model.save("{}_pandareach{}".format(policy_name, total_timesteps))
    del model

    end_time = time.time()
    print("Let's end training -------")

    print("Training {} timesteps takes about {} seconds".format(total_timesteps, end_time-begin_time))


def test(model):
    vec_env = model.get_env()
    obs = vec_env.reset()
    episode_num = 100
    episode_return = []

    for i in range(episode_num):
        episode_reward = 0
        time.sleep(0.01)
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
       # vec_env.render("human")
        while not done:
            action, _state = model.predict(obs, deterministic = True)
            obs, reward, done, info = vec_env.step(action)
            episode_reward += reward
        print(info)
        episode_return.append(episode_reward)
    print(episode_return)
    print("average return", sum(episode_return)/episode_num)


def test_random_agent(env, num_episodes=100):
    images = []
    episodes_returns = []

    for _ in range(num_episodes):
        obs = env.reset()
        episode_return = 0

        done = False
        images.append(env.render())
        while not done:
            action = env.action_space.sample() ## random sample action
            obs, reward, terminated, truncated, info = env.step(action)
            print(truncated, info)
            done = terminated or truncated
            episode_return += reward
        
        episodes_returns.append(episode_return)
    
    print("Random Agent with average return", sum(episodes_returns)/num_episodes)




if __name__ == "__main__":
    check_device()
    total_timesteps = int(1e5)
    env = gym.make('PandaReach-v3') #render_mode="human")
    policy_name = "TD3"
    
    #model = TD3(policy = "MultiInputPolicy", env = env, verbose = 0)
    #model = DDPG(policy = "MultiInputPolicy", env = env, verbose = 0)
    #model = SAC(policy = "MultiInputPolicy", env = env, verbose = 0)
    #train(model, total_timesteps=total_timesteps, policy_name = policy_name)

    model = TD3.load("{}_pandareach{}".format(policy_name, total_timesteps), env = env, print_system_info = False)
    test(model)
    
    #test_random_agent(env)
    
    # VecEnv resets automatically
    # if done:
    #     obs = vec_env.reset()



