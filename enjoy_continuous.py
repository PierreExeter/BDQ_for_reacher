import gym
import numpy as np
import time
import deepq

env_name = 'Reacher-v1'
env = gym.make(env_name)
num_actions_pad = 33 # number of subaction per joints. Ensure it's set correctly to the value used during training   

model_file_name = '2019-09-06_10-37-46_Reacher-v1.pkl'
act = deepq.load("trained_models/{}".format(model_file_name))

num_action_dims = env.action_space.shape[0] 
num_action_streams = num_action_dims
num_actions = num_actions_pad*num_action_streams
low = env.action_space.low 
high = env.action_space.high 
actions_range = np.subtract(high, low) 

total_rewards = 0
for i in range(100):
    obs, done = env.reset(), False
    episode_rew = 0
    while not done:
        env.render()
        time.sleep(0.02)
        
        action_idx = np.array(act(np.array(obs)[None], stochastic=False))
        action = action_idx / (num_actions_pad-1) * actions_range - high

        obs, rew, done, _ = env.step(action)
        episode_rew += rew
    print('Episode reward', episode_rew)
    total_rewards += episode_rew

print('Mean episode reward: {}'.format(total_rewards/100))

