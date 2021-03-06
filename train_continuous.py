import gym
import time
import deepq

# Set environment and number of training episodes
env_name = 'Reacher-v1'
env = gym.make(env_name)
total_num_episodes = 20000
num_actions_pad = 33 # numb discrete sub-actions per action dimension

time_stamp = time.strftime('%Y-%m-%d_%H-%M-%S') 

model = deepq.models.mlp_branching(
    hiddens_common=[512, 256], 
    hiddens_actions=[128],  
    hiddens_value=[128],
    num_action_branches=env.action_space.shape[0]
)

act = deepq.learn_continuous_tasks(
    env,
    q_func=model,
    env_name=env_name, 
    time_stamp=time_stamp,
    total_num_episodes=total_num_episodes,
    lr=1e-4,
    gamma=0.99,
    batch_size=64,
    buffer_size=int(1e6),
    prioritized_replay_alpha=0.6,
    prioritized_replay_beta0=0.4,
    prioritized_replay_beta_iters=2e6,  
    num_actions_pad=num_actions_pad,
    grad_norm_clipping=10,
    learning_starts=1000, 
    target_network_update_freq=1000, 
    train_freq=1, 
    initial_std=0.2,
    final_std=0.2,
    timesteps_std=1e8,
    eval_freq=50,
    n_eval_episodes=30, 
    eval_std=0.0,
    num_cpu=16,
    print_freq=10, 
    callback=None)

print('Saving model...')
act.save('trained_models/{}_{}.pkl'.format(time_stamp, env_name))

