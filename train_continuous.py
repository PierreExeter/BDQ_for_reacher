import gym
import time
import deepq

# Set environment and number of training episodes
env_name = 'Reacher-v1'
total_num_episodes = 50

# parameters
dueling = True # with dueling (best-performing)
agg_method = 'reduceLocalMean' # naive, reduceLocalMax, reduceLocalMean (best-performing)   
target_version = 'mean' # indep, max, mean (best-performing)
losses_version = 2 # 1,2 (best-performing),3,4,5 
num_actions_pad = 33 # numb discrete sub-actions per action dimension
independent = False # only set to True for trying training without the shared network module (does not work well)

env = gym.make(env_name)

time_stamp = time.strftime('%Y-%m-%d_%H-%M-%S') 

model = deepq.models.mlp_branching(
    hiddens_common=[512, 256], 
    hiddens_actions=[128],  
    hiddens_value=[128],
    independent=independent,
    num_action_branches=env.action_space.shape[0],
    dueling=dueling,
    aggregator=agg_method  
)

act = deepq.learn_continuous_tasks(
    env,
    q_func=model,
    env_name=env_name, 
    # dir_path=os.path.abspath(path_logs),
    time_stamp=time_stamp,
    total_num_episodes=total_num_episodes,
    lr=1e-4,
    gamma=0.99,
    batch_size=64,
    buffer_size=int(1e6),
    prioritized_replay=True,
    prioritized_replay_alpha=0.6,
    prioritized_replay_beta0=0.4,
    prioritized_replay_beta_iters=2e6,  
    dueling=dueling,
    independent=independent,
    target_version=target_version,
    losses_version=losses_version,
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
    callback=None 
)


print('Saving model...')
act.save('trained_models/{}_{}.pkl'.format(time_stamp, env_name))

