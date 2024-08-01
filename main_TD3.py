import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pickle
import gym
import numpy as np
import torch
from datetime import datetime
import time
import argparse
from envs.hopper_sparse import SparseHopperEnv
from envs.ant_sparse import SparseAntEnv
from envs.half_cheetah_sparse import SparseHalfCheetahEnv
from envs.walker_2d_sparse import SparseWalker2dEnv

import utils
import dateutil.tz

from algorithm import TD3_GILD
from algorithm import TD3_GILD_ws


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def evaluate_policy(policy, eval_env, eval_idx, total_step, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes

    print ("---------------------------------------")
    print ("In Evaluation %d, Toal training step %d, avg_eval_reward over %d episodes: %f" % (eval_idx, total_step, eval_episodes, avg_reward))
    print ("---------------------------------------")
    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default='Hopper-v2')
    parser.add_argument('--sparse_env', default=1, type=int)  # sparse environment . When 1(default), sparse; Otherwise, dense.
    parser.add_argument("--method", default="TD3_GILD_ws")  # Policy name
    parser.add_argument("--seed", default=6, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument('--actor_lr', type=float, default=3e-4)  # learning rate
    parser.add_argument('--critic_lr', type=float, default=3e-4)  # learning rate
    parser.add_argument('--gild_lr', type=float, default=1e-4)  # learning rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--warm_start_timesteps", default=1e4, type=float)  # warm start steps for GILD
    parser.add_argument("--save_models", default=True, type=bool)  # Whether or not models are saved
    parser.add_argument("--save_freq", default=5e5, type=float)  # How often (time steps) we save models
    args = parser.parse_args()


    # Create directory
    now = datetime.now(dateutil.tz.tzlocal())
    time_dir = now.strftime('%Y_%m_%d_%H_%M_%S')
    if args.sparse_env == 1:
        env_path = "%s(sparse)" % (args.env_name)
    else:
        env_path = "%s(dense)" % (args.env_name)

    time_dir = ("%s/%s" % (env_path, time_dir))

    if not os.path.exists('Results/%s/%s/' % (args.method, time_dir)):
        os.makedirs('Results/%s/%s/' % (args.method, time_dir))
        os.makedirs('Results/%s/%s/evaluation/' % (args.method, time_dir))
        os.makedirs('Results/%s/%s/trained_models/' % (args.method, time_dir))
    flags_log = os.path.join('Results/%s/%s/' % (args.method, time_dir), 'log.txt')
    save_path = 'Results/{}/{}'.format(args.method,time_dir)

    localtime = time.asctime(time.localtime(time.time()))
    utils.write_log("localtime:", localtime, flags_log)

    # # Build environment
    if args.sparse_env == 1:
        print(f"Running on sparse reward Env")
        utils.write_log("Running on sparse reward Env ", args.env_name, flags_log)
        if args.env_name == "Hopper-v2":
            args.seed = 6
            args.sparse_val = 1.
            env = SparseHopperEnv(args.sparse_val)
            eval_env = gym.make(args.env_name)

        elif args.env_name == 'Walker2d-v2':
            args.seed = 0
            args.sparse_val = 1.
            env = SparseWalker2dEnv(args.sparse_val)
            eval_env = gym.make(args.env_name)

        elif args.env_name == 'HalfCheetah-v2':
            args.seed = 0
            args.sparse_val = 2.
            env = SparseHalfCheetahEnv(args.sparse_val)
            eval_env = gym.make(args.env_name)

        elif args.env_name == 'Ant-v2':
            args.seed = 45
            args.sparse_val = 1.
            env = SparseAntEnv(args.sparse_val)
            eval_env = gym.make(args.env_name)
            
    else:
        utils.write_log("Running on custom dense rewardEnv ", args.env_name, flags_log)
        print('Running on custom dense reward Env', args.env_name)
        env = gym.make(args.env_name)
        args.data_path = 'Traj_Data/%s_data.p' % (args.env_name)
        eval_env = gym.make(args.env_name)
    
    # Set seeds
    env.seed(args.seed)
    eval_env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # load demonstration data
    il_method = 'TD3'
    args.data_path = 'Traj_Data/Traj_Behavior_%s_%s.p' % (args.env_name, il_method)
    demo_traj = pickle.load(open(args.data_path, "rb"))

    # Initialize policy
    if 'ws' in args.method:
        policy = TD3_GILD_ws.TD3_GILD(state_dim, action_dim, max_action, demo_traj, args)
    else:
        policy = TD3_GILD.TD3_GILD(state_dim, action_dim, max_action, demo_traj, args)
        

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # # Store the parameter
    utils.write_log("args:", args, flags_log)

    # Evaluate untrained policy
    eval_idx = 0
    avg_reward = evaluate_policy(policy, eval_env, eval_idx, 0)
    eval_idx += 1
    evaluation_rewards = [avg_reward]
    utils.write_log("------------------------------------------------------------------------------","", flags_log)
    utils.write_log("", "After episode %d, Total step %d, Average evaluation reward over 10 episodes: %f" % (
        0, 0, avg_reward), flags_log)
    utils.write_log("------------------------------------------------------------------------------","", flags_log)

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    max_episode_steps = eval_env._max_episode_steps

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action) # select action with noise

        # Perform action
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done: 
            utils.write_log("", "Total T %d Episode Num %d Episode T %d Eposide reward %f" % (
                t, episode_num, episode_timesteps, episode_reward), flags_log)
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t) % args.eval_freq == 0:
            avg_reward = evaluate_policy(policy, eval_env, eval_idx, t)
            eval_idx += 1
            evaluation_rewards.append(avg_reward)
            np.save('%s/evaluation/evaluation_reward.npy' % (save_path), evaluation_rewards)
            utils.write_log("------------------------------------------------------------------------------","", flags_log)
            utils.write_log("", "After episode %d, Total step %d, Average evaluation reward over 10 episodes: %f" % (
                episode_num, t, avg_reward), flags_log)
            utils.write_log("------------------------------------------------------------------------------","", flags_log)

        
        # Save period models
        if args.save_models:
            if (t) % args.save_freq == 0:
                model_period = 'step%d_epi%d' %(t, episode_num)
                if not os.path.exists('%s/trained_models/%s/' % (save_path, model_period)):
                    os.makedirs('%s/trained_models/%s/' % (save_path, model_period))
                policy.save(save_path, model_period)


    # Final evaluation
    avg_reward = evaluate_policy(policy, eval_env, eval_idx, t)
    eval_idx += 1
    evaluation_rewards.append(avg_reward)
    utils.write_log("------------------------------------------------------------------------------","", flags_log)
    utils.write_log("", "After episode %d, Total step %d, Average evaluation reward over 10 episodes: %f" % (
        episode_num, t, avg_reward), flags_log)
    utils.write_log("------------------------------------------------------------------------------","", flags_log)

    np.save('%s/evaluation/evaluation_reward.npy' % (save_path), evaluation_rewards)

    # Save final models
    if args.save_models:
        model_period = 'step%d_epi%d' % (t, episode_num)
        if not os.path.exists('%s/trained_models/%s/' % (save_path, model_period)):
            os.makedirs('%s/trained_models/%s/' % (save_path, model_period))
        policy.save(save_path, model_period)

    utils.plot_results(save_path)

    env.close()