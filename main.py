import argparse
import random

import gymnasium as gym
import numpy as np
import torch
import os
import sac
from agents import train_agent
from utils import ReplayBuffer
from actor_critic import ActorCriticAgent
from rollouts import evaluate, evaluate_agent
from policy_gradient import simulate_policy_pg
from networks import PGPolicy, PGBaseline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device', device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pg',
                        help='task: pg (or policy_gradient), actor_critic (or ac), sac')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--env', type=str, default="pendulum", help='choose environment, pendulum or ant')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()
    args.task = {'policy_gradient': 'pg', 'ac': 'actor_critic'}.get(args.task, args.task)
    if args.render:
        os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libGLEW.so"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.env == 'pendulum':
        env = gym.make("InvertedPendulum-v4", render_mode='human' if args.render else None)
        max_episode_steps = 200
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    elif args.env == 'ant':
        max_episode_steps = 500
        env = gym.make("Ant-v4", render_mode='human' if args.render else None)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    else:
        raise ValueError('Invalid environment')

    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    obs_size = env.observation_space.shape[0]
    ac_size = env.action_space.shape[0]
    action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max())]

    if args.task == 'pg':
        # Define policy and value function
        hidden_dim_pol = 64
        hidden_depth_pol = 2
        hidden_dim_baseline = 64
        hidden_depth_baseline = 2
        policy = PGPolicy(obs_size, ac_size, hidden_dim=hidden_dim_pol,
                          hidden_depth=hidden_depth_pol)
        baseline = PGBaseline(obs_size, hidden_dim=hidden_dim_baseline,
                              hidden_depth=hidden_depth_baseline)
        policy.to(device)
        baseline.to(device)

        # Training hyperparameters
        num_epochs = 100
        batch_size = 64
        gamma = 0.99
        baseline_train_batch_size = 64
        baseline_num_epochs = 5
        print_freq = 10

        if not args.test:
            # Train policy gradient
            simulate_policy_pg(env, policy, baseline, num_epochs=num_epochs, batch_size=batch_size,
                               gamma=gamma, baseline_train_batch_size=baseline_train_batch_size, device=device,
                               baseline_num_epochs=baseline_num_epochs, print_freq=print_freq, render=args.render)
            torch.save(policy.state_dict(), f'pg_{args.env}_final.pth')
        else:
            print('loading pretrained pg')
            policy.load_state_dict(torch.load(f'pg_{args.env}_final.pth'))
        evaluate(env, policy, num_validation_runs=100, render=args.render)
    else:
        num_train_steps = 30_000
        num_seed_steps = 5000
        eval_frequency = 10_000
        num_eval_episodes = 10
        replay_buffer = ReplayBuffer(obs_size, ac_size, num_train_steps, device)
        hidden_dim = 256
        hidden_depth = 2
        batch_size = 256
        discount_factor = 0.99
        if args.task == 'actor_critic':
            agent = ActorCriticAgent(
                obs_dim=obs_size,
                action_dim=ac_size,
                action_range=action_range,
                device=device,
                discount=discount_factor,
                actor_lr=3e-4,
                critic_lr=3e-4,
                critic_tau=5e-3,
                batch_size=batch_size,
                hidden_dim=hidden_dim,
                hidden_depth=hidden_depth,
                double_critic=False,
                temperature=False
            )

        elif args.task == "sac":
            agent = sac.SACAgent(
                obs_dim=obs_size,
                action_dim=ac_size,
                action_range=action_range,
                device=device,
                discount=discount_factor,
                init_temperature=0.1,
                alpha_lr=3e-4,
                actor_lr=3e-4,
                critic_lr=3e-4,
                critic_tau=0.005,
                batch_size=batch_size,
                target_entropy=-ac_size,
                hidden_dim=hidden_dim,
                hidden_depth=hidden_depth,
                double_critic=True,
                temperature=True
            )
        else:
            raise ValueError('Invalid task')

        if not args.test:
            policy_loss, critic_loss, batch_reward = (
                train_agent(agent,
                            env,
                            num_train_steps=num_train_steps,
                            num_seed_steps=num_seed_steps,
                            eval_frequency=eval_frequency,
                            num_eval_episodes=num_eval_episodes,
                            replay_buffer=replay_buffer))

            agent.save(f'{args.task}_{args.env}_final.pth')
        else:
            print('loading pretrained', args.task)
            agent.load(f'{args.task}_{args.env}_final.pth')

        # final evaluation
        evaluate_agent(env, agent, "final", num_episodes=100, verbose=True)
