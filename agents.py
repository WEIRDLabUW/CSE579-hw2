import abc

import gymnasium as gym
import numpy as np
import torch
from networks import SingleQCritic, DoubleQCritic, DiagGaussianActor
import utils
from rollouts import evaluate_agent
from utils import eval_mode


class GenericACAgent:
    """SAC algorithm."""

    def __init__(self, obs_dim, action_dim, action_range, device,
                 discount,
                 actor_lr, critic_lr,
                 critic_tau, batch_size,
                 init_temperature=None, alpha_lr=None, target_entropy=None, double_critic=False, temperature=False,
                 hidden_dim=1024, hidden_depth=2):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.batch_size = batch_size
        self.double_critic = double_critic
        self.temperature = temperature

        if double_critic:
            self.critic = DoubleQCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_depth=hidden_depth,
                                        hidden_dim=hidden_dim).to(self.device)
            self.critic_target = DoubleQCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_depth=hidden_depth,
                                               hidden_dim=hidden_dim).to(self.device)
        else:
            self.critic = SingleQCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_depth=hidden_depth,
                                        hidden_dim=hidden_dim).to(self.device)
            self.critic_target = SingleQCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_depth=hidden_depth,
                                               hidden_dim=hidden_dim).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = DiagGaussianActor(obs_dim=obs_dim, action_dim=action_dim, hidden_depth=hidden_depth,
                                       hidden_dim=hidden_dim, log_std_bounds=[-5, 2]).to(self.device)

        if temperature:
            self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
            self.log_alpha.requires_grad = True

            self.target_entropy = target_entropy
            self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        if self.temperature:
            return self.log_alpha.exp()
        else:
            return None

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1

        # convert the action to numpy
        def to_np(t):
            if t is None:
                return None
            elif t.nelement() == 0:
                return np.array([])
            else:
                return t.cpu().detach().numpy()

        return to_np(action[0])

    def update(self, replay_buffer):
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
            self.batch_size)

        train_batch_reward = reward.mean().item()

        critic_loss = self.update_critic(obs, action, reward, next_obs, not_done_no_max)

        actor_loss, actor_entropy, alpha_loss = self.update_actor(obs)

        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_tau)

        return train_batch_reward, critic_loss, actor_loss, actor_entropy, alpha_loss

    @abc.abstractmethod
    def update_actor(self, obs):
        pass

    @abc.abstractmethod
    def update_critic(self, obs, action, reward, next_obs, not_done_no_max):
        pass

    def save(self, save_path):
        state = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }

        if self.temperature:
            state['log_alpha'] = self.log_alpha
            state['log_alpha_optimizer'] = self.log_alpha_optimizer.state_dict()

        torch.save(state, save_path)

    def load(self, load_path):
        state = torch.load(load_path, map_location=self.device)
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])

        if self.temperature:
            self.log_alpha = state['log_alpha']
            self.log_alpha_optimizer.load_state_dict(state['log_alpha_optimizer'])


def train_agent(agent, env, num_train_steps, num_seed_steps, eval_frequency, num_eval_episodes, replay_buffer):
    """
    Generic training loop for an agent. It runs num_seed_steps of random exploration and then does the training loop.
    In the training loop, it samples an action, takes an environment step, and then updates the agent with a sampled batch
    from the replay buffer. It also evaluates the agent periodically.
    """
    episode, episode_reward, done = 0, 0, True
    step = 0
    since_last_eval = 0
    actor_loss = []
    critic_loss = []
    batch_reward = []
    while step < num_train_steps:
        if done:
            # evaluate agent periodically
            if since_last_eval > eval_frequency:
                # Note the step here will fluctuate as it waits until terminaton to evaluate, but this is ok.
                evaluate_agent(env, agent, step, num_episodes=num_eval_episodes)
                since_last_eval = 0

            obs = env.reset()[0] if isinstance(env, gym.Env) else env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
        # sample action for data collection
        if step < num_seed_steps:
            action = env.action_space.sample()
        else:
            with eval_mode(agent):
                action = agent.act(obs, sample=True)
            since_last_eval += 1

        # run training update
        if step >= num_seed_steps:
            result = agent.update(replay_buffer)
            # add the policy and critic loss to the list
            actor_loss.append(result[2])
            critic_loss.append(result[1])
            batch_reward.append(result[0])
            if result is not None:
                # tuple is train_batch_reward, critic_loss, actor_loss, actor_entropy, alpha_loss
                if step % 5000 == 0:
                    # round these tensors to 4 decimal places
                    result = tuple(map(lambda x: round(x, 4), result))
                    print(
                        f"step {step}, batch_r {result[0]}, critic_l {result[1]}, actor_l {result[2]}"
                        f", actor_ent {result[3]}, alpha_l {result[4]}")

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # allow infinite bootstrap
        done = float(done)
        done_no_max = 0 if episode_step + 1 == env.spec.max_episode_steps else done
        episode_reward += reward

        replay_buffer.add(obs, action, reward, next_obs, done,
                          done_no_max)

        obs = next_obs
        episode_step += 1
        step += 1
    return actor_loss, critic_loss, batch_reward
