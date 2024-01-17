import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box

class mlp(nn.Module):
    def __init__(self, obs_dim, hidden_size, n_acts):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self.n_acts = n_acts

        self.activation = nn.Tanh
        self.output_activation = nn.Identity

        self.layer = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_size),
            self.activation(),
            nn.Linear(hidden_size, n_acts),
            self.output_activation()
        )

    def forward(self, x):
        output = self.layer(x)
        return output

def reward_to_go(rewards):
    n = len(rewards)
    rtgs = np.zeros_like(rewards)
    for i in reversed(range(n)):
        rtgs[i] = rewards[i] + (rtgs[i+1] if i+1<n else 0)
    return rtgs

def train(env_name='CartPole-v0', hidden_size=32, lr=1e-2,
          epochs=50, batch_size=5000, render=False):
    
    # env = gym.make(env_name, render_mode=render)
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    logits_policy = mlp(obs_dim, hidden_size, n_acts)

    def get_policy(obs):
        logits = logits_policy(obs)
        return Categorical(logits=logits)

    def get_action(obs):
        return get_policy(obs).sample().item()
    
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()
    
    optimizer = Adam(logits_policy.parameters(), lr=lr)

    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_returns = []
        batch_lens = []

        # S0
        obs = env.reset()
        obs = obs[0]
        done = False
        ep_rewards = []
        
        finished_rendering_this_epoch = False

        while True:

            if (not finished_rendering_this_epoch) and render:
                env.render()
            
            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, reward, done, _, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rewards.append(reward)

            if done:
                ep_return, ep_len = sum(ep_rewards), len(ep_rewards)
                batch_returns.append(ep_return)
                batch_lens.append(ep_len)

                batch_weights += list(reward_to_go(ep_rewards))

                obs, done, ep_rewards = env.reset()[0], False, []

                finished_rendering_this_epoch = True

                if len(batch_obs) > batch_size:
                    break

        optimizer.zero_grad()
        batch_loss = compute_loss(
            obs=torch.as_tensor(batch_obs, dtype=torch.float32),
            act=torch.as_tensor(batch_acts, dtype=torch.float32),
            weights=torch.as_tensor(batch_weights, dtype=torch.float32)
        )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_returns, batch_lens
    
    for i in range(epochs):
        batch_loss, batch_returns, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_returns), np.mean(batch_lens)))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing Simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, epochs=50)

    # print(reward_to_go([1, 2, 3, 4, 5]))
