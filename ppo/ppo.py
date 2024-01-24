import torch
import torch.nn as nn
from network import FeedForwardNN
import numpy as np

class PPO:
    def __init__(self, env):
        # Initialize hyperparameters
        self._init_hyperparameters()

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # ALG STEP 1
        # Initialize actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        # Create the covariance matrix
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_pre_episode = 1600
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 0.005

    def get_action(self, obs):
        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(obs)
        mean = self.actor(obs)

        # Create our Multivariate Normal Distribution
        dist = torch.distributions.MultivariateNormal(mean, self.cov_mat)

        # Sample an action from distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0

            for rew in reversed(ep_rews):
                discounted_reward = rew + self.gamma * discounted_reward
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs
    
    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most 
        # recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = torch.distributions.MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        t = 0

        while t < self.timesteps_per_batch:
            # Rewards this episodes
            ep_rews = []

            obs = self.env.reset()[0]
            done = False

            for ep_t in range(self.max_timesteps_pre_episode):
                # Increment timesteps ran this batch so far
                t += 1

                # Collect observation
                batch_obs.append(obs)
            
                action, log_prob = self.get_action(obs)
                obs, rew, done, _, _ = self.env.step(action)
                
                
                # Collect reward, action, log_prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
            
            batch_lens.append(ep_t+1)
            batch_rews.append(ep_rews)
                
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
        
    def learn(self, total_timesteps):
        t_so_far = 0 # Timesteps simulated so far
        while t_so_far < total_timesteps:              # ALG STEP 2
            # Increment t_so_far somewhere below
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            t_so_far = np.sum(batch_lens)

            V, _ = self.evaluate(batch_obs, batch_acts)

            A_k = batch_rtgs - V.detach()

            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()


if __name__ == '__main__':
    import gym
    env = gym.make('Pendulum-v1')
    model = PPO(env)
    model.learn(100)
    print('done')