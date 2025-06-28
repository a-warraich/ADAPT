import importlib
import Network.Actor
importlib.reload(Network.Actor)

import torch
from torch import nn
import torch.nn.functional as F
from Network.Actor import Actor
from Network.Critic import Critic
from Network.ReplayBuffer import ReplayBuffer

class TD3Agent:
    def __init__(self, state_dim, action_dim, action_low, action_high, device, state_shape=None):
        self.device = device
        self.action_dim = action_dim
        self.gamma = 0.97
        self.tau = 0.005
        self.policy_noise = 0.3
        self.noise_clip = 0.5
        self.policy_freq = 4
        self.total_it = 0

        self.action_low = torch.tensor(action_low, dtype=torch.float32).to(device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32).to(device)

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=5e-05)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=5e-05)

        if state_shape is None:
            state_shape = (state_dim,)
        self.replay_buffer = ReplayBuffer(
            max_len=1_000_000,
            state_shape=state_shape,
            action_dim=action_dim,
            device=device
        )

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        raw_action = self.actor(state).cpu().detach().numpy().flatten()

        scaled_action = self.action_low.cpu().numpy() + 0.5 * (raw_action + 1.0) * (self.action_high.cpu().numpy() - self.action_low.cpu().numpy())
        return scaled_action

    def train(self, batch_size=100):
        if len(self.replay_buffer) < batch_size:
            return

        self.total_it += 1

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = self.actor_target(next_states) + noise

            next_actions = torch.max(torch.min(next_actions, self.action_high), self.action_low)

            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * target_q

        current_q1, current_q2 = self.critic(states, actions)

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.forward_q1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
