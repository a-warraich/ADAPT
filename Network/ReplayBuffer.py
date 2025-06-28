import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, max_len, state_shape, action_dim, device):
        self.device = device
        
        self.max_len = max_len
        
        self.state_buffer = torch.zeros((max_len, *state_shape), dtype=torch.float32).to(device)
        self.action_buffer = torch.zeros((max_len, action_dim), dtype=torch.float32).to(device)
        self.reward_buffer = torch.zeros(max_len, dtype=torch.float32).to(device)
        self.next_state_buffer = torch.zeros((max_len, *state_shape), dtype=torch.float32).to(device)

        self.done_buffer = torch.zeros(max_len, dtype=torch.float32).to(device)
        
        self.ptr = 0
        
        self.size = 0

    def __len__(self):
        return self.size

    def append(self, state, action, reward, next_state, done):
        self.state_buffer[self.ptr] = torch.tensor(state).to(self.device)
        self.action_buffer[self.ptr] = torch.tensor(action).to(self.device)
        self.reward_buffer[self.ptr] = torch.tensor(reward).to(self.device)
        self.next_state_buffer[self.ptr] = torch.tensor(next_state).to(self.device)
        self.done_buffer[self.ptr] = torch.tensor(done).to(self.device)

        self.ptr = (self.ptr + 1) % self.max_len
        self.size = min(self.size + 1, self.max_len)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        states = self.state_buffer[indices]
        actions = self.action_buffer[indices]
        rewards = self.reward_buffer[indices]
        next_states = self.next_state_buffer[indices]
        dones = self.done_buffer[indices]
        return states, actions, rewards, next_states, dones
