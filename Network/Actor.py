from torch import nn
import torch
import torch.nn.functional as F
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        hidden_size = 256
        self.dense_1 = nn.Linear(state_dim, hidden_size)
        self.dense_2 = nn.Linear(hidden_size, hidden_size)
        self.dense_3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = torch.tanh(self.dense_3(x)) 
        return x