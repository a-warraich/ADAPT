import torch
from torch import nn
import torch.nn.functional as F
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        hidden_size = 256
        
        self.dense_1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.dense_2 = nn.Linear(hidden_size, hidden_size)
        self.dense_3 = nn.Linear(hidden_size, 1)
        
        self.dense_4 = nn.Linear(state_dim + action_dim, hidden_size)
        self.dense_5 = nn.Linear(hidden_size, hidden_size)
        self.dense_6 = nn.Linear(hidden_size, 1)

    def forward(self, x, x_actions):
        x = torch.cat([x, x_actions], dim=1)

        q1 = F.relu(self.dense_1(x))
        q1 = F.relu(self.dense_2(q1))
        q1 = self.dense_3(q1)

        q2 = F.relu(self.dense_4(x))
        q2 = F.relu(self.dense_5(q2))
        q2 = self.dense_6(q2)

        return q1, q2

    def forward_q1(self, x, x_actions):
        x = torch.cat([x, x_actions], dim=1)
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = self.dense_3(x)
        return x