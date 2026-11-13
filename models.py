
import torch.nn as nn


class LinearProbe(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.head(x).squeeze(-1)


class MLPProbe(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        intermediate_dim = hidden_dim // 4

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, 1)
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)
