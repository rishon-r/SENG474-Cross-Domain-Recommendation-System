"""Mapping networks g: R^k -> R^k (Section 2 'Mappings' of the paper; trained on L_g, Eq. 3)."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """g(x) = W2 relu(W1 x + b1) + b2"""

    def __init__(self, k, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k, hidden), nn.ReLU(), nn.Linear(hidden, k))

    def forward(self, x):
        return self.net(x)


class MetaNet(nn.Module):
    """m_phi: R^k -> R^{k x k}; g_specific(x) = m_phi(x) x  (user-specific linear bridge)."""

    def __init__(self, k, hidden):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(nn.Linear(k, hidden), nn.ReLU(), nn.Linear(hidden, k * k))

    def forward(self, x):
        W = self.net(x).view(-1, self.k, self.k)
        return torch.bmm(W, x.unsqueeze(-1)).squeeze(-1)
