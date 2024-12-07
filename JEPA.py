import torch
from torch import nn
import torch.nn.functional as F


class JEPAEncoder(nn.Module):
    def __init__(self, repr_dim=256):
        super().__init__()
        # 简单CNN编码器示例，可根据需要加深网络
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, repr_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        return h


class JEPAPredictor(nn.Module):
    def __init__(self, input_dim=256 + 2, repr_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, repr_dim)
        )

    def forward(self, s, u):
        # s: [B, D], u: [B, 2]
        inp = torch.cat([s, u], dim=-1)
        return self.fc(inp)


class JEPAModel(nn.Module):
    def __init__(self, repr_dim=256, device="cuda"):
        super().__init__()
        self.repr_dim = repr_dim
        self.encoder = JEPAEncoder(repr_dim=repr_dim)
        self.predictor = JEPAPredictor(input_dim=repr_dim + 2, repr_dim=repr_dim)
        self.device = device

    def forward(self, states, actions):
        # states: [B, 1, 2, 64, 64]
        # actions: [B, T-1, 2]
        B = states.size(0)
        T_minus_1 = actions.size(1)
        # Encode initial state
        s_0 = self.encoder(states[:, 0])  # [B, D]

        # Predict future states
        # We'll store s_0 and predicted s_1,...,s_{T-1} in a tensor
        pred_states = [s_0]
        s_prev = s_0
        for t in range(T_minus_1):
            u_t = actions[:, t]  # [B,2]
            s_next = self.predictor(s_prev, u_t)
            pred_states.append(s_next)
            s_prev = s_next

        pred_encs = torch.stack(pred_states, dim=1)  # [B, T, D]
        return pred_encs
