import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),  # 64->32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 32->16
            nn.ReLU(),
            nn.Conv2d(64,128, 3, stride=2, padding=1), # 16->8
            nn.ReLU(),
        )
        self.fc = nn.Linear(128*8*8, feature_dim)

    def forward(self, x):
        # x: (B, 2, 64, 64)
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        return h


class Predictor(nn.Module):
    def __init__(self, feature_dim=128, action_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim+action_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, s, u):
        # s: (B, feature_dim)
        # u: (B, 2)
        x = torch.cat([s, u], dim=-1)
        return self.fc(x)


def vicreg_loss(x, y, sim_weight=25.0, var_weight=25.0, cov_weight=1.0, eps=1e-4):
    """
    x, y: (B, D)
    使用VICReg损失来防止表示坍缩
    """
    # invariance
    inv_loss = F.mse_loss(x, y)

    # variance
    x_var = torch.sqrt(x.var(dim=0) + eps)
    y_var = torch.sqrt(y.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(1 - x_var)) + torch.mean(F.relu(1 - y_var))

    # covariance
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    cov_x = (x.T @ x) / (x.shape[0] - 1)
    cov_y = (y.T @ y) / (y.shape[0] - 1)

    off_diag_x = cov_x.flatten()[1::cov_x.size(0)+1]
    off_diag_y = cov_y.flatten()[1::cov_y.size(0)+1]
    cov_loss = (off_diag_x**2).mean() + (off_diag_y**2).mean()

    loss = sim_weight * inv_loss + var_weight * var_loss + cov_weight * cov_loss
    return loss


class JEPAFullModel(nn.Module):
    """
    整合 JEPA 模型，用于在评估阶段使用。
    接受 states 和 actions，返回预测的表示序列 (B, T, D)。
    """
    def __init__(self, encoder_theta, predictor):
        super().__init__()
        self.encoder_theta = encoder_theta
        self.predictor = predictor
        self.repr_dim = encoder_theta.fc.out_features  # encoder最终输出维度

    @torch.no_grad()
    def forward(self, states, actions):
        # states: (B, T, 2, 64,64)
        # actions: (B, T-1, 2)
        # 返回pred_s: (B, T, D)
        self.eval()  # 确保评估模式
        B, T, C, H, W = states.shape
        s0 = self.encoder_theta(states[:,0]) # (B, D)
        pred_s = [s0]
        for t in range(1, T):
            u_t_1 = actions[:, t-1] # (B, 2)
            prev_s = pred_s[-1]
            next_s = self.predictor(prev_s, u_t_1)
            pred_s.append(next_s)
        pred_s = torch.stack(pred_s, dim=1) # (B, T, D)
        return pred_s
