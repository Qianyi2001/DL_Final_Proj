import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from JEPA import Encoder, Predictor, vicreg_loss
from dataset import create_wall_dataloader


def train_jepa(encoder_theta, encoder_psi, predictor, dataloader, optimizer, device='cuda', epochs=1):
    encoder_theta.train()
    encoder_psi.train()
    predictor.train()

    ema_decay = 0.99
    best_loss = float('inf')  # 初始化最优损失
    for epoch in range(epochs):
        total_loss = 0  # 用于累计当前epoch的总损失
        num_steps = len(dataloader)  # 总步数

        for i, batch in enumerate(dataloader):
            states = batch.states.to(device)
            actions = batch.actions.to(device)

            if i % 100 == 0:  # 每100个batch打印一次
                print(f"Epoch {epoch + 1}/{epochs}, Step {i + 1}/{num_steps}")
                print("Actions mean:", actions.mean().item(), "std:", actions.std().item())

            N, T, C, H, W = states.shape

            s0 = encoder_theta(states[:, 0])
            target_reps = []
            with torch.no_grad():
                for t in range(T):
                    tr = encoder_psi(states[:, t])
                    tr = torch.nn.functional.layer_norm(tr, [tr.size(-1)])  # LayerNorm处理
                    target_reps.append(tr)
                target_reps = torch.stack(target_reps, dim=1)

            pred_s = [s0]
            for t in range(1, T):
                u_t_1 = actions[:, t - 1]
                prev_s = pred_s[-1]
                next_s_pred = predictor(prev_s, u_t_1)
                pred_s.append(next_s_pred)
            pred_s = torch.stack(pred_s, dim=1)

            loss = vicreg_loss(
                pred_s[:, 1:].reshape(-1, pred_s.size(-1)),
                target_reps[:, 1:].reshape(-1, target_reps.size(-1)),
                sim_weight=50.0,
                var_weight=10.0,
                cov_weight=0.5
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 使用EMA更新encoder_psi
            with torch.no_grad():
                for p_theta, p_psi in zip(encoder_theta.parameters(), encoder_psi.parameters()):
                    p_psi.data = ema_decay * p_psi.data + (1 - ema_decay) * p_theta.data

            total_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Step {i + 1}/{num_steps}, Loss: {loss.item():.4f}")
                print(f"pred_s mean: {pred_s.mean().item():.4f}, std: {pred_s.std().item():.4f}")
                print(f"target_reps mean: {target_reps.mean().item():.4f}, std: {target_reps.std().item():.4f}")

        avg_loss = total_loss / num_steps  # 当前 epoch 的平均损失
        print(f"Epoch {epoch + 1}/{epochs} complete. Average Loss: {avg_loss:.4f}")

        # 如果当前模型是最优的，则保存为最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = "/home/qx690/best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'encoder_theta': encoder_theta.state_dict(),
                'encoder_psi': encoder_psi.state_dict(),
                'predictor': predictor.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': best_loss,
            }, best_model_path)
            print(f"Best model updated with loss {best_loss:.4f} at {best_model_path}")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feature_dim = 128
    encoder_theta = Encoder(feature_dim=feature_dim).to(device)
    encoder_psi = Encoder(feature_dim=feature_dim).to(device)
    predictor = Predictor(feature_dim=feature_dim, action_dim=2).to(device)

    with torch.no_grad():
        for p_psi, p_theta in zip(encoder_psi.parameters(), encoder_theta.parameters()):
            p_psi.data = p_theta.data.clone()

    optimizer = torch.optim.Adam(
        list(encoder_theta.parameters()) + list(predictor.parameters()),
        lr=1e-3
    )

    dataloader = create_wall_dataloader(
        data_path="/scratch/DL24FA/train",
        probing=False,
        device=device,
        batch_size=128,
        train=True,
    )

    train_jepa(encoder_theta, encoder_psi, predictor, dataloader, optimizer, device, epochs=10)
