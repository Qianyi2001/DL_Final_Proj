import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from JEPA import Encoder, Predictor, vicreg_loss
from dataset import create_wall_dataloader

def train_jepa(encoder_theta, encoder_psi, predictor, dataloader,
               optimizer, device='cuda', epochs=1):
    encoder_theta.train()
    encoder_psi.train()
    predictor.train()

    for epoch in range(epochs):
        for batch in dataloader:
            states = batch.states  # (N, T, 2, 64, 64)
            actions = batch.actions # (N, T-1, 2)
            states = states.to(device)
            actions = actions.to(device)

            N, T, C, H, W = states.shape
            s0 = encoder_theta(states[:,0])

            with torch.no_grad():
                target_reps = []
                for t in range(T):
                    target_reps.append(encoder_psi(states[:,t]))
                target_reps = torch.stack(target_reps, dim=1) # (N, T, feature_dim)

            pred_s = [s0]
            for t in range(1, T):
                u_t_1 = actions[:, t-1]
                prev_s = pred_s[-1]
                next_s_pred = predictor(prev_s, u_t_1)
                pred_s.append(next_s_pred)
            pred_s = torch.stack(pred_s, dim=1) # (N, T, feature_dim)

            loss = vicreg_loss(
                pred_s[:,1:].reshape(-1, pred_s.size(-1)),
                target_reps[:,1:].reshape(-1, target_reps.size(-1))
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 如果需要EMA更新，可在此处添加EMA逻辑

        print(f"Epoch {epoch} complete. Loss: {loss.item()}")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feature_dim = 128
    encoder_theta = Encoder(feature_dim=feature_dim).to(device)
    encoder_psi = Encoder(feature_dim=feature_dim).to(device)
    predictor = Predictor(feature_dim=feature_dim, action_dim=2).to(device)

    with torch.no_grad():
        for p_psi, p_theta in zip(encoder_psi.parameters(), encoder_theta.parameters()):
            p_psi.data = p_theta.data.clone()

    optimizer = torch.optim.Adam(list(encoder_theta.parameters()) +
                                 list(predictor.parameters()),
                                 lr=1e-3)

    # 示例：用户应在外部通过 create_wall_dataloader 创建 dataloader
    dataloader = create_wall_dataloader(
        data_path="/scratch/DL24FA/train",
        probing=False,
        device=device,
        batch_size=32,
        train=True,
    )

    train_jepa(encoder_theta, encoder_psi, predictor, dataloader, optimizer, device, epochs=10)
    torch.save({
        'encoder_theta': encoder_theta.state_dict(),
        'encoder_psi': encoder_psi.state_dict(),
        'predictor': predictor.state_dict()
    }, 'model_weights.pth')
