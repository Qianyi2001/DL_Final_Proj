from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
import glob
from JEPA import Encoder, Predictor, JEPAFullModel

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "/scratch/DL24FA"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    return probe_train_ds, probe_val_ds


def load_model():
    """Load or initialize the model."""
    # 创建空的模型组件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dim = 128
    encoder_theta = Encoder(feature_dim=feature_dim).to(device)
    predictor = Predictor(feature_dim=feature_dim, action_dim=2).to(device)

    # 从文件加载参数
    # 确保 'model_weights.pth' 是您训练好后保存在同目录的权重文件
    checkpoint = torch.load('/home/qx690/model_weights.pth', map_location=device)
    encoder_theta.load_state_dict(checkpoint['encoder_theta'])
    predictor.load_state_dict(checkpoint['predictor'])

    # 创建JEPAFullModel实例
    model = JEPAFullModel(encoder_theta, predictor).to(device)
    return model

def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()

    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")


if __name__ == "__main__":
    device = get_device()
    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model()
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
