from typing import NamedTuple, List, Any, Optional, Dict
from itertools import chain
from dataclasses import dataclass
import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
from matplotlib import pyplot as plt

from schedulers import Scheduler, LRSchedule
from models import Prober, build_mlp
from configs import ConfigBase
from dataset import WallDataset
from normalizer import Normalizer


@dataclass
class ProbingConfig(ConfigBase):
    probe_targets: str = "locations"
    lr: float = 0.0002
    epochs: int = 20
    schedule: LRSchedule = LRSchedule.Cosine
    sample_timesteps: int = 30
    prober_arch: str = "256"


class ProbeResult(NamedTuple):
    model: torch.nn.Module
    average_eval_loss: float
    eval_losses_per_step: List[float]
    plots: List[Any]


default_config = ProbingConfig()


def location_losses(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert pred.shape == target.shape
    mse = (pred - target).pow(2).mean(dim=0)
    return mse


class ProbingEvaluator:
    def __init__(
        self,
        device: "cuda",
        model: torch.nn.Module,
        probe_train_ds,
        probe_val_ds: dict,
        config: ProbingConfig = default_config,
        quick_debug: bool = False,
    ):
        self.device = device
        self.config = config
        self.model = model
        self.model.eval()
        self.quick_debug = quick_debug
        self.ds = probe_train_ds
        self.val_ds = probe_val_ds
        self.normalizer = Normalizer()

    def train_pred_prober(self):
        """
        Probes whether the predicted embeddings capture the future locations.
        """
        repr_dim = self.model.repr_dim
        dataset = self.ds
        model = self.model

        config = self.config
        epochs = config.epochs

        if self.quick_debug:
            epochs = 1
        test_batch = next(iter(dataset))

        prober_output_shape = getattr(test_batch, "locations")[0, 0].shape
        prober = Prober(
            repr_dim,
            config.prober_arch,
            output_shape=prober_output_shape,
        ).to(self.device)

        all_parameters = list(prober.parameters())
        optimizer_pred_prober = torch.optim.Adam(all_parameters, config.lr)

        scheduler = Scheduler(
            schedule=self.config.schedule,
            base_lr=config.lr,
            data_loader=dataset,
            epochs=epochs,
            optimizer=optimizer_pred_prober,
            batch_steps=None,
            batch_size=dataset.batch_size,
        )

        step = 0

        for epoch in tqdm(range(epochs), desc=f"Probe prediction epochs"):
            for batch in tqdm(dataset, desc="Probe prediction step"):
                ################################################################################
                # Step 1: Use the JEPA model to predict embeddings for the entire sequence
                pred_encs = model(states=batch.states, actions=batch.actions)  # (B, T, D)
                pred_encs = pred_encs.transpose(0, 1)  # Transpose to (T, B, D) if needed

                # Step 2: Prepare target locations and normalize
                target = getattr(batch, "locations").cuda()
                target = self.normalizer.normalize_location(target)

                # Step 3: Subsample timesteps if needed
                if (
                    config.sample_timesteps is not None
                    and config.sample_timesteps < pred_encs.shape[0]
                ):
                    sampled_pred_encs, sampled_target_locs = [], []
                    for i in range(pred_encs.shape[1]):  # Iterate over batch size
                        indices = torch.randperm(pred_encs.shape[0])[: config.sample_timesteps]
                        sampled_pred_encs.append(pred_encs[indices, i, :])
                        sampled_target_locs.append(target[i, indices])
                    pred_encs = torch.stack(sampled_pred_encs, dim=1)  # (T, B, D)
                    target = torch.stack(sampled_target_locs, dim=0).cuda()  # (B, T, 2)

                # Step 4: Predict locations using the Prober
                pred_locs = torch.stack([prober(x) for x in pred_encs], dim=1)  # (B, T, 2)
                losses = location_losses(pred_locs, target)
                per_probe_loss = losses.mean()

                if step % 100 == 0:
                    print(f"Normalized pred locations loss: {per_probe_loss.item()}")

                # Step 5: Backpropagation and optimization
                optimizer_pred_prober.zero_grad()
                per_probe_loss.backward()
                optimizer_pred_prober.step()
                scheduler.adjust_learning_rate(step)

                step += 1

                if self.quick_debug and step > 2:
                    break

        return prober

    @torch.no_grad()
    def evaluate_all(self, prober):
        """
        Evaluates on all the different validation datasets.
        """
        avg_losses = {}
        for prefix, val_ds in self.val_ds.items():
            avg_losses[prefix] = self.evaluate_pred_prober(prober=prober, val_ds=val_ds)
        return avg_losses

    @torch.no_grad()
    def evaluate_pred_prober(self, prober, val_ds):
        """
        Evaluate the Prober on a single validation dataset.
        """
        model = self.model
        probing_losses = []
        prober.eval()

        for batch in tqdm(val_ds, desc="Eval probe pred"):
            pred_encs = model(states=batch.states, actions=batch.actions)  # (B, T, D)
            pred_encs = pred_encs.transpose(0, 1)  # Transpose to (T, B, D)

            target = getattr(batch, "locations").cuda()
            target = self.normalizer.normalize_location(target)

            pred_locs = torch.stack([prober(x) for x in pred_encs], dim=1)  # (B, T, 2)
            losses = location_losses(pred_locs, target)
            probing_losses.append(losses.cpu())

        losses_t = torch.stack(probing_losses, dim=0).mean(dim=0)
        losses_t = self.normalizer.unnormalize_mse(losses_t)

        losses_t = losses_t.mean(dim=-1)
        average_eval_loss = losses_t.mean().item()

        return average_eval_loss
