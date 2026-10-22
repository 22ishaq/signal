"""Training loop for AMCNet."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from amc.model import AMCNet


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 0
    width: int = 32
    label_smoothing: float = 0.0
    log: bool = True


def train_model(
    train: dict[str, np.ndarray], val: dict[str, np.ndarray] | None, cfg: TrainConfig, device: torch.device,
    epoch_data=None,
) -> tuple[AMCNet, list[dict]]:
    """Trains with AdamW + one-cycle cosine schedule; returns the model with the best validation accuracy.

    `epoch_data(epoch)`, if given, supplies a fresh training set for each epoch (same size as `train`).
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = AMCNet(n_classes=int(train["y"].max()) + 1, width=cfg.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train["y"]) / cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch, pct_start=0.15)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    x_all = torch.from_numpy(train["x"])
    y_all = torch.from_numpy(train["y"])
    history: list[dict] = []
    best_acc, best_state = -1.0, None
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        if epoch_data is not None:
            fresh = epoch_data(epoch)
            x_all, y_all = torch.from_numpy(fresh["x"]), torch.from_numpy(fresh["y"])
        order = torch.from_numpy(rng.permutation(len(y_all)))
        total, correct, loss_sum = 0, 0, 0.0
        for s in range(0, len(order), cfg.batch_size):
            idx = order[s : s + cfg.batch_size]
            xb = x_all[idx].to(device, non_blocking=True)
            yb = y_all[idx].to(device, non_blocking=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(idx)
        record = {"epoch": epoch + 1, "loss": loss_sum / total, "train_acc": correct / total, "seconds": time.time() - t0}
        if val is not None:
            from amc.evaluate import predict_proba

            pv = predict_proba(model, val["x"], device)
            record["val_acc"] = float(np.mean(pv.argmax(1) == val["y"]))
            if record["val_acc"] > best_acc:
                best_acc = record["val_acc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(record)
        if cfg.log:
            print(
                f"  epoch {record['epoch']:3d}  loss {record['loss']:.4f}  train {record['train_acc']:.4f}"
                + (f"  val {record['val_acc']:.4f}" if "val_acc" in record else "")
                + f"  ({record['seconds']:.1f}s)",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history
