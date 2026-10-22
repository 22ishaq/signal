"""Evaluation helpers: accuracy per SNR, confusion matrices, calibration, OOD scores, latency."""

from __future__ import annotations

import time

import numpy as np
import torch


@torch.no_grad()
def predict_proba(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(x), batch):
        xb = torch.from_numpy(x[s : s + batch]).to(device)
        out.append(torch.softmax(model(xb), dim=1).float().cpu().numpy())
    return np.concatenate(out)


def accuracy_by_snr(y: np.ndarray, pred: np.ndarray, snr: np.ndarray) -> dict[float, float]:
    return {float(s): float(np.mean(pred[snr == s] == y[snr == s])) for s in np.unique(snr)}


def confusion(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    m = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(m, (y, pred), 1)
    return m


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> tuple[float, list[dict]]:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    ece, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(pred[mask] == y[mask]))
        avg_conf = float(np.mean(conf[mask]))
        ece += mask.mean() * abs(acc - avg_conf)
        table.append({"lo": float(lo), "hi": float(hi), "accuracy": acc, "confidence": avg_conf, "count": int(mask.sum())})
    return float(ece), table


def auroc(scores_in: np.ndarray, scores_out: np.ndarray) -> float:
    """Probability that an in-distribution sample scores higher than an OOD sample (Mann-Whitney U)."""
    scores = np.concatenate([scores_in, scores_out])
    ranks = scores.argsort().argsort().astype(np.float64) + 1
    r_in = ranks[: len(scores_in)].sum()
    n1, n2 = len(scores_in), len(scores_out)
    return float((r_in - n1 * (n1 + 1) / 2) / (n1 * n2))


def time_call(fn, repeats: int = 20) -> float:
    """Median wall-clock seconds of fn() over `repeats` calls (after one warm-up)."""
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))
