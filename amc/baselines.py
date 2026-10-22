"""Classical (non-deep-learning) modulation classifiers used as baselines.

* ``CumulantQDA``: cumulant feature vector + quadratic discriminant analysis (one Gaussian per
  class, regularised covariance). Trained on the same data as the CNN, a few hundred parameters.
* ``AnalyticCumulantRule``: no training at all. Compares the noise-corrected |C40| and C42 with
  their theoretical values and picks the nearest class. It is given the true SNR (genie-aided), so
  it shows what the textbook decision rule can do under ideal SNR knowledge.
"""

from __future__ import annotations

import numpy as np

from amc import features
from amc.modulation import CLASSES, theoretical_cumulants


class CumulantQDA:
    def __init__(self, reg: float = 1e-3):
        self.reg = reg
        self.means: np.ndarray | None = None
        self.prec: np.ndarray | None = None
        self.logdet: np.ndarray | None = None
        self.log_prior: np.ndarray | None = None

    def fit(self, feats: np.ndarray, y: np.ndarray) -> "CumulantQDA":
        classes = np.unique(y)
        d = feats.shape[1]
        self.means = np.stack([feats[y == c].mean(axis=0) for c in classes])
        covs = []
        for c in classes:
            cov = np.cov(feats[y == c], rowvar=False) + self.reg * np.eye(d)
            covs.append(cov)
        covs = np.stack(covs)
        self.prec = np.linalg.inv(covs)
        self.logdet = np.linalg.slogdet(covs)[1]
        counts = np.array([np.sum(y == c) for c in classes], dtype=np.float64)
        self.log_prior = np.log(counts / counts.sum())
        return self

    def log_likelihood(self, feats: np.ndarray) -> np.ndarray:
        """log p(features | class) + log prior, shape (n, n_classes)."""
        assert self.means is not None, "fit() first"
        diff = feats[:, None, :] - self.means[None, :, :]
        maha = np.einsum("nci,cij,ncj->nc", diff, self.prec, diff)
        return -0.5 * (maha + self.logdet[None, :]) + self.log_prior[None, :]

    def predict_proba(self, feats: np.ndarray) -> np.ndarray:
        ll = self.log_likelihood(feats)
        ll -= ll.max(axis=1, keepdims=True)
        p = np.exp(ll)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, feats: np.ndarray) -> np.ndarray:
        return np.argmax(self.log_likelihood(feats), axis=1)

    def n_parameters(self) -> int:
        c, d = self.means.shape
        return c * d + c * d * (d + 1) // 2 + c


class AnalyticCumulantRule:
    def __init__(self, classes: tuple[str, ...] = CLASSES):
        self.classes = classes
        ref = [theoretical_cumulants(c) for c in classes]
        self.ref = np.array([[abs(r["C40"]), r["C42"]] for r in ref])

    def predict(self, x: np.ndarray, esn0_db: np.ndarray, sps: int = 2, rolloff: float = 0.35) -> np.ndarray:
        z = features.classical_front_end(x, sps, rolloff)
        c = features.cumulants(z)
        # After the matched filter the per-symbol SNR is Es/N0. Fourth-order cumulants of Gaussian
        # noise vanish, so normalising by the signal power S (not S + N) recovers the signal's own
        # cumulants: S = C21 * snr / (1 + snr).
        snr = 10 ** (esn0_db / 10.0)
        s = c["C21"] * snr / (1.0 + snr)
        point = np.stack([c["C40"] / s**2, c["C42"] / s**2], axis=1)
        dist = np.linalg.norm(point[:, None, :] - self.ref[None, :, :], axis=2)
        return np.argmin(dist, axis=1)


class FeatureMLP:
    """Small MLP on the same cumulant feature vector as ``CumulantQDA``.

    It separates two questions: how much does *learning* a decision rule help (QDA -> MLP), and how
    much do *learned features from raw I/Q* add on top (MLP -> CNN)?
    """

    def __init__(self, hidden: int = 64, epochs: int = 40, lr: float = 3e-3, seed: int = 0):
        self.hidden, self.epochs, self.lr, self.seed = hidden, epochs, lr, seed
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.net = None

    def fit(self, feats: np.ndarray, y: np.ndarray) -> "FeatureMLP":
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        self.mean, self.std = feats.mean(axis=0), feats.std(axis=0) + 1e-9
        xs = torch.from_numpy(((feats - self.mean) / self.std).astype(np.float32))
        ys = torch.from_numpy(y.astype(np.int64))
        d, c = feats.shape[1], int(y.max()) + 1
        self.net = nn.Sequential(nn.Linear(d, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.hidden), nn.ReLU(),
                                 nn.Linear(self.hidden, c))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        gen = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(len(ys), generator=gen)
            for s in range(0, len(perm), 512):
                idx = perm[s : s + 512]
                loss = nn.functional.cross_entropy(self.net(xs[idx]), ys[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        self.net.eval()
        return self

    def predict_proba(self, feats: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            xs = torch.from_numpy(((feats - self.mean) / self.std).astype(np.float32))
            return torch.softmax(self.net(xs), dim=1).numpy()

    def predict(self, feats: np.ndarray) -> np.ndarray:
        return self.predict_proba(feats).argmax(axis=1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())
