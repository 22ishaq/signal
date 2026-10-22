"""Synthetic dataset of impaired I/Q frames.

A frame is `frame_len` complex baseband samples stored as float32 with shape (2, frame_len)
(row 0 = I, row 1 = Q). Labels index `CLASSES`; SNR is Es/N0 in dB.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from amc import channel
from amc.modulation import CLASSES, random_symbols


@dataclass(frozen=True)
class ChannelConfig:
    sps: int = 2                      # samples per symbol
    rolloff: float = 0.35             # RRC roll-off factor
    span: int = 8                     # RRC filter span in symbols
    fading: str = "rician"            # "none", "rician" or "rayleigh"
    k_factor_db: tuple[float, float] = (0.0, 10.0)   # Rician K range (uniform)
    max_doppler: float = 2e-4         # upper bound of normalised Doppler (uniform in [0, max])
    max_cfo: float = 1e-3             # residual CFO, uniform in [-max, max] cycles/sample
    timing_offset: bool = True        # uniform fractional delay in [0, sps)


@dataclass(frozen=True)
class DatasetConfig:
    frame_len: int = 1024
    classes: tuple[str, ...] = CLASSES
    snrs_db: tuple[float, ...] = tuple(float(s) for s in range(-10, 21, 2))
    channel: ChannelConfig = field(default_factory=ChannelConfig)

    def with_channel(self, **kwargs) -> "DatasetConfig":
        return replace(self, channel=replace(self.channel, **kwargs))


def synthesize(
    labels: np.ndarray, snr_db: np.ndarray, cfg: DatasetConfig, rng: np.random.Generator, class_names=None
) -> np.ndarray:
    """Generates impaired complex frames for the given labels and SNRs. Shape (n, frame_len)."""
    names = class_names or cfg.classes
    ch = cfg.channel
    n = labels.shape[0]
    guard = ch.span + 4
    n_sym = cfg.frame_len // ch.sps + 2 * guard
    taps = channel.rrc_taps(ch.rolloff, ch.sps, ch.span)

    frames = np.empty((n, n_sym * ch.sps), dtype=np.complex128)
    for c, name in enumerate(names):
        idx = np.flatnonzero(labels == c)
        if idx.size:
            frames[idx] = channel.pulse_shape(random_symbols(name, idx.size, n_sym, rng), ch.sps, taps)

    if ch.timing_offset:
        frames = channel.fractional_delay(frames, rng.uniform(0, ch.sps, size=n))
    start = guard * ch.sps
    x = frames[:, start : start + cfg.frame_len]

    if ch.fading != "none":
        doppler = rng.uniform(0, ch.max_doppler, size=n)
        k_db = None if ch.fading == "rayleigh" else rng.uniform(*ch.k_factor_db, size=n)
        x = x * channel.rician_fading(n, cfg.frame_len, k_db, doppler, rng)
    x = channel.apply_cfo(x, rng.uniform(-ch.max_cfo, ch.max_cfo, size=n), rng.uniform(0, 2 * np.pi, size=n))
    return channel.awgn(x, snr_db, ch.sps, rng)


def to_iq(z: np.ndarray) -> np.ndarray:
    """Complex (n, L) -> float32 (n, 2, L)."""
    return np.stack([z.real, z.imag], axis=1).astype(np.float32)


def generate(
    n_frames: int, cfg: DatasetConfig, seed: int, snr_db: float | None = None, chunk: int = 1024
) -> dict[str, np.ndarray]:
    """Random dataset: uniform class labels and SNRs drawn from `cfg.snrs_db` (or fixed)."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, len(cfg.classes), size=n_frames)
    snrs = np.full(n_frames, snr_db) if snr_db is not None else rng.choice(np.array(cfg.snrs_db), size=n_frames)
    x = np.empty((n_frames, 2, cfg.frame_len), dtype=np.float32)
    for s in range(0, n_frames, chunk):
        e = min(s + chunk, n_frames)
        x[s:e] = to_iq(synthesize(labels[s:e], snrs[s:e], cfg, rng))
    return {"x": x, "y": labels.astype(np.int64), "snr": snrs.astype(np.float32)}


def generate_grid(per_point: int, cfg: DatasetConfig, seed: int, class_names=None) -> dict[str, np.ndarray]:
    """Balanced test set: `per_point` frames for every (class, SNR) pair."""
    names = class_names or cfg.classes
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(len(names)), per_point * len(cfg.snrs_db))
    snrs = np.tile(np.repeat(np.array(cfg.snrs_db), per_point), len(names))
    x = np.empty((labels.size, 2, cfg.frame_len), dtype=np.float32)
    chunk = 1024
    for s in range(0, labels.size, chunk):
        e = min(s + chunk, labels.size)
        x[s:e] = to_iq(synthesize(labels[s:e], snrs[s:e], cfg, rng, class_names=names))
    return {"x": x, "y": labels.astype(np.int64), "snr": snrs.astype(np.float32)}


class FreshData:
    """A new random training set for every epoch, generated in worker processes.

    Training on fresh synthetic frames each epoch removes the memorisation that a fixed set allows
    (the first run reached 96% training vs 66% validation accuracy). The next epoch's frames are
    generated while the current epoch trains.
    """

    def __init__(self, n_frames: int, cfg: DatasetConfig, base_seed: int, workers: int = 8):
        from concurrent.futures import ProcessPoolExecutor

        self.n, self.cfg, self.base_seed, self.workers = n_frames, cfg, base_seed, workers
        self.pool = ProcessPoolExecutor(max_workers=workers)
        self.pending = {0: self._submit(0)}

    def _submit(self, epoch: int):
        sizes = [self.n // self.workers + (1 if i < self.n % self.workers else 0) for i in range(self.workers)]
        return [self.pool.submit(generate, size, self.cfg, self.base_seed + 1000 * epoch + i)
                for i, size in enumerate(sizes) if size]

    def __call__(self, epoch: int) -> dict[str, np.ndarray]:
        futures = self.pending.pop(epoch) if epoch in self.pending else self._submit(epoch)
        self.pending[epoch + 1] = self._submit(epoch + 1)
        parts = [f.result() for f in futures]
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    def close(self) -> None:
        self.pool.shutdown(cancel_futures=True)
