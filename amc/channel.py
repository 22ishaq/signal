"""Transmitter and channel impairments applied to frames of symbols.

The chain for each frame is::

    symbols -> upsample by sps -> root-raised-cosine filter -> fractional timing delay
            -> flat Rician/Rayleigh fading with Doppler (sum-of-sinusoids)
            -> carrier frequency offset and phase offset -> complex AWGN at a given Es/N0

All operations are vectorised over the frame axis (axis 0).
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps_signal


def rrc_taps(beta: float, sps: int, span: int) -> np.ndarray:
    """Root-raised-cosine impulse response, `span` symbols long, unit energy."""
    n = np.arange(-span * sps // 2, span * sps // 2 + 1, dtype=np.float64)
    t = n / sps
    taps = np.empty_like(t)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            taps[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta > 0 and np.isclose(abs(ti), 1.0 / (4 * beta)):
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            taps[i] = num / den
    return taps / np.sqrt(np.sum(taps**2))


def pulse_shape(symbols: np.ndarray, sps: int, taps: np.ndarray) -> np.ndarray:
    """Upsample by `sps` and filter. Output has unit average power per sample."""
    n_frames, n_sym = symbols.shape
    up = np.zeros((n_frames, n_sym * sps), dtype=np.complex128)
    up[:, ::sps] = symbols
    shaped = sps_signal.oaconvolve(up, taps[None, :], mode="same", axes=1)
    return shaped * np.sqrt(sps)


def fractional_delay(x: np.ndarray, delay: np.ndarray) -> np.ndarray:
    """Delays each frame by `delay[i]` samples (non-integer allowed) with an FFT phase ramp.

    The shift is circular, so callers generate guard samples and crop the middle of the frame.
    """
    n = x.shape[1]
    freqs = np.fft.fftfreq(n)
    ramp = np.exp(-2j * np.pi * freqs[None, :] * delay[:, None])
    return np.fft.ifft(np.fft.fft(x, axis=1) * ramp, axis=1)


def rician_fading(
    n_frames: int,
    length: int,
    k_factor_db: np.ndarray | None,
    max_doppler: np.ndarray,
    rng: np.random.Generator,
    n_paths: int = 16,
) -> np.ndarray:
    """Time-varying flat-fading gains h[i, n] with E|h|^2 = 1.

    The scattered component is a sum of `n_paths` sinusoids with uniformly distributed angles of
    arrival and phases (Clarke/Jakes model); `max_doppler` is normalised to the sample rate.
    ``k_factor_db=None`` gives Rayleigh fading (no line-of-sight component).
    """
    n = np.arange(length, dtype=np.float64)[None, :, None]
    alpha = rng.uniform(0, 2 * np.pi, size=(n_frames, 1, n_paths))
    phi = rng.uniform(0, 2 * np.pi, size=(n_frames, 1, n_paths))
    fd = max_doppler[:, None, None]
    scattered = np.exp(1j * (2 * np.pi * fd * np.cos(alpha) * n + phi)).sum(axis=2) / np.sqrt(n_paths)
    if k_factor_db is None:
        return scattered
    k = 10.0 ** (k_factor_db / 10.0)[:, None]
    los_phase = rng.uniform(0, 2 * np.pi, size=(n_frames, 1))
    los_angle = rng.uniform(0, 2 * np.pi, size=(n_frames, 1))
    los = np.exp(1j * (2 * np.pi * max_doppler[:, None] * np.cos(los_angle) * n[..., 0] + los_phase))
    return np.sqrt(k / (k + 1)) * los + np.sqrt(1 / (k + 1)) * scattered


def apply_cfo(x: np.ndarray, cfo: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Rotates frame i by exp(j(2*pi*cfo[i]*n + phase[i])); `cfo` in cycles per sample."""
    n = np.arange(x.shape[1], dtype=np.float64)[None, :]
    return x * np.exp(1j * (2 * np.pi * cfo[:, None] * n + phase[:, None]))


def awgn(x: np.ndarray, esn0_db: np.ndarray, sps: int, rng: np.random.Generator) -> np.ndarray:
    """Adds circular complex Gaussian noise for a given Es/N0 per frame.

    The transmitted signal has unit average power per sample, so the symbol energy is `sps`
    sample-powers and the per-sample noise variance is sps / (Es/N0).
    """
    sigma2 = sps / (10.0 ** (esn0_db / 10.0))
    noise = rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
    return x + noise * np.sqrt(sigma2 / 2.0)[:, None]
