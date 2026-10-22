"""Digital modulation constellations, normalised to unit average symbol energy."""

from __future__ import annotations

import numpy as np

CLASSES: tuple[str, ...] = ("BPSK", "QPSK", "16QAM", "64QAM")


def _square_qam(m: int) -> np.ndarray:
    side = int(round(np.sqrt(m)))
    if side * side != m:
        raise ValueError(f"{m}-QAM is not a square constellation")
    levels = np.arange(-(side - 1), side, 2, dtype=np.float64)
    points = (levels[:, None] + 1j * levels[None, :]).ravel()
    return points / np.sqrt(np.mean(np.abs(points) ** 2))


def constellation(name: str) -> np.ndarray:
    """Constellation points with E|s|^2 = 1.

    ``8PSK`` is also available; it is not one of the trained classes and is used as an
    out-of-distribution probe.
    """
    if name == "BPSK":
        return np.array([-1.0, 1.0], dtype=np.complex128)
    if name == "QPSK":
        return np.exp(1j * (np.pi / 4 + np.pi / 2 * np.arange(4)))
    if name == "8PSK":
        return np.exp(1j * 2 * np.pi * np.arange(8) / 8)
    if name == "16QAM":
        return _square_qam(16)
    if name == "64QAM":
        return _square_qam(64)
    raise ValueError(f"unknown modulation {name!r}")


def random_symbols(name: str, n_frames: int, n_symbols: int, rng: np.random.Generator) -> np.ndarray:
    """i.i.d. equiprobable symbols, shape (n_frames, n_symbols), complex128."""
    points = constellation(name)
    return points[rng.integers(0, points.size, size=(n_frames, n_symbols))]


def theoretical_cumulants(name: str) -> dict[str, float]:
    """Normalised cumulants of the noise-free constellation (exact expectations over the points).

    For the classical values: C40 = -2, -1, -0.68, -0.619 and C42 = -2, -1, -0.68, -0.619 for
    BPSK, QPSK, 16QAM, 64QAM respectively (Swami & Sadler, 2000).
    """
    s = constellation(name)
    m20 = np.mean(s**2)
    m21 = np.mean(np.abs(s) ** 2)
    m40 = np.mean(s**4)
    m42 = np.mean(np.abs(s) ** 4)
    c40 = m40 - 3 * m20**2
    c42 = m42 - np.abs(m20) ** 2 - 2 * m21**2
    return {
        "C20": float(np.abs(m20)),
        "C21": float(m21),
        "C40": float(np.real(c40)),
        "C42": float(np.real(c42)),
    }
