"""Classical signal features: higher-order cumulants, blind CFO and SNR estimation.

Cumulant estimators follow Swami & Sadler, "Hierarchical digital modulation classification using
cumulants" (IEEE Trans. Commun., 2000), written in phase-invariant form so that an unknown carrier
phase does not change the features.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps_signal

from amc import channel


def to_complex(x: np.ndarray) -> np.ndarray:
    """float (n, 2, L) -> complex128 (n, L)."""
    return x[:, 0, :].astype(np.float64) + 1j * x[:, 1, :].astype(np.float64)


def normalize_power(z: np.ndarray) -> np.ndarray:
    return z / np.sqrt(np.mean(np.abs(z) ** 2, axis=1, keepdims=True))


def estimate_cfo(z: np.ndarray, nfft: int = 8192) -> np.ndarray:
    """Blind CFO estimate (cycles/sample) from the spectral line of z^4.

    For BPSK, QPSK and square QAM, E[s^4] != 0, so z^4 contains a tone at 4x the frequency offset.
    The estimate is unambiguous for |cfo| < 1/8.
    """
    spec = np.abs(np.fft.fft(z**4, n=nfft, axis=1))
    k = np.argmax(spec, axis=1)
    f = np.fft.fftfreq(nfft)[k]
    return f / 4.0


def correct_cfo(z: np.ndarray, cfo: np.ndarray) -> np.ndarray:
    n = np.arange(z.shape[1])[None, :]
    return z * np.exp(-2j * np.pi * cfo[:, None] * n)


def cumulants(z: np.ndarray) -> dict[str, np.ndarray]:
    """Sample cumulants per frame (z is assumed zero mean). Phase-invariant magnitudes."""
    a2 = np.abs(z) ** 2
    m20 = np.mean(z**2, axis=1)
    m21 = np.mean(a2, axis=1)
    m40 = np.mean(z**4, axis=1)
    m41 = np.mean(z**2 * a2, axis=1)  # E[z^3 z*]
    m42 = np.mean(a2**2, axis=1)
    m60 = np.mean(z**6, axis=1)
    m63 = np.mean(a2**3, axis=1)
    c40 = m40 - 3 * m20**2
    c41 = m41 - 3 * m20 * m21
    c42 = m42 - np.abs(m20) ** 2 - 2 * m21**2
    c60 = m60 - 15 * m20 * m40 + 30 * m20**3
    c63 = m63 - 6 * np.real(np.conj(m20) * m41) - 9 * m42 * m21 + 18 * np.abs(m20) ** 2 * m21 + 12 * m21**3
    return {
        "C20": np.abs(m20),
        "C21": m21,
        "C40": np.abs(c40),
        "C41": np.abs(c41),
        "C42": np.real(c42),
        "C60": np.abs(c60),
        "C63": np.real(c63),
    }


def m2m4_snr_db(z: np.ndarray) -> np.ndarray:
    """Blind M2M4 SNR estimate (exact for constant-modulus signals in AWGN), per sample, in dB."""
    m2 = np.mean(np.abs(z) ** 2, axis=1)
    m4 = np.mean(np.abs(z) ** 4, axis=1)
    s = np.sqrt(np.clip(2 * m2**2 - m4, 1e-12, None))
    n = np.clip(m2 - s, 1e-12, None)
    return 10 * np.log10(s / n)


def symbol_rate_samples(
    z: np.ndarray, sps: int = 2, rolloff: float = 0.35, span: int = 8, phases: int = 8
) -> np.ndarray:
    """Matched-filters with the (assumed known) RRC pulse and samples once per symbol.

    The sampling phase is chosen blindly per frame among `phases` candidates per symbol using the
    maximum-output-energy criterion: for a Nyquist overall pulse, E|y(t)|^2 at the symbol instants
    exceeds its value at any other phase. Filter transients at both ends are discarded.
    """
    taps = channel.rrc_taps(rolloff, sps, span)
    mf = sps_signal.oaconvolve(z, taps[None, :], mode="same", axes=1)
    up = phases // sps
    fine = sps_signal.resample(mf, mf.shape[1] * up, axis=1)
    trim = span * phases
    fine = fine[:, trim:-trim]
    energy = np.stack([np.mean(np.abs(fine[:, p::phases]) ** 2, axis=1) for p in range(phases)], axis=1)
    best = np.argmax(energy, axis=1)
    n_sym = (fine.shape[1] - phases) // phases
    idx = best[:, None] + phases * np.arange(n_sym)[None, :]
    return np.take_along_axis(fine, idx, axis=1)


FEATURE_NAMES = ("|C20|", "|C40|", "|C41|", "C42", "|C60|", "C63", "SNR_m2m4")


def classical_front_end(x: np.ndarray, sps: int = 2, rolloff: float = 0.35, span: int = 8) -> np.ndarray:
    """Power normalisation -> blind CFO correction -> matched filter + timing -> symbol samples."""
    z = normalize_power(to_complex(x))
    z = correct_cfo(z, estimate_cfo(z))
    return normalize_power(symbol_rate_samples(z, sps, rolloff, span))


def feature_matrix(x: np.ndarray, sps: int = 2, rolloff: float = 0.35, span: int = 8) -> np.ndarray:
    """Feature vector per frame: normalised symbol-rate cumulants + blind SNR estimate."""
    z = classical_front_end(x, sps, rolloff, span)
    c = cumulants(z)
    c21 = c["C21"]
    feats = np.stack(
        [
            c["C20"] / c21,
            c["C40"] / c21**2,
            c["C41"] / c21**2,
            c["C42"] / c21**2,
            c["C60"] / c21**3,
            c["C63"] / c21**3,
            np.clip(m2m4_snr_db(z), -20, 40) / 10.0,
        ],
        axis=1,
    )
    return feats.astype(np.float64)
