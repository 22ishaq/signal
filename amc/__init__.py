"""Automatic modulation classification (AMC) from raw I/Q samples.

Modules:
    modulation  constellations and symbol generation
    channel     pulse shaping, timing offset, fading, CFO, AWGN
    dataset     configurable synthetic dataset generator
    features    higher-order cumulants, blind CFO and SNR estimation
    baselines   classical feature-based classifiers
    model       1-D residual CNN
    train       training loop
    evaluate    metrics (accuracy vs SNR, confusion, calibration, latency)
"""

from amc.modulation import CLASSES

__all__ = ["CLASSES"]
__version__ = "0.1.0"
