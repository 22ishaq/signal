"""Physics checks for the transmitter/channel model and the classical features."""

import numpy as np
import pytest

from amc import channel, features
from amc.dataset import DatasetConfig, synthesize
from amc.modulation import CLASSES, constellation, random_symbols, theoretical_cumulants


@pytest.mark.parametrize("name", CLASSES + ("8PSK",))
def test_constellations_have_unit_energy(name):
    assert np.isclose(np.mean(np.abs(constellation(name)) ** 2), 1.0)


def test_theoretical_cumulants_match_the_literature():
    # Swami & Sadler (2000): C42 = -2, -1, -0.68, -0.619
    expected = {"BPSK": -2.0, "QPSK": -1.0, "16QAM": -0.68, "64QAM": -0.619}
    for name, c42 in expected.items():
        assert theoretical_cumulants(name)["C42"] == pytest.approx(c42, abs=1e-3)
    assert theoretical_cumulants("BPSK")["C20"] == pytest.approx(1.0)
    assert theoretical_cumulants("QPSK")["C20"] == pytest.approx(0.0, abs=1e-12)


def test_unknown_modulation_rejected():
    with pytest.raises(ValueError):
        constellation("256QAM")


def test_random_symbols_come_from_the_constellation():
    rng = np.random.default_rng(0)
    s = random_symbols("16QAM", 3, 500, rng)
    pts = constellation("16QAM")
    assert s.shape == (3, 500)
    assert np.all(np.min(np.abs(s[..., None] - pts[None, None, :]), axis=-1) < 1e-12)


def test_rrc_filter_is_unit_energy_and_symmetric():
    taps = channel.rrc_taps(0.35, 2, 8)
    assert np.isclose(np.sum(taps ** 2), 1.0)
    assert np.allclose(taps, taps[::-1])


def test_pulse_shaping_gives_unit_power_per_sample():
    rng = np.random.default_rng(1)
    x = channel.pulse_shape(random_symbols("QPSK", 200, 512, rng), 2, channel.rrc_taps(0.35, 2, 8))
    assert np.mean(np.abs(x[:, 64:-64]) ** 2) == pytest.approx(1.0, rel=0.03)


def test_awgn_matches_the_requested_es_n0():
    rng = np.random.default_rng(2)
    x = np.ones((400, 2048), dtype=complex)
    y = channel.awgn(x, np.full(400, 10.0), sps=2, rng=rng)
    noise_power = np.mean(np.abs(y - x) ** 2)
    # Es/N0 = 10 dB with 2 samples per symbol -> per-sample noise variance 2/10
    assert noise_power == pytest.approx(0.2, rel=0.02)


@pytest.mark.parametrize("k_db", [None, np.full(2000, 6.0)])
def test_fading_has_unit_average_power(k_db):
    rng = np.random.default_rng(3)
    h = channel.rician_fading(2000, 256, k_db, np.full(2000, 2e-4), rng)
    assert np.mean(np.abs(h) ** 2) == pytest.approx(1.0, rel=0.05)


def test_cfo_rotates_at_the_requested_rate():
    x = np.ones((1, 100), dtype=complex)
    y = channel.apply_cfo(x, np.array([0.01]), np.array([0.0]))
    assert np.allclose(np.angle(y[0, 1:] / y[0, :-1]), 2 * np.pi * 0.01)


def test_integer_fractional_delay_is_a_rotation_of_the_frame():
    x = np.arange(16, dtype=complex)[None, :]
    assert np.allclose(channel.fractional_delay(x, np.array([3.0])), np.roll(x, 3))


@pytest.mark.parametrize("name", CLASSES)
def test_front_end_recovers_symbol_rate_cumulants(name):
    """Matched filter + blind timing + blind CFO correction give back the textbook C42 at high SNR."""
    cfg = DatasetConfig(classes=(name,)).with_channel(fading="none")
    rng = np.random.default_rng(4)
    z = synthesize(np.zeros(200, dtype=int), np.full(200, 40.0), cfg, rng)
    x = np.stack([z.real, z.imag], axis=1).astype(np.float32)
    c = features.cumulants(features.classical_front_end(x))
    assert np.mean(c["C42"]) == pytest.approx(theoretical_cumulants(name)["C42"], abs=0.04)


def test_blind_cfo_estimate():
    rng = np.random.default_rng(5)
    s = random_symbols("QPSK", 50, 2048, rng)
    cfo = rng.uniform(-1e-3, 1e-3, 50)
    est = features.estimate_cfo(channel.apply_cfo(s, cfo, np.zeros(50)))
    assert np.max(np.abs(est - cfo)) < 5e-5


def test_m2m4_snr_estimate_is_unbiased_for_psk():
    rng = np.random.default_rng(6)
    s = random_symbols("QPSK", 200, 4096, rng)
    y = channel.awgn(s, np.full(200, 10.0), sps=1, rng=rng)
    assert np.mean(features.m2m4_snr_db(y)) == pytest.approx(10.0, abs=0.3)
