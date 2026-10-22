"""Dataset generation, the CNN, the baselines and the evaluation metrics."""

import numpy as np
import pytest
import torch

from amc import features
from amc.baselines import AnalyticCumulantRule, CumulantQDA, FeatureMLP
from amc.dataset import DatasetConfig, FreshData, generate, generate_grid
from amc.evaluate import accuracy_by_snr, auroc, confusion, expected_calibration_error, predict_proba
from amc.model import AMCNet, count_macs, count_parameters
from amc.train import TrainConfig, train_model

SMALL = DatasetConfig(frame_len=256, snrs_db=(10.0, 20.0))


def test_dataset_shapes_and_determinism():
    a = generate(64, SMALL, seed=1)
    b = generate(64, SMALL, seed=1)
    c = generate(64, SMALL, seed=2)
    assert a["x"].shape == (64, 2, 256) and a["x"].dtype == np.float32
    assert set(np.unique(a["snr"])) <= {10.0, 20.0}
    assert np.array_equal(a["x"], b["x"]) and not np.array_equal(a["x"], c["x"])


def test_grid_is_balanced():
    g = generate_grid(5, SMALL, seed=3)
    assert len(g["y"]) == 4 * 2 * 5
    for c in range(4):
        for s in (10.0, 20.0):
            assert np.sum((g["y"] == c) & (g["snr"] == s)) == 5


def test_fresh_data_changes_every_epoch_but_is_reproducible():
    f1, f2 = FreshData(40, SMALL, base_seed=9, workers=2), FreshData(40, SMALL, base_seed=9, workers=2)
    try:
        e0, e1 = f1(0), f1(1)
        assert e0["x"].shape == (40, 2, 256)
        assert not np.array_equal(e0["x"], e1["x"])
        assert np.array_equal(e0["x"], f2(0)["x"])
    finally:
        f1.close()
        f2.close()


def test_model_accepts_any_length_and_is_scale_invariant():
    m = AMCNet().eval()
    x = torch.randn(3, 2, 512)
    with torch.no_grad():
        assert m(x).shape == (3, 4)
        assert m(torch.randn(2, 2, 128)).shape == (2, 4)
        assert torch.allclose(m(x), m(7.5 * x), atol=1e-4)  # per-frame RMS normalisation


def test_model_size():
    m = AMCNet()
    assert count_parameters(m) == 399_108
    assert count_macs(m, 1024) == 52_659_200


def test_cnn_learns_bpsk_vs_qpsk_quickly():
    cfg = DatasetConfig(frame_len=256, classes=("BPSK", "QPSK"), snrs_db=(20.0,))
    train, val = generate(1200, cfg, seed=10), generate(200, cfg, seed=11)
    model, hist = train_model(train, val, TrainConfig(epochs=3, batch_size=64, log=False), torch.device("cpu"))
    assert hist[-1]["val_acc"] > 0.9


def test_classical_baselines_on_easy_data():
    cfg = DatasetConfig(frame_len=1024, snrs_db=(25.0,)).with_channel(fading="none")
    train, test = generate(800, cfg, seed=12), generate_grid(40, cfg, seed=13)
    f_tr, f_te = features.feature_matrix(train["x"]), features.feature_matrix(test["x"])
    assert np.mean(CumulantQDA().fit(f_tr, train["y"]).predict(f_te)[test["y"] < 2] == test["y"][test["y"] < 2]) > 0.95
    mlp = FeatureMLP(epochs=20).fit(f_tr, train["y"])
    assert mlp.predict_proba(f_te).shape == (len(test["y"]), 4)
    rule = AnalyticCumulantRule().predict(test["x"], test["snr"])
    assert np.mean(rule[test["y"] < 2] == test["y"][test["y"] < 2]) > 0.95  # BPSK and QPSK


def test_metrics():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    snr = np.array([0.0, 0.0, 10.0, 10.0])
    assert accuracy_by_snr(y, pred, snr) == {0.0: 0.5, 10.0: 1.0}
    assert confusion(y, pred, 2).tolist() == [[1, 1], [0, 2]]
    assert auroc(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == 1.0
    assert auroc(np.array([0.1, 0.2]), np.array([0.9, 0.8])) == 0.0
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(probs, np.array([0, 1]))[0] == pytest.approx(0.0)


def test_predict_proba_rows_sum_to_one():
    p = predict_proba(AMCNet(), np.random.randn(10, 2, 256).astype(np.float32), torch.device("cpu"), batch=4)
    assert p.shape == (10, 4) and np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
