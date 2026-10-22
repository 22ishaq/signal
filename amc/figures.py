"""Report figures, drawn from saved results (shared by the experiment runner and make_figures.py)."""

from __future__ import annotations

import json
import os

import numpy as np

from amc import plots
from amc.modulation import CLASSES

NAMES = ("CNN (raw I/Q)", "MLP on cumulants", "Cumulants + QDA", "Analytic rule (genie SNR)")


def ood_confidences(root: str, res: dict) -> tuple[np.ndarray, np.ndarray]:
    path = os.path.join(root, "results", "ood_confidence.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["trained"], d["unseen"]
    import torch

    from amc.dataset import DatasetConfig, generate_grid
    from amc.evaluate import predict_proba
    from amc.model import AMCNet
    from amc.train import pick_device

    device = pick_device()
    model = AMCNet().to(device)
    model.load_state_dict(torch.load(os.path.join(root, "models", "amcnet.pt"), map_location=device))
    cfg = DatasetConfig()
    sizes = res["config"]["sizes"]
    test = generate_grid(sizes["per_point"], cfg, seed=3)
    ood = generate_grid(sizes["ood_pp"], cfg, seed=70, class_names=("8PSK",))
    p_in, p_out = predict_proba(model, test["x"], device), predict_proba(model, ood["x"], device)
    trained, unseen = p_in[test["snr"] >= 10].max(1), p_out[ood["snr"] >= 10].max(1)
    np.savez_compressed(path, trained=trained, unseen=unseen)
    return trained, unseen


def draw_all(root: str) -> str:
    """Draws every report figure into <root>/figures from <root>/results; returns the directory."""
    res_dir, fig = os.path.join(root, "results"), os.path.join(root, "figures")
    os.makedirs(fig, exist_ok=True)
    r = json.load(open(os.path.join(res_dir, "results.json")))
    curves = {n: {float(s): a for s, a in r["E1_accuracy_vs_snr"][n]["curve"].items()} for n in NAMES}
    plots.accuracy_vs_snr(curves, os.path.join(fig, "accuracy_vs_snr.png"),
                          "Classification accuracy vs Es/N0 (Rician fading, CFO, timing offset)")
    plots.confusion_matrices({n: np.array(m) for n, m in r["E1_confusion_10dB"].items()}, CLASSES,
                             os.path.join(fig, "confusion_10dB.png"), "Confusion at Es/N0 = 10 dB (row-normalised, %)")
    e2 = r["E2_calibration"]
    plots.reliability(e2["bins"], os.path.join(fig, "calibration.png"), f"CNN reliability (ECE {e2['ece']:.3f})")
    groups = list(r["E3_channel_shift"])
    plots.grouped_bars(groups, {n: [100 * r["E3_channel_shift"][g][n]["mean_0_to_20dB"] for g in groups] for n in NAMES[:3]},
                       os.path.join(fig, "channel_shift.png"),
                       "Trained once, tested on a different channel (mean accuracy, Es/N0 0–20 dB)", "Accuracy (%)")
    lengths = [int(k) for k in r["E5_frame_length"]]
    plots.lines_vs_x(lengths, {n: [100 * r["E5_frame_length"][str(k)][n]["at_10dB"] for k in lengths] for n in NAMES[:3]},
                     os.path.join(fig, "frame_length.png"), "Accuracy at 10 dB vs observation length (models trained at 1024)",
                     "samples per frame", "Accuracy (%)", logx=True)
    # Data efficiency: lines are fixed training sets (E6). A fixed set of the main run's size comes from
    # the first run (results_fixed_dataset.json) when that file exists. The final CNN, which saw a new
    # set every epoch, is a separate marker at the total number of frames it saw.
    eff_rows = {int(k): v for k, v in r["E6_data_efficiency"].items()}
    fixed_path = os.path.join(res_dir, "results_fixed_dataset.json")
    n_train = r["config"]["sizes"]["n_train"]
    if n_train not in eff_rows and os.path.exists(fixed_path):
        first = json.load(open(fixed_path))["E1_accuracy_vs_snr"]
        eff_rows[n_train] = {n: {"mean_0_to_20dB": first[n]["mean_0_to_20dB"]} for n in NAMES[:3]}
    sizes = sorted(eff_rows)
    eff = {n: [100 * eff_rows[k][n]["mean_0_to_20dB"] for k in sizes] for n in NAMES[:3]}
    seen = n_train * r["config"]["sizes"]["epochs"]
    extra = [(seen, 100 * r["E1_accuracy_vs_snr"][NAMES[0]]["mean_0_to_20dB"], 0, "CNN, fresh frames every epoch")]
    plots.lines_vs_x(sizes, eff, os.path.join(fig, "data_efficiency.png"),
                     "Accuracy vs training data (mean over Es/N0 0–20 dB)", "training frames", "Accuracy (%)",
                     logx=True, extra=extra)
    trained, unseen = ood_confidences(root, r)
    plots.confidence_histograms(trained, unseen, os.path.join(fig, "unseen_8psk.png"),
                                "CNN confidence on an unseen modulation (Es/N0 >= 10 dB)")
    return os.path.abspath(fig)
