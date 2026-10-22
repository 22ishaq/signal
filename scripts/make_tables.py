"""Renders the result tables used in REPORT.md from results/results.json (no hand-copied numbers).

    python scripts/make_tables.py > results/tables.md
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(__file__)
NAMES = ("CNN (raw I/Q)", "MLP on cumulants", "Cumulants + QDA", "Analytic rule (genie SNR)")


def pct(v: float) -> str:
    return f"{100 * v:.1f}"


def main(path: str) -> None:
    r = json.load(open(path))
    out: list[str] = []

    out += ["### Accuracy vs Es/N0 (%)", "", "| Es/N0 (dB) | " + " | ".join(NAMES) + " |",
            "|---:|" + "---:|" * len(NAMES)]
    snrs = sorted(r["E1_accuracy_vs_snr"][NAMES[0]]["curve"], key=float)
    for s in snrs:
        out.append(f"| {float(s):.0f} | " + " | ".join(pct(r["E1_accuracy_vs_snr"][n]["curve"][s]) for n in NAMES) + " |")
    out.append("| **mean 0–20 dB** | " + " | ".join(f"**{pct(r['E1_accuracy_vs_snr'][n]['mean_0_to_20dB'])}**"
                                                 for n in NAMES) + " |")

    out += ["", "### Per-class accuracy at 10 dB (%)", "", "| Method | BPSK | QPSK | 16QAM | 64QAM |", "|---|---:|---:|---:|---:|"]
    for n in NAMES:
        c = r["E1_per_class_at_10dB"][n]
        out.append(f"| {n} | " + " | ".join(pct(c[k]) for k in ("BPSK", "QPSK", "16QAM", "64QAM")) + " |")

    def table(title: str, key: str, label: str, methods=NAMES[:3]) -> None:
        nonlocal out
        out += ["", f"### {title}", "", f"| {label} | " + " | ".join(f"{m} @10 dB / mean" for m in methods) + " |",
                "|---|" + "---:|" * len(methods)]
        for g, row in r[key].items():
            out.append(f"| {g} | " + " | ".join(f"{pct(row[m]['at_10dB'])} / {pct(row[m]['mean_0_to_20dB'])}"
                                               for m in methods) + " |")

    table("Trained once, tested on another channel (%)", "E3_channel_shift", "Test channel")
    table("Trained without fading (AWGN only) (%)", "E4_train_awgn", "Test set")
    table("Observation length (models trained at 1024 samples) (%)", "E5_frame_length", "Samples per frame")
    table("Training-set size (%)", "E6_data_efficiency", "Training frames")

    e7 = r["E7_unseen_8psk"]
    out += ["", "### Unseen modulation (8PSK, Es/N0 ≥ 10 dB)", "",
            f"* CNN labels 8PSK as: " + ", ".join(f"{k} {pct(v)}%" for k, v in e7["cnn_predicted_as"].items()),
            f"* CNN mean max-softmax confidence: {e7['cnn_mean_confidence_8psk']:.3f} on 8PSK vs "
            f"{e7['cnn_mean_confidence_in']:.3f} on the trained classes; "
            f"{pct(e7['cnn_share_8psk_confidence_over_0.9'])}% of 8PSK frames get confidence > 0.9",
            f"* Out-of-distribution detection AUROC: CNN max-softmax {e7['auroc_cnn_max_softmax']:.3f}, "
            f"QDA max log-likelihood {e7['auroc_qda_max_loglik']:.3f} (0.5 = chance)"]

    e8 = r["E8_cost"]
    dev = [k for k in e8 if k.startswith("cnn_") and k.endswith("_batch256") and "cpu" not in k]
    out += ["", "### Cost", "", "| | CNN | Cumulants + QDA |", "|---|---:|---:|",
            f"| Trainable parameters | {e8['cnn_parameters']:,} | {e8['qda_parameters']} |",
            f"| Arithmetic per frame | {e8['cnn_macs_per_frame'] / 1e6:.1f} M multiply-accumulates (counted) | "
            f"≈ 1 M floating-point operations (estimated; the 8192-point FFT for CFO estimation and the "
            f"4x resampling FFTs dominate) |",
            f"| CPU time per frame, batch 1 | {e8['cnn_cpu_ms_per_frame_batch1']:.2f} ms | — |",
            f"| CPU time per frame, batch 256 | {e8['cnn_cpu_ms_per_frame_batch256']:.2f} ms | "
            f"{e8['classical_features_cpu_ms_per_frame'] + e8['qda_cpu_ms_per_frame']:.2f} ms (features + QDA) |"]
    if dev:
        out.append(f"| GPU ({dev[0].split('_')[1]}) per frame, batch 256 | {e8[dev[0]]:.3f} ms | — |")
    out += ["", f"Calibration: CNN expected calibration error {r['E2_calibration']['ece']:.3f}.",
            f"Total experiment wall-clock time: {r['wall_clock_minutes']:.1f} min on {r['config']['device']}."]
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "results.json"))
