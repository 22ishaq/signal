"""Runs every experiment in the report and writes results/results.json, figures/*.png and the
trained model to models/amcnet.pt.

    python scripts/run_experiments.py            # full run (~40 min on an Apple M-series GPU)
    python scripts/run_experiments.py --quick    # tiny sizes, checks the pipeline end to end
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from amc import features  # noqa: E402
from amc.figures import draw_all  # noqa: E402
from amc.baselines import AnalyticCumulantRule, CumulantQDA, FeatureMLP  # noqa: E402
from amc.dataset import DatasetConfig, FreshData, generate, generate_grid  # noqa: E402
from amc.evaluate import (  # noqa: E402
    accuracy_by_snr,
    auroc,
    confusion,
    expected_calibration_error,
    predict_proba,
    time_call,
)
from amc.model import count_macs, count_parameters  # noqa: E402
from amc.modulation import CLASSES  # noqa: E402
from amc.train import TrainConfig, pick_device, train_model  # noqa: E402

FULL = dict(workers=8, n_train=100_000, n_val=8_000, per_point=500, epochs=40, n_awgn=60_000, awgn_epochs=20, shift_pp=300,
            len_pp=300, eff_sizes=(1_000, 4_000, 16_000, 64_000), eff_budget=800_000, ood_pp=1_000)
QUICK = dict(workers=2, n_train=2_000, n_val=400, per_point=10, epochs=1, n_awgn=1_000, awgn_epochs=1, shift_pp=10, len_pp=10,
             eff_sizes=(500, 1_000), eff_budget=2_000, ood_pp=20)

NAMES = ("CNN (raw I/Q)", "MLP on cumulants", "Cumulants + QDA", "Analytic rule (genie SNR)")


def summary(curve: dict[float, float]) -> dict[str, float]:
    hi = [v for s, v in curve.items() if 0 <= s <= 20]
    return {"at_10dB": curve.get(10.0, float("nan")), "mean_0_to_20dB": float(np.mean(hi)) if hi else float("nan")}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), ".."))
    args = ap.parse_args()
    S = QUICK if args.quick else FULL
    out = os.path.abspath(args.out)
    for d in ("results", "figures", "models"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    device = pick_device(args.device)
    cfg = DatasetConfig()
    res: dict = {"config": {"dataset": asdict(cfg), "sizes": S, "device": str(device), "torch": torch.__version__,
                            "machine": platform.platform()}}
    t_start = time.time()

    # ---- data -----------------------------------------------------------------------------------
    log("generating datasets")
    train = generate(S["n_train"], cfg, seed=1)
    val = generate(S["n_val"], cfg, seed=2)
    test = generate_grid(S["per_point"], cfg, seed=3)
    log(f"train {train['x'].shape}, test {test['x'].shape}")

    # ---- classical baselines --------------------------------------------------------------------
    log("classical features")
    f_train = features.feature_matrix(train["x"])
    f_test = features.feature_matrix(test["x"])
    qda = CumulantQDA().fit(f_train, train["y"])
    mlp = FeatureMLP().fit(f_train, train["y"])
    analytic = AnalyticCumulantRule()

    # ---- CNN ------------------------------------------------------------------------------------
    log(f"training CNN on {device} (fresh {S['n_train']} frames per epoch)")
    fresh = FreshData(S["n_train"], cfg, base_seed=100_000, workers=S["workers"])
    model, history = train_model(train, val, TrainConfig(epochs=S["epochs"]), device, epoch_data=fresh)
    fresh.close()
    torch.save(model.state_dict(), os.path.join(out, "models", "amcnet.pt"))
    res["training_history"] = history

    # ---- E1: accuracy vs SNR --------------------------------------------------------------------
    log("E1 accuracy vs SNR")
    p_cnn = predict_proba(model, test["x"], device)
    preds = {
        NAMES[0]: p_cnn.argmax(1),
        NAMES[1]: mlp.predict(f_test),
        NAMES[2]: qda.predict(f_test),
        NAMES[3]: analytic.predict(test["x"], test["snr"]),
    }
    curves = {k: accuracy_by_snr(test["y"], v, test["snr"]) for k, v in preds.items()}
    res["E1_accuracy_vs_snr"] = {k: {"curve": c, **summary(c)} for k, c in curves.items()}
    m10 = test["snr"] == 10.0
    res["E1_per_class_at_10dB"] = {
        k: {CLASSES[c]: float(np.mean(v[m10 & (test["y"] == c)] == c)) for c in range(len(CLASSES))}
        for k, v in preds.items()
    }
    mats = {k: confusion(test["y"][m10], preds[k][m10], len(CLASSES)) for k in NAMES[:3]}
    res["E1_confusion_10dB"] = {k: m.tolist() for k, m in mats.items()}

    # ---- E2: calibration ------------------------------------------------------------------------
    ece, table = expected_calibration_error(p_cnn, test["y"])
    res["E2_calibration"] = {"ece": ece, "bins": table}

    # ---- E3: channel shift at test time -----------------------------------------------------------
    log("E3 channel shift")
    shifts = {
        "trained channel": test,
        "AWGN only": generate_grid(S["shift_pp"], cfg.with_channel(fading="none"), seed=31),
        "Rayleigh (no LOS)": generate_grid(S["shift_pp"], cfg.with_channel(fading="rayleigh"), seed=32),
        "fast fading (10x Doppler)": generate_grid(S["shift_pp"], cfg.with_channel(max_doppler=2e-3), seed=33),
        "large CFO (10x)": generate_grid(S["shift_pp"], cfg.with_channel(max_cfo=1e-2), seed=34),
        "roll-off 0.2 (unseen)": generate_grid(S["shift_pp"], cfg.with_channel(rolloff=0.2), seed=35),
    }
    res["E3_channel_shift"] = {}
    for name, ds in shifts.items():
        f = f_test if ds is test else features.feature_matrix(ds["x"])
        pc = p_cnn.argmax(1) if ds is test else predict_proba(model, ds["x"], device).argmax(1)
        row = {
            NAMES[0]: summary(accuracy_by_snr(ds["y"], pc, ds["snr"])),
            NAMES[1]: summary(accuracy_by_snr(ds["y"], mlp.predict(f), ds["snr"])),
            NAMES[2]: summary(accuracy_by_snr(ds["y"], qda.predict(f), ds["snr"])),
        }
        res["E3_channel_shift"][name] = row

    # ---- E4: trained without fading, deployed with fading ------------------------------------------
    log("E4 train on AWGN, test on fading")
    cfg_awgn = cfg.with_channel(fading="none")
    train_awgn = generate(S["n_awgn"], cfg_awgn, seed=11)
    val_awgn = generate(S["n_val"], cfg_awgn, seed=12)
    fresh_awgn = FreshData(S["n_awgn"], cfg_awgn, base_seed=200_000, workers=S["workers"])
    model_awgn, _ = train_model(train_awgn, val_awgn, TrainConfig(epochs=S["awgn_epochs"], log=False), device,
                                epoch_data=fresh_awgn)
    fresh_awgn.close()
    f_awgn = features.feature_matrix(train_awgn["x"])
    qda_awgn = CumulantQDA().fit(f_awgn, train_awgn["y"])
    mlp_awgn = FeatureMLP().fit(f_awgn, train_awgn["y"])
    awgn_test = shifts["AWGN only"]
    f_awgn_test = features.feature_matrix(awgn_test["x"])
    res["E4_train_awgn"] = {}
    for label, ds, f in (("test: AWGN", awgn_test, f_awgn_test), ("test: Rician fading", test, f_test)):
        res["E4_train_awgn"][label] = {
            NAMES[0]: summary(accuracy_by_snr(ds["y"], predict_proba(model_awgn, ds["x"], device).argmax(1), ds["snr"])),
            NAMES[1]: summary(accuracy_by_snr(ds["y"], mlp_awgn.predict(f), ds["snr"])),
            NAMES[2]: summary(accuracy_by_snr(ds["y"], qda_awgn.predict(f), ds["snr"])),
        }

    # ---- E5: observation length -----------------------------------------------------------------
    log("E5 frame length")
    res["E5_frame_length"] = {}
    lengths = (128, 256, 512, 1024)
    for length in lengths:
        if length == cfg.frame_len:
            ds, f, pc = test, f_test, p_cnn.argmax(1)
        else:
            ds = generate_grid(S["len_pp"], DatasetConfig(frame_len=length), seed=50 + length)
            f = features.feature_matrix(ds["x"])
            pc = predict_proba(model, ds["x"], device).argmax(1)
        res["E5_frame_length"][length] = {
            NAMES[0]: summary(accuracy_by_snr(ds["y"], pc, ds["snr"])),
            NAMES[1]: summary(accuracy_by_snr(ds["y"], mlp.predict(f), ds["snr"])),
            NAMES[2]: summary(accuracy_by_snr(ds["y"], qda.predict(f), ds["snr"])),
        }

    # ---- E6: data efficiency ----------------------------------------------------------------------
    log("E6 data efficiency")
    res["E6_data_efficiency"] = {}
    for n in S["eff_sizes"]:
        sub = {k: v[:n] for k, v in train.items()}
        epochs = max(20 if not args.quick else 1, math.ceil(S["eff_budget"] / n))
        m_n, _ = train_model(sub, val, TrainConfig(epochs=epochs, log=False, seed=n), device)
        pc = predict_proba(m_n, test["x"], device).argmax(1)
        f_sub = f_train[:n]
        res["E6_data_efficiency"][n] = {
            "epochs": epochs,
            NAMES[0]: summary(accuracy_by_snr(test["y"], pc, test["snr"])),
            NAMES[1]: summary(accuracy_by_snr(test["y"], FeatureMLP().fit(f_sub, sub["y"]).predict(f_test), test["snr"])),
            NAMES[2]: summary(accuracy_by_snr(test["y"], CumulantQDA().fit(f_sub, sub["y"]).predict(f_test), test["snr"])),
        }
        log(f"  n={n}: CNN {res['E6_data_efficiency'][n][NAMES[0]]['mean_0_to_20dB']:.3f}")

    # ---- E7: an unseen modulation (open-set behaviour) ---------------------------------------------
    log("E7 unseen modulation (8PSK)")
    ood = generate_grid(S["ood_pp"], cfg, seed=70, class_names=("8PSK",))
    p_ood = predict_proba(model, ood["x"], device)
    hi_in, hi_out = test["snr"] >= 10, ood["snr"] >= 10
    ll_in = qda.log_likelihood(f_test).max(1)
    ll_out = qda.log_likelihood(features.feature_matrix(ood["x"])).max(1)
    labels_out = p_ood[hi_out].argmax(1)
    res["E7_unseen_8psk"] = {
        "cnn_predicted_as": {CLASSES[c]: float(np.mean(labels_out == c)) for c in range(len(CLASSES))},
        "cnn_mean_confidence_in": float(p_cnn[hi_in].max(1).mean()),
        "cnn_mean_confidence_8psk": float(p_ood[hi_out].max(1).mean()),
        "cnn_share_8psk_confidence_over_0.9": float(np.mean(p_ood[hi_out].max(1) > 0.9)),
        "auroc_cnn_max_softmax": auroc(p_cnn[hi_in].max(1), p_ood[hi_out].max(1)),
        "auroc_qda_max_loglik": auroc(ll_in[hi_in], ll_out[hi_out]),
    }
    np.savez_compressed(os.path.join(out, "results", "ood_confidence.npz"), trained=p_cnn[hi_in].max(1),
                        unseen=p_ood[hi_out].max(1))

    # ---- E8: cost --------------------------------------------------------------------------------
    log("E8 compute cost")
    cpu = torch.device("cpu")
    model_cpu = type(model)().to(cpu)
    model_cpu.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    model_cpu.eval()
    one = torch.from_numpy(test["x"][:1])
    batch = torch.from_numpy(test["x"][:256])
    with torch.no_grad():
        t_cpu_1 = time_call(lambda: model_cpu(one))
        t_cpu_256 = time_call(lambda: model_cpu(batch), repeats=5)
    t_dev_256 = time_call(lambda: predict_proba(model, test["x"][:256], device), repeats=5)
    t_feat_256 = time_call(lambda: features.feature_matrix(test["x"][:256]), repeats=5)
    t_qda_256 = time_call(lambda: qda.predict(f_test[:256]), repeats=5)
    res["E8_cost"] = {
        "cnn_parameters": count_parameters(model),
        "cnn_macs_per_frame": count_macs(model, cfg.frame_len),
        "qda_parameters": qda.n_parameters(),
        "mlp_parameters": mlp.n_parameters(),
        "cnn_cpu_ms_per_frame_batch1": 1000 * t_cpu_1,
        "cnn_cpu_ms_per_frame_batch256": 1000 * t_cpu_256 / 256,
        f"cnn_{device.type}_ms_per_frame_batch256": 1000 * t_dev_256 / 256,
        "classical_features_cpu_ms_per_frame": 1000 * t_feat_256 / 256,
        "qda_cpu_ms_per_frame": 1000 * t_qda_256 / 256,
        "cpu_threads": torch.get_num_threads(),
    }
    res["wall_clock_minutes"] = (time.time() - t_start) / 60
    with open(os.path.join(out, "results", "results.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    draw_all(out)
    log(f"done in {res['wall_clock_minutes']:.1f} min")
    for k in NAMES:
        s = res["E1_accuracy_vs_snr"][k]
        print(f"  {k:28s} @10dB {100 * s['at_10dB']:5.1f}%   mean 0-20 dB {100 * s['mean_0_to_20dB']:5.1f}%")


if __name__ == "__main__":
    main()
