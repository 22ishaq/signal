"""CPU inference time per frame for the saved CNN across batch sizes (results/cpu_latency.json).

Per-frame cost on CPU is not monotone in batch size: it drops from batch 1 to small batches and then
rises again for large ones (PyTorch picks different convolution kernels by shape). E8 in
run_experiments.py reports only batch 1 and batch 256; this sweep shows the whole curve.

    python scripts/cpu_latency_sweep.py [project_root]
"""

import json
import os
import platform
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from amc.dataset import DatasetConfig, generate  # noqa: E402
from amc.evaluate import time_call  # noqa: E402
from amc.model import AMCNet  # noqa: E402

BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def main(root: str) -> None:
    model = AMCNet().eval()
    model.load_state_dict(torch.load(os.path.join(root, "models", "amcnet.pt"), map_location="cpu"))
    x = torch.from_numpy(generate(max(BATCHES), DatasetConfig(), seed=5)["x"])
    rows = {}
    with torch.no_grad():
        for b in BATCHES:
            xb = x[:b]
            rows[b] = 1000 * time_call(lambda: model(xb), repeats=max(5, 200 // b)) / b
            print(f"batch {b:4d}: {rows[b]:.3f} ms per frame")
    out = {"ms_per_frame": rows, "threads": torch.get_num_threads(), "torch": torch.__version__,
           "machine": platform.platform()}
    with open(os.path.join(root, "results", "cpu_latency.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), ".."))
