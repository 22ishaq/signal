"""Redraws every figure from results/ (and the saved model for the 8PSK histogram if
results/ood_confidence.npz is missing), without re-running the experiments.

    python scripts/make_figures.py [project_root]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from amc.figures import draw_all  # noqa: E402

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..")
    print("figures written to", draw_all(root))
