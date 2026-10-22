"""Static figures for the report (matplotlib, light theme).

Colours follow a validated categorical order (blue, orange, aqua, yellow) that keeps adjacent
series apart under colour-vision deficiency. Two of the hues are below 3:1 contrast on the
surface, so every line carries a direct end label and every figure has a table in REPORT.md.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SEQ_BLUE = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def _style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)


def accuracy_vs_snr(curves: dict[str, dict[float, float]], path: str, title: str) -> None:
    """One line per method; legend plus a direct label at each line's right end."""
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    ends = []
    for (name, curve), color in zip(curves.items(), SERIES):
        snrs = sorted(curve)
        acc = [100 * curve[s] for s in snrs]
        ax.plot(snrs, acc, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round", label=name,
                marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ends.append((acc[-1], name, color, snrs[-1]))
    # direct labels, nudged apart so they never overlap
    ends.sort()
    placed: list[float] = []
    for y, name, color, x in ends:
        yy = y
        while placed and yy - placed[-1] < 4.5:
            yy = placed[-1] + 4.5
        placed.append(yy)
        ax.annotate(f"{name}  {y:.1f}%", xy=(x, y), xytext=(x + 0.6, yy), color=INK, fontsize=8.5, va="center",
                    arrowprops=dict(arrowstyle="-", color=color, linewidth=1))
    ax.set_ylim(15, 102)
    ax.set_xlim(min(min(c) for c in curves.values()) - 0.5, max(max(c) for c in curves.values()) + 9)
    ax.axhline(25, color=INK_2, linewidth=1, linestyle="-", alpha=0.35)
    ax.text(ax.get_xlim()[0] + 0.3, 21.5, "chance (25%)", color=INK_2, fontsize=8, va="center")
    _style(ax, "Es/N0 (dB)", "Accuracy (%)", title)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def confusion_matrices(mats: dict[str, np.ndarray], classes: tuple[str, ...], path: str, title: str) -> None:
    """Row-normalised confusion matrices side by side, single-hue sequential ramp."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", ("#f4f8fd",) + SEQ_BLUE)
    fig, axes = plt.subplots(1, len(mats), figsize=(4.2 * len(mats), 4), dpi=150)
    axes = np.atleast_1d(axes)
    for ax, (name, m) in zip(axes, mats.items()):
        norm = m / m.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                v = 100 * norm[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(classes)), classes, fontsize=8, color=INK_2)
        ax.set_yticks(range(len(classes)), classes, fontsize=8, color=INK_2)
        ax.set_xlabel("predicted", color=INK_2, fontsize=9)
        ax.set_ylabel("true", color=INK_2, fontsize=9)
        ax.set_title(name, color=INK, fontsize=10, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def grouped_bars(groups: list[str], series: dict[str, list[float]], path: str, title: str, ylabel: str) -> None:
    """Grouped bars (<= 24px wide, 2px surface gap, value at the tip)."""
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    n = len(series)
    width = min(0.8 / n, 0.22)
    x = np.arange(len(groups))
    for k, ((name, vals), color) in enumerate(zip(series.items(), SERIES)):
        pos = x + (k - (n - 1) / 2) * width
        bars = ax.bar(pos, vals, width=width * 0.92, color=color, label=name, edgecolor=SURFACE, linewidth=1)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    import textwrap

    ax.set_xticks(x, [textwrap.fill(g, 13) for g in groups], fontsize=9, color=INK_2)
    ax.set_ylim(0, 108)
    _style(ax, "", ylabel, title)
    ax.grid(axis="x", visible=False)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=n, frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def lines_vs_x(x: list[float], series: dict[str, list[float]], path: str, title: str, xlabel: str, ylabel: str,
               logx: bool = False, extra: list[tuple[float, float, int, str]] | None = None) -> None:
    """One line per series; `extra` adds separate hollow markers (x, y, series index, label)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    for ex, ey, idx, label in extra or []:
        ax.plot([ex], [ey], marker="o", markersize=8, markerfacecolor=SURFACE, markeredgecolor=SERIES[idx],
                markeredgewidth=2, linestyle="none")
        ax.annotate(f"{label}  {ey:.1f}", xy=(ex, ey), xytext=(0, 10), textcoords="offset points", ha="center",
                    color=INK, fontsize=8.5)
    for (name, vals), color in zip(series.items(), SERIES):
        ax.plot(x, vals, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=name)
        ax.annotate(f"{name}  {vals[-1]:.1f}", xy=(x[-1], vals[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK, fontsize=8.5, va="center")
    if logx:
        ticks = list(x) + [e[0] for e in extra or []]
        ax.set_xscale("log")
        ticks = sorted(ticks)
        ax.set_xticks(ticks, [f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"
                              for v in ticks])
        ax.minorticks_off()
        # Ticks less than 2x apart would collide on a log axis: push their labels away from each other.
        labels = ax.get_xticklabels()
        for i in range(len(ticks) - 1):
            if ticks[i + 1] / ticks[i] < 2:
                labels[i].set_horizontalalignment("right")
                labels[i + 1].set_horizontalalignment("left")
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] * (2.2 if logx else 1.12))
    _style(ax, xlabel, ylabel, title)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def confidence_histograms(in_conf: np.ndarray, out_conf: np.ndarray, path: str, title: str) -> None:
    """Max-softmax confidence for in-distribution frames vs an unseen modulation (two hues)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    bins = np.linspace(0.25, 1.0, 31)
    # Outlines rather than translucent fills, so overlapping bins do not mix into a third colour.
    ax.hist(in_conf, bins=bins, density=True, histtype="step", color=SERIES[0], linewidth=2, label="trained classes")
    ax.hist(out_conf, bins=bins, density=True, histtype="step", color=SERIES[1], linewidth=2,
            label="8PSK (never seen)")
    _style(ax, "max softmax probability", "density", title)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def reliability(table: list[dict], path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    conf = [r["confidence"] for r in table]
    acc = [r["accuracy"] for r in table]
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, alpha=0.5)
    ax.plot(conf, acc, color=SERIES[0], linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
            markeredgewidth=1.5)
    ax.set_xlim(0.25, 1.0)
    ax.set_ylim(0.25, 1.0)
    _style(ax, "mean confidence", "accuracy", title)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
