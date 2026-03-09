from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def normalize(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    lo, hi = float(np.min(y)), float(np.max(y))
    if hi > lo:
        return (y - lo) / (hi - lo)
    return np.zeros_like(y)


def local_peaks(y: np.ndarray) -> np.ndarray:
    if len(y) < 5:
        return np.array([], dtype=int)
    return np.where((y[1:-1] >= y[:-2]) & (y[1:-1] > y[2:]))[0] + 1


def find_secondary_peak(y: np.ndarray, x: np.ndarray) -> int:
    peaks = local_peaks(y)
    if len(peaks) == 0:
        return int(np.argmax(y))
    order = peaks[np.argsort(y[peaks])[::-1]]
    main = int(order[0])
    min_sep = max(8, int(0.08 * len(y)))
    for idx in order[1:]:
        i = int(idx)
        if abs(i - main) >= min_sep:
            return i
    # Fallback: strongest point on the right side of main peak
    right = np.arange(main + 1, len(y))
    if len(right) > 0:
        return int(right[np.argmax(y[right])])
    return main


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "processed" / "dataset_b_array.npz"
    out_path = project_root / "results" / "paper_figures" / "Fig4_spectrum_distortion_refined.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(data_path, allow_pickle=True)
    x = np.asarray(d["wavelengths"], dtype=float).reshape(-1)

    # Keep scientific content consistent with current Figure 4 picks.
    ideal_idx = 5090
    distorted_idx = 917
    X_local = np.asarray(d["X_local"], dtype=float)
    X_total = np.asarray(d["X_total"], dtype=float)
    if ideal_idx >= X_local.shape[0]:
        ideal_idx = 0
    if distorted_idx >= X_total.shape[0]:
        distorted_idx = 0

    y_ideal = normalize(X_local[ideal_idx])
    y_dist = normalize(X_total[distorted_idx])

    # 1800 x 800 px @ 300 dpi
    dpi = 300
    fig_w, fig_h = 1800 / dpi, 800 / dpi

    plt.rcParams["font.family"] = "Arial"
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=dpi, sharey=True)
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_linewidth(1.2)
        ax.tick_params(axis="both", labelsize=8, width=1.0, length=4)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_ylim(-0.02, 1.02)

    line_width = 1.8
    axes[0].plot(x, y_ideal, color="black", linewidth=line_width)
    axes[1].plot(x, y_dist, color="black", linewidth=line_width)

    axes[0].set_title("(a) Ideal spectrum", fontsize=11, pad=4)
    axes[1].set_title("(b) Distorted spectrum", fontsize=11, pad=4)

    axes[0].set_xlabel("Wavelength (nm)", fontsize=10)
    axes[1].set_xlabel("Wavelength (nm)", fontsize=10)
    axes[0].set_ylabel("Normalized Reflectivity", fontsize=10)

    sec_idx = find_secondary_peak(y_dist, x)
    x_span = float(x[-1] - x[0])
    annotation_text = "Spectral overlap"
    axes[1].annotate(
        annotation_text,
        xy=(x[sec_idx], y_dist[sec_idx]),
        xytext=(x[sec_idx] - 0.16 * x_span, min(0.92, y_dist[sec_idx] + 0.18)),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="black"),
        fontsize=9,
        ha="left",
        va="center",
    )

    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.18, top=0.88, wspace=0.10)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    print("Figure path:", out_path.as_posix())
    print("Font:", "Arial")
    print("Line width:", f"curve={line_width}, axes=1.2, tick_width=1.0, tick_length=4, arrow=1.2")
    print("Annotation:", annotation_text)
    print("SCI style:", "Yes")


if __name__ == "__main__":
    main()

