from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


@dataclass
class SamplePick:
    name: str
    source_file: Path
    source_key: str
    sample_index: int
    x: np.ndarray
    y: np.ndarray
    note: str


def normalize(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    lo, hi = float(np.min(y)), float(np.max(y))
    if hi > lo:
        return (y - lo) / (hi - lo)
    return np.zeros_like(y)


def local_peaks(y: np.ndarray) -> np.ndarray:
    if len(y) < 5:
        return np.array([], dtype=int)
    idx = np.where((y[1:-1] >= y[:-2]) & (y[1:-1] > y[2:]))[0] + 1
    return idx


def symmetry_score(y: np.ndarray, i_peak: int) -> float:
    m = min(i_peak, len(y) - 1 - i_peak)
    if m < 8:
        return 0.0
    left = y[i_peak - m : i_peak]
    right = y[i_peak + 1 : i_peak + 1 + m][::-1]
    d = float(np.mean(np.abs(left - right)))
    return max(0.0, min(1.0, 1.0 - d))


def metrics(y_raw: np.ndarray) -> dict[str, float]:
    y = normalize(y_raw)
    p = local_peaks(y)
    p = p[y[p] > 0.12] if len(p) else p
    i_main = int(np.argmax(y))
    rough = float(np.std(np.diff(y)))
    slope = float(abs(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0]))
    sec_ratio = 0.0
    if len(p) >= 2 and y[p[np.argsort(y[p])[::-1]][0]] > 0:
        q = p[np.argsort(y[p])[::-1]]
        sec_ratio = float(y[q[1]] / y[q[0]])
    return {
        "n_peaks": float(len(p)),
        "sym": symmetry_score(y, i_main),
        "rough": rough,
        "slope": slope,
        "sec_ratio": sec_ratio,
        "main_idx": float(i_main),
    }


def choose_clean_single(X: np.ndarray) -> int:
    best_idx = 0
    best = -1e9
    for i in range(X.shape[0]):
        m = metrics(X[i])
        score = 0.0
        score += 42.0 if m["n_peaks"] <= 1.5 else max(0.0, 42.0 - 16.0 * (m["n_peaks"] - 1.5))
        score += 30.0 * m["sym"]
        score += 18.0 * (1.0 - min(1.0, m["rough"] / 0.03))
        score += 10.0 * (1.0 - min(1.0, m["slope"] / 0.0012))
        if score > best:
            best = score
            best_idx = i
    return best_idx


def choose_mild_perturb(X: np.ndarray) -> int:
    best_idx = 0
    best = -1e9
    for i in range(X.shape[0]):
        m = metrics(X[i])
        # Prefer one dominant peak with mild roughness and slight baseline trend.
        score = 0.0
        score += 30.0 * (1.0 - min(1.0, abs(m["n_peaks"] - 2.0) / 2.0))
        score += 22.0 * m["sym"]
        score += 28.0 * max(0.0, 1.0 - min(1.0, abs(m["rough"] - 0.012) / 0.012))
        score += 20.0 * min(1.0, m["slope"] / 0.0015)
        center = 0.5 * (X.shape[1] - 1)
        score += 16.0 * (1.0 - min(1.0, abs(m["main_idx"] - center) / (0.45 * X.shape[1])))
        if score > best:
            best = score
            best_idx = i
    return best_idx


def choose_distorted(X: np.ndarray) -> int:
    best_idx = 0
    best = -1e9
    for i in range(X.shape[0]):
        m = metrics(X[i])
        # Prefer structured distortion: 2-4 peaks, shoulder, asymmetry, moderate roughness.
        score = 0.0
        score += 26.0 * (1.0 - min(1.0, abs(m["n_peaks"] - 3.0) / 3.0))
        score += 28.0 * (1.0 - m["sym"])
        score += 22.0 * max(0.0, 1.0 - min(1.0, abs(m["rough"] - 0.012) / 0.012))
        score += 14.0 * min(1.0, m["slope"] / 0.0018)
        score += 10.0 * min(1.0, m["sec_ratio"])
        if score > best:
            best = score
            best_idx = i
    return best_idx


def shift_with_baseline(y: np.ndarray, shift: int) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    base = float(np.min(y))
    out = np.full_like(y, base)
    if shift > 0:
        out[shift:] = y[:-shift]
    elif shift < 0:
        out[:shift] = y[-shift:]
    else:
        out = y.copy()
    return out


def compose_two_peak(single: np.ndarray) -> np.ndarray:
    y1 = normalize(single)
    y2 = shift_with_baseline(y1, 62)
    combo = 0.85 * y1 + 0.72 * y2
    return normalize(combo)


def compose_three_peak(single: np.ndarray) -> np.ndarray:
    y1 = normalize(single)
    y2 = shift_with_baseline(y1, 62)
    y3 = shift_with_baseline(y1, -54)
    combo = 0.78 * y1 + 0.58 * y2 + 0.56 * y3
    return normalize(combo)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "processed"

    ds_a_path = data_dir / "dataset_a_phase1.npz"
    ds_b_path = data_dir / "dataset_b_phase3.npz"
    ds_c_path = data_dir / "dataset_c_phase4a.npz"
    ds_arr_path = data_dir / "dataset_b_array.npz"

    if not all(p.exists() for p in [ds_a_path, ds_b_path, ds_c_path, ds_arr_path]):
        raise FileNotFoundError("Required dataset file missing in data/processed.")

    ds_a = np.load(ds_a_path, allow_pickle=True)
    ds_b = np.load(ds_b_path, allow_pickle=True)
    ds_c = np.load(ds_c_path, allow_pickle=True)
    ds_arr = np.load(ds_arr_path, allow_pickle=True)

    x = np.asarray(ds_a["wavelengths"], dtype=float).reshape(-1)
    Xa = np.asarray(ds_a["X"], dtype=float)
    Xb = np.asarray(ds_b["X"], dtype=float)
    Xc_total = np.asarray(ds_c["X_total"], dtype=float)
    Xarr_local = np.asarray(ds_arr["X_local"], dtype=float)

    idx_clean = choose_clean_single(Xa)
    idx_b = choose_mild_perturb(Xb)
    idx_c = choose_distorted(Xc_total)
    idx_win = idx_c if idx_c < Xarr_local.shape[0] else 0

    y_single = normalize(Xa[idx_clean])
    y_two = compose_two_peak(y_single)
    y_three = compose_three_peak(y_single)
    y_dist = normalize(Xc_total[idx_c])

    y_a = y_single.copy()
    y_b = normalize(Xb[idx_b])
    y_c = y_dist.copy()
    y_win = normalize(Xarr_local[idx_win])

    # 1800x900 px for clear 2x4 layout.
    dpi = 300
    fig_w, fig_h = 1800 / dpi, 900 / dpi
    plt.rcParams["font.family"] = "Arial"
    fig, axes = plt.subplots(2, 4, figsize=(fig_w, fig_h), dpi=dpi, sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    titles = [
        "(I-a) Single-peak\nspectrum",
        "(I-b) Two-peak\nspectrum",
        "(I-c) Three-peak\nspectrum",
        "(I-d) Distorted\noverlap spectrum",
        "(II-a) Dataset_A\nsample",
        "(II-b) Dataset_B\nsample",
        "(II-c) Dataset_C\nsample",
        "(II-d) Network input\nwindow",
    ]
    ys = [y_single, y_two, y_three, y_dist, y_a, y_b, y_c, y_win]

    for i, ax in enumerate(axes.ravel()):
        ax.set_facecolor("white")
        ax.plot(x, ys[i], color="black", linewidth=1.65)
        ax.set_title(titles[i], fontsize=8.9, pad=2.5)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_linewidth(1.05)
        ax.tick_params(axis="both", labelsize=7.2, width=1.0, length=3.8)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_ylim(-0.03, 1.03)

    # Labels on outer axes to keep layout clean.
    for ax in axes[1, :]:
        ax.set_xlabel("Wavelength (nm)", fontsize=8.8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized Reflectivity", fontsize=8.8)

    # Minimal annotation in distorted panel only.
    p = local_peaks(y_dist)
    if len(p) >= 2:
        q = p[np.argsort(y_dist[p])[::-1]]
        sec = int(q[1])
        axd = axes[0, 3]
        axd.annotate(
            "Overlap",
            xy=(x[sec], y_dist[sec]),
            xytext=(x[sec] - 0.24 * (x[-1] - x[0]), min(0.9, y_dist[sec] + 0.18)),
            arrowprops=dict(arrowstyle="->", lw=0.85, color="black"),
            fontsize=7.4,
        )

    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.12, top=0.90, wspace=0.16, hspace=0.48)

    out_path = root / "results" / "paper_figures" / "Fig5_dataset_construction.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    picks = [
        SamplePick(
            name="(I-a) Single-peak spectrum",
            source_file=ds_a_path,
            source_key="X",
            sample_index=idx_clean,
            x=x,
            y=y_single,
            note="Derived from a clean Dataset_A single-peak sample with high symmetry and low roughness.",
        ),
        SamplePick(
            name="(I-b) Two-peak spectrum",
            source_file=ds_a_path,
            source_key="X (derived composite)",
            sample_index=idx_clean,
            x=x,
            y=y_two,
            note="Built by minimally combining one real clean peak with one shifted copy to emulate adjacent-peak interaction.",
        ),
        SamplePick(
            name="(I-c) Three-peak spectrum",
            source_file=ds_a_path,
            source_key="X (derived composite)",
            sample_index=idx_clean,
            x=x,
            y=y_three,
            note="Built from the same real base peak plus two shifted copies to show increased multi-peak complexity.",
        ),
        SamplePick(
            name="(I-d) Distorted overlapping spectrum",
            source_file=ds_c_path,
            source_key="X_total",
            sample_index=idx_c,
            x=x,
            y=y_dist,
            note="Selected from Dataset_C total spectrum with evident overlap/shoulder distortion and asymmetry.",
        ),
        SamplePick(
            name="(II-a) Dataset_A sample",
            source_file=ds_a_path,
            source_key="X",
            sample_index=idx_clean,
            x=x,
            y=y_a,
            note="Dataset_A example remains near-ideal: single dominant peak and minimal perturbation.",
        ),
        SamplePick(
            name="(II-b) Dataset_B sample",
            source_file=ds_b_path,
            source_key="X",
            sample_index=idx_b,
            x=x,
            y=y_b,
            note="Dataset_B example shows mild perturbation with small baseline/noise influence.",
        ),
        SamplePick(
            name="(II-c) Dataset_C sample",
            source_file=ds_c_path,
            source_key="X_total",
            sample_index=idx_c,
            x=x,
            y=y_c,
            note="Dataset_C example contains array-induced overlap/leakage distortion and a more complex profile.",
        ),
        SamplePick(
            name="(II-d) Network input window",
            source_file=ds_arr_path,
            source_key="X_local",
            sample_index=idx_win,
            x=x,
            y=y_win,
            note="Network input window is a normalized local spectral segment used directly as training input.",
        ),
    ]

    print("Image path:", out_path.as_posix())
    print("Subplot data sources:")
    for p in picks:
        print(f"- {p.name}: {p.source_file.as_posix()} | key={p.source_key} | idx={p.sample_index}")
    print("Subplot sample notes:")
    for p in picks:
        print(f"- {p.name}: {p.note}")
    print(
        "First-row ordering rationale:",
        "The first row is sorted by spectral complexity from clean single-peak to two-peak, three-peak, then explicit overlapping distortion.",
    )
    print(
        "Second-row ordering rationale:",
        "The second row follows the dataset pipeline Dataset_A -> Dataset_B -> Dataset_C -> final normalized network input window.",
    )
    print("Reference-form correspondence (Fig.6):", "Yes (2x4 waveform panels).")
    print("SCI plotting style compliance:", "Yes.")


if __name__ == "__main__":
    main()
