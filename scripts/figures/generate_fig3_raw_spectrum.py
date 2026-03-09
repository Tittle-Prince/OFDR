from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

try:
    from scipy.io import loadmat  # type: ignore
except Exception:
    loadmat = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = [PROJECT_ROOT / "data", PROJECT_ROOT / "results"]
OUT_PATH = PROJECT_ROOT / "results" / "paper_figures" / "Fig3_raw_spectrum.png"

X_KEYWORDS = [
    "wavelength",
    "lambda",
    "distance",
    "freq",
    "frequency",
    "wavenumber",
    "index",
    "x",
]
Y_KEYWORDS_RAW = ["raw", "spectrum", "spectra", "reflection", "intensity", "amplitude", "fft"]
Y_KEYWORDS_GENERIC = ["signal", "trace", "y", "values", "magnitude", "x_total", "x_local", "x"]
LABEL_KEYWORDS = ["y_dlambda_target", "y_dlambda", "y_dlam", "label", "target", "y"]


@dataclass
class Candidate:
    path: Path
    source_type: str
    x: np.ndarray
    y: np.ndarray
    x_label: str
    y_key: str
    x_key: str
    score: float
    notes: list[str]


def norm(s: str) -> str:
    return s.strip().lower()


def score_text(path: Path) -> float:
    t = norm(path.as_posix())
    score = 0.0
    for k in ["ofdr", "raw", "reflection", "spectrum", "spectra"]:
        if k in t:
            score += 12
    for k in ["uwfbg", "dataset_b_array", "phase4a"]:
        if k in t:
            score += 9
    if "prediction" in t or "metrics" in t:
        score -= 20
    return score


def infer_x_label(x_key: str) -> str:
    k = norm(x_key)
    if "wave" in k or "lambda" in k:
        return "Wavelength (nm)"
    if "dist" in k or k == "z":
        return "Distance (m)"
    if "freq" in k:
        return "Frequency (a.u.)"
    return "Sample Index"


def pick_repr_row(y2d: np.ndarray, data_map: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
    lbl_key = None
    for k in data_map.keys():
        nk = norm(k)
        if any(lk in nk for lk in LABEL_KEYWORDS):
            if data_map[k].ndim == 1 and data_map[k].shape[0] == y2d.shape[0]:
                lbl_key = k
                break
    if lbl_key is None:
        return y2d[0], "selected row=0 (no label vector found)"

    ylbl = np.asarray(data_map[lbl_key]).reshape(-1)
    idx = int(np.argmin(np.abs(ylbl - np.median(ylbl))))
    return y2d[idx], f"selected row={idx} (median label from {lbl_key})"


def find_xy_from_dict(path: Path, data_map: dict[str, np.ndarray], source_type: str) -> Candidate | None:
    keys = list(data_map.keys())
    lkeys = [norm(k) for k in keys]

    x_key = ""
    x_arr = None
    for pref in X_KEYWORDS:
        for k, nk in zip(keys, lkeys):
            if pref in nk and data_map[k].ndim == 1:
                x_key = k
                x_arr = np.asarray(data_map[k]).reshape(-1)
                break
        if x_arr is not None:
            break

    y_candidates: list[tuple[str, np.ndarray, float]] = []
    for k, nk in zip(keys, lkeys):
        arr = np.asarray(data_map[k])
        if arr.ndim not in (1, 2):
            continue
        if arr.size < 128:
            continue
        base = 0.0
        if any(w in nk for w in Y_KEYWORDS_RAW):
            base += 45
        if nk == "x_total":
            base += 40
        elif nk == "x_local":
            base += 24
        elif nk == "x":
            base += 20
        elif any(w in nk for w in Y_KEYWORDS_GENERIC):
            base += 12
        if arr.ndim == 1:
            base += 6
        if arr.ndim == 2 and arr.shape[1] >= 256:
            base += 8
        y_candidates.append((k, arr, base))

    if not y_candidates:
        return None

    y_candidates.sort(key=lambda t: t[2], reverse=True)
    y_key, y_arr, y_score = y_candidates[0]
    notes: list[str] = []
    if y_arr.ndim == 2:
        y_vec, msg = pick_repr_row(y_arr, data_map)
        notes.append(msg)
    else:
        y_vec = y_arr.reshape(-1)
        notes.append("selected 1D spectrum directly")

    if x_arr is None or len(x_arr) != len(y_vec):
        x_arr = np.arange(len(y_vec), dtype=float)
        x_key = "index"
        notes.append("x axis fallback to sample index")
    else:
        notes.append(f"x axis from key={x_key}")

    score = score_text(path) + y_score
    if x_key != "index":
        score += 18
    if len(y_vec) > 1000:
        score += 10
    elif len(y_vec) > 500:
        score += 6

    return Candidate(
        path=path,
        source_type=source_type,
        x=np.asarray(x_arr, dtype=float),
        y=np.asarray(y_vec),
        x_label=infer_x_label(x_key),
        y_key=y_key,
        x_key=x_key,
        score=score,
        notes=notes,
    )


def load_candidate(path: Path) -> Candidate | None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".npz":
            d = np.load(path, allow_pickle=True)
            data_map = {k: np.asarray(d[k]) for k in d.files}
            return find_xy_from_dict(path, data_map, "npz")

        if suffix == ".npy":
            arr = np.load(path, allow_pickle=True)
            if isinstance(arr, np.ndarray):
                if arr.ndim == 1 and arr.size >= 128:
                    x = np.arange(arr.size, dtype=float)
                    return Candidate(path, "npy", x, arr.astype(float), "Sample Index", "array", "index", score_text(path) + 10, ["1D npy"])
                if arr.ndim == 2 and arr.shape[1] >= 128:
                    x = np.arange(arr.shape[1], dtype=float)
                    y = arr[0].astype(float)
                    return Candidate(path, "npy", x, y, "Sample Index", "array[0]", "index", score_text(path) + 8, ["2D npy row=0"])
            return None

        if suffix == ".csv":
            try:
                data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
                if getattr(data, "dtype", None) is not None and data.dtype.names and len(data.dtype.names) >= 2:
                    names = list(data.dtype.names)
                    x = np.asarray(data[names[0]], dtype=float).reshape(-1)
                    y = np.asarray(data[names[1]], dtype=float).reshape(-1)
                    if len(x) >= 128 and len(x) == len(y):
                        return Candidate(
                            path,
                            "csv",
                            x,
                            y,
                            infer_x_label(names[0]),
                            names[1],
                            names[0],
                            score_text(path) + 14,
                            [f"csv named columns {names[0]}, {names[1]}"],
                        )
            except Exception:
                pass
            arr = np.genfromtxt(path, delimiter=",", dtype=float)
            if isinstance(arr, np.ndarray):
                if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 128:
                    x = arr[:, 0].astype(float)
                    y = arr[:, 1].astype(float)
                    return Candidate(path, "csv", x, y, "Sample Index", "col1", "col0", score_text(path) + 9, ["csv numeric fallback"])
            return None

        if suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                data_map: dict[str, np.ndarray] = {}
                for k, v in obj.items():
                    try:
                        arr = np.asarray(v)
                        if arr.ndim in (1, 2):
                            data_map[k] = arr
                    except Exception:
                        pass
                if data_map:
                    return find_xy_from_dict(path, data_map, "json")
            return None

        if suffix == ".pkl":
            obj = pickle.loads(path.read_bytes())
            if isinstance(obj, dict):
                data_map = {str(k): np.asarray(v) for k, v in obj.items() if hasattr(v, "__len__")}
                if data_map:
                    return find_xy_from_dict(path, data_map, "pkl")
            return None

        if suffix == ".mat" and loadmat is not None:
            m = loadmat(path)
            data_map = {}
            for k, v in m.items():
                if k.startswith("__"):
                    continue
                arr = np.asarray(v).squeeze()
                if arr.ndim in (1, 2):
                    data_map[k] = arr
            if data_map:
                return find_xy_from_dict(path, data_map, "mat")
            return None
    except Exception:
        return None
    return None


def collect_candidates() -> list[Candidate]:
    exts = {".npz", ".npy", ".csv", ".mat", ".pkl", ".json"}
    out: list[Candidate] = []
    for root in SEARCH_DIRS:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            c = load_candidate(p)
            if c is not None and len(c.y) >= 128:
                out.append(c)
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def simple_local_peaks(y: np.ndarray) -> np.ndarray:
    if len(y) < 5:
        return np.array([], dtype=int)
    return np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1


def build_figure(candidate: Candidate) -> tuple[np.ndarray, np.ndarray, list[str]]:
    notes: list[str] = []
    x = candidate.x.astype(float).copy()
    y = candidate.y.copy()

    if np.iscomplexobj(y):
        y = np.abs(y)
        notes.append("converted complex spectrum to magnitude")
    y = np.asarray(y, dtype=float).reshape(-1)

    if np.any(~np.isfinite(y)):
        mask = np.isfinite(y)
        x = x[mask]
        y = y[mask]
        notes.append("removed non-finite points")

    if len(x) != len(y):
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        notes.append("trimmed x/y to equal length")

    if len(x) > 1 and np.any(np.diff(x) < 0):
        idx = np.argsort(x)
        x, y = x[idx], y[idx]
        notes.append("sorted by ascending x")

    # Normalize for publication readability while preserving shape.
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if y_max > y_min:
        y = (y - y_min) / (y_max - y_min)
        notes.append("min-max normalized intensity to [0, 1]")
    else:
        notes.append("kept intensity unchanged (constant signal)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dpi = 300
    fig_w, fig_h = 1400 / dpi, 900 / dpi
    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(x, y, color="black", linewidth=2.0)
    ax.set_xlabel(candidate.x_label, fontsize=11)
    ax.set_ylabel("Reflection Intensity (a.u.)", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis="both", labelsize=10, width=1.1, length=5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    if "Wavelength" in candidate.x_label:
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)

    # Minimal annotations: peak and local window.
    if len(y) > 10:
        i_peak = int(np.argmax(y))
        xp, yp = x[i_peak], y[i_peak]
        ax.annotate(
            "UWBFG peak region",
            xy=(xp, yp),
            xytext=(xp + 0.08 * (x[-1] - x[0]), 0.98),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.9),
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.2),
        )

        # Optional distorted region using secondary local peak.
        peaks = simple_local_peaks(y)
        if len(peaks) > 1:
            order = np.argsort(y[peaks])[::-1]
            p1 = int(peaks[order[0]])
            p2 = None
            for j in order[1:]:
                pj = int(peaks[j])
                if abs(pj - p1) > max(6, int(0.05 * len(y))):
                    p2 = pj
                    break
            if p2 is not None:
                ax.annotate(
                    "distorted region",
                    xy=(x[p2], y[p2]),
                    xytext=(x[p2] - 0.24 * (x[-1] - x[0]), 0.43),
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.9),
                    fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.2),
                )

        # Target local window hint around center.
        c = 0.5 * (x[0] + x[-1])
        span = 0.08 * (x[-1] - x[0])
        ax.axvspan(c - span, c + span, color="gray", alpha=0.08, linewidth=0)
        x_text = c - 0.18 * (x[-1] - x[0])
        ax.annotate(
            "target local window",
            xy=(c, 0.06),
            xytext=(x_text, 0.10),
            ha="left",
            va="center",
            fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.0),
        )

    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.20, top=0.96)
    fig.savefig(OUT_PATH, dpi=dpi)
    plt.close(fig)
    return x, y, notes


def main() -> None:
    candidates = collect_candidates()
    if not candidates:
        raise RuntimeError("No usable spectral candidates found under data/ or results/.")

    best = candidates[0]
    x, y, pre_notes = build_figure(best)

    likely_full_raw = ("raw" in norm(best.path.name) or "ofdr" in norm(best.path.as_posix())) and len(y) > 1000
    if not likely_full_raw:
        note = "No full raw OFDR sweep found; used representative local/processed spectrum."
    else:
        note = "Used near-raw OFDR spectrum candidate."

    print("Selected source file:", best.path.as_posix())
    print("Source type:", best.source_type, "| selected y key:", best.y_key, "| selected x key:", best.x_key)
    print("x/y length:", len(x), len(y))
    print("x-axis label:", best.x_label)
    print("Preprocessing:", "; ".join(best.notes + pre_notes))
    print("Raw-spectrum note:", note)
    print("Output file:", OUT_PATH.as_posix())
    print("Top candidate scores:")
    for c in candidates[:5]:
        print(f"  - {c.score:.1f} | {c.path.as_posix()} | y_key={c.y_key} | n={len(c.y)}")


if __name__ == "__main__":
    main()
