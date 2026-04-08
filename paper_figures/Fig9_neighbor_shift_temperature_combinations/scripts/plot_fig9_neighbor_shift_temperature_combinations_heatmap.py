from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.9,
        }
    )


def ensure_dirs(root: Path) -> tuple[Path, Path, Path]:
    scripts_dir = root / "scripts"
    outputs_dir = root / "outputs"
    data_dir = root / "data_copy"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return scripts_dir, outputs_dir, data_dir


def load_temperature_combinations(repo_root: Path) -> tuple[pd.DataFrame, str]:
    candidates = [
        repo_root / "neighbor_shift_temperature_combinations_full.csv",
        repo_root
        / "paper_figures"
        / "Fig9_neighbor_shift_result"
        / "data_copy"
        / "neighbor_shift_temperature_combinations.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path), str(path)
    raise FileNotFoundError("Missing neighbor_shift temperature combination file.")


def export_data_copy(df: pd.DataFrame, data_dir: Path) -> Path:
    out_path = data_dir / "neighbor_shift_temperature_combinations.csv"
    df.to_csv(out_path, index=False)
    return out_path


def build_heatmap_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.vstack(
        [
            df["temp_left"].to_numpy(dtype=np.float64),
            df["temp_target"].to_numpy(dtype=np.float64),
            df["temp_right"].to_numpy(dtype=np.float64),
        ]
    )


def plot_temperature_heatmap(ax: plt.Axes, matrix: np.ndarray, exp_idx: np.ndarray) -> plt.AxesImage:
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="Greys",
        interpolation="nearest",
        origin="upper",
    )

    ax.set_xlabel("Experiment index")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Left neighbor", "Target", "Right neighbor"])

    tick_step = 10 if exp_idx.size > 40 else 5
    tick_positions = np.arange(0, exp_idx.size, tick_step, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(int(exp_idx[i])) for i in tick_positions])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Light separators between the three temperature-control channels.
    ax.hlines([0.5, 1.5], xmin=-0.5, xmax=exp_idx.size - 0.5, colors="white", linewidth=0.8, alpha=0.85)
    return image


def add_row_summaries(ax: plt.Axes, matrix: np.ndarray) -> None:
    labels = ["L", "T", "R"]
    for row, label in enumerate(labels):
        row_min = np.min(matrix[row])
        row_max = np.max(matrix[row])
        ax.text(
            1.005,
            (2.5 - row) / 3.0,
            f"{label}: {row_min:.1f}–{row_max:.1f} °C",
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=8.8,
            color="#2f2f2f",
        )


def main() -> None:
    configure_style()

    repo_root = Path(__file__).resolve().parents[3]
    figure_root = repo_root / "paper_figures" / "Fig9_neighbor_shift_temperature_combinations"
    _, outputs_dir, data_dir = ensure_dirs(figure_root)

    df, source_path = load_temperature_combinations(repo_root)
    data_copy_path = export_data_copy(df, data_dir)

    exp_idx = df["exp_idx"].to_numpy(dtype=np.int32)
    matrix = build_heatmap_matrix(df)

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 2.9), constrained_layout=True)
    image = plot_temperature_heatmap(ax, matrix, exp_idx)
    add_row_summaries(ax, matrix)

    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Temperature (°C)")

    png_path = outputs_dir / "fig9_neighbor_shift_temperature_combinations_heatmap.png"
    pdf_path = outputs_dir / "fig9_neighbor_shift_temperature_combinations_heatmap.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Temperature combination source: {source_path}")
    print(f"Data copy saved to: {data_copy_path}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
