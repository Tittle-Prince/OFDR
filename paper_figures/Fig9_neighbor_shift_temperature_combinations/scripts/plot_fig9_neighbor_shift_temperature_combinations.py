
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CHANNELS = [
    ("temp_left", "Left neighbor", "#6f6f6f", "s"),
    ("temp_target", "Target", "#1f4e79", "o"),
    ("temp_right", "Right neighbor", "#4f4f4f", "^"),
]


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
        repo_root / "paper_figures" / "Fig9_neighbor_shift_result" / "data_copy" / "neighbor_shift_temperature_combinations.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path), str(path)
    raise FileNotFoundError("Missing neighbor_shift temperature combination file.")



def export_data_copy(df: pd.DataFrame, data_dir: Path) -> Path:
    out_path = data_dir / "neighbor_shift_temperature_combinations.csv"
    df.to_csv(out_path, index=False)
    return out_path



def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", alpha=0.16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)



def plot_temperature_combinations(fig: plt.Figure, axes: list[plt.Axes], df: pd.DataFrame) -> None:
    x = df["exp_idx"].to_numpy()
    temp_cols = [df[key].to_numpy() for key, *_ in CHANNELS]
    ymin = min(arr.min() for arr in temp_cols)
    ymax = max(arr.max() for arr in temp_cols)
    margin = 0.55

    for ax, (key, label, color, marker) in zip(axes, CHANNELS):
        y = df[key].to_numpy()
        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.1,
            marker=marker,
            markersize=3.0,
            markerfacecolor="white" if label != "Target" else color,
            markeredgecolor=color,
            markeredgewidth=0.8 if label != "Target" else 0.0,
            markevery=8,
        )
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_xlim(1, int(x.max()))
        ax.set_ylabel("Temp. (?C)")
        style_axis(ax)
        ax.text(
            0.01,
            0.84,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=9.3,
            color=color,
        )
        ax.text(
            0.99,
            0.84,
            f"{y.min():.1f}?{y.max():.1f} ?C",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=8.7,
            color="#444444",
        )

    axes[0].set_title("Neighbor-shift temperature combinations", loc="left", pad=4)
    axes[-1].set_xlabel("Experiment index")
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    fig.text(0.012, 0.5, "Temperature control channels", rotation=90, va="center", ha="center", fontsize=10)



def main() -> None:
    configure_style()
    repo_root = Path(__file__).resolve().parents[3]
    figure_root = repo_root / "paper_figures" / "Fig9_neighbor_shift_temperature_combinations"
    _, outputs_dir, data_dir = ensure_dirs(figure_root)

    df, source_path = load_temperature_combinations(repo_root)
    data_copy_path = export_data_copy(df, data_dir)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.4, 5.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"hspace": 0.02},
    )
    plot_temperature_combinations(fig, list(axes), df)

    png_path = outputs_dir / "fig9_neighbor_shift_temperature_combinations.png"
    pdf_path = outputs_dir / "fig9_neighbor_shift_temperature_combinations.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Temperature combination source: {source_path}")
    print(f"Data copy saved to: {data_copy_path}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
