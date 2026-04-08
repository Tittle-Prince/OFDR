from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Rectangle
import numpy as np


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9.5,
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
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


def draw_block(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fontsize: float = 8.9,
    lw: float = 0.95,
) -> Rectangle:
    rect = Rectangle((x, y), w, h, facecolor="white", edgecolor="black", linewidth=lw)
    ax.add_patch(rect)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.15,
    )
    return rect


def draw_arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, *, lw: float = 0.9) -> None:
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="->",
        mutation_scale=8.0,
        linewidth=lw,
        color="black",
        shrinkA=0.0,
        shrinkB=0.0,
    )
    ax.add_patch(arrow)


def add_array_icon(ax: plt.Axes, rect: Rectangle) -> None:
    x, y = rect.get_xy()
    w, h = rect.get_width(), rect.get_height()
    y_mid = y + 0.25 * h
    x0 = x + 0.10 * w
    x1 = x + 0.90 * w
    ax.plot([x0, x1], [y_mid, y_mid], color="black", linewidth=0.8)
    tick_x = np.linspace(x + 0.16 * w, x + 0.84 * w, 10)
    for tx in tick_x:
        ax.plot([tx, tx], [y_mid - 0.04 * h, y_mid + 0.07 * h], color="black", linewidth=0.75)


def add_unit_icon(ax: plt.Axes, rect: Rectangle) -> None:
    x, y = rect.get_xy()
    w, h = rect.get_width(), rect.get_height()
    xx = np.linspace(x + 0.12 * w, x + 0.88 * w, 120)
    yy = y + 0.26 * h + 0.055 * h * np.sin(np.linspace(0, 4 * np.pi, 120))
    ax.plot(xx, yy, color="black", linewidth=0.8)
    ax.plot([x + 0.12 * w, x + 0.12 * w], [y + 0.17 * h, y + 0.35 * h], color="black", linewidth=0.8)


def add_fft_icon(ax: plt.Axes, rect: Rectangle) -> None:
    x, y = rect.get_xy()
    w, h = rect.get_width(), rect.get_height()
    bars = np.array([0.10, 0.28, 0.55, 0.82, 0.44, 0.18])
    xs = np.linspace(x + 0.14 * w, x + 0.86 * w, len(bars))
    for bx, bh in zip(xs, bars):
        ax.plot([bx, bx], [y + 0.16 * h, y + (0.16 + 0.26 * bh) * h], color="black", linewidth=1.0)


def add_window_icon(ax: plt.Axes, rect: Rectangle) -> None:
    x, y = rect.get_xy()
    w, h = rect.get_width(), rect.get_height()
    xx = np.linspace(x + 0.10 * w, x + 0.90 * w, 200)
    curve = (
        0.20
        + 0.48 * np.exp(-0.5 * ((xx - (x + 0.47 * w)) / (0.11 * w)) ** 2)
        + 0.15 * np.exp(-0.5 * ((xx - (x + 0.61 * w)) / (0.055 * w)) ** 2)
    )
    ax.plot(xx, y + curve * h, color="black", linewidth=0.9)
    ax.plot([x + 0.28 * w, x + 0.28 * w], [y + 0.15 * h, y + 0.80 * h], color="black", linestyle="--", linewidth=0.7)
    ax.plot([x + 0.72 * w, x + 0.72 * w], [y + 0.15 * h, y + 0.80 * h], color="black", linestyle="--", linewidth=0.7)


def add_actual_cnn_icon(ax: plt.Axes, rect: Rectangle) -> None:
    x, y = rect.get_xy()
    w, h = rect.get_width(), rect.get_height()
    cols = [
        (x + 0.15 * w, "1×512"),
        (x + 0.35 * w, "32"),
        (x + 0.53 * w, "64"),
        (x + 0.71 * w, "128"),
        (x + 0.88 * w, "1"),
    ]
    heights = [0.16, 0.34, 0.46, 0.60, 0.12]
    widths = [0.035, 0.045, 0.045, 0.045, 0.035]
    for (cx, label), hh, ww in zip(cols, heights, widths):
        bar = Rectangle(
            (cx - ww * w / 2, y + 0.18 * h),
            ww * w,
            hh * h,
            facecolor="white",
            edgecolor="black",
            linewidth=0.75,
        )
        ax.add_patch(bar)
        ax.text(cx, y + 0.10 * h, label, ha="center", va="center", fontsize=7.1)
    for i in range(len(cols) - 1):
        draw_arrow(ax, cols[i][0] + 0.035 * w, y + 0.48 * h, cols[i + 1][0] - 0.035 * w, y + 0.48 * h, lw=0.65)


def add_output_icon(ax: plt.Axes, rect: Rectangle) -> None:
    x, y = rect.get_xy()
    w, h = rect.get_width(), rect.get_height()
    xx = np.linspace(x + 0.14 * w, x + 0.86 * w, 120)
    yy = y + 0.25 * h + 0.42 * h / (1 + np.exp(-(xx - (x + 0.50 * w)) / (0.08 * w)))
    ax.plot(xx, yy, color="black", linewidth=0.9)


def add_zoom_spectra(ax: plt.Axes, fig: plt.Figure, anchor_rect: Rectangle) -> None:
    inset = fig.add_axes([0.48, 0.57, 0.24, 0.25])
    x = np.linspace(0.0, 1.0, 700)
    base = 0.06 + 0.012 * np.sin(2 * np.pi * 2.1 * x + 0.4)
    target = 0.92 * np.exp(-0.5 * ((x - 0.50) / 0.072) ** 2)
    left_neighbor = 0.38 * np.exp(-0.5 * ((x - 0.42) / 0.050) ** 2)
    right_neighbor = 0.30 * np.exp(-0.5 * ((x - 0.60) / 0.062) ** 2)
    asym = 0.11 * np.exp(-0.5 * ((x - 0.57) / 0.11) ** 2)
    ripple = 0.025 * np.sin(2 * np.pi * 10.5 * x + 0.7)
    spike = 0.10 * np.exp(-0.5 * ((x - 0.70) / 0.012) ** 2)
    y = base + target + left_neighbor + right_neighbor + asym + ripple + spike
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)

    inset.plot(x, y, color="black", linewidth=0.95)
    inset.set_xlim(0.0, 1.0)
    inset.set_ylim(-0.02, 1.05)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_linewidth(0.85)
    inset.text(0.05, 0.93, "Local overlapped spectra", transform=inset.transAxes, fontsize=8.1, ha="left", va="top")
    inset.annotate(
        "neighbor shift",
        xy=(0.42, 0.68),
        xytext=(0.06, 0.62),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=0.65, color="black"),
        fontsize=7.7,
    )
    inset.annotate(
        "linewidth\nasymmetry",
        xy=(0.56, 0.88),
        xytext=(0.43, 0.83),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=0.65, color="black"),
        fontsize=7.7,
        ha="center",
    )
    inset.annotate(
        "system artifact",
        xy=(0.69, 0.66),
        xytext=(0.70, 0.28),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=0.65, color="black"),
        fontsize=7.7,
    )

    con = ConnectionPatch(
        xyA=(0.18, 0.00),
        coordsA=inset.transAxes,
        xyB=(anchor_rect.get_x() + 0.58 * anchor_rect.get_width(), anchor_rect.get_y() + anchor_rect.get_height()),
        coordsB=ax.transData,
        linestyle="--",
        linewidth=0.65,
        color="black",
        alpha=0.8,
    )
    fig.add_artist(con)


def add_loss_panel(ax: plt.Axes, cnn_rect: Rectangle, output_rect: Rectangle) -> Rectangle:
    loss_rect = draw_block(
        ax,
        output_rect.get_x() + output_rect.get_width() + 0.02,
        output_rect.get_y(),
        0.18,
        output_rect.get_height(),
        "Training loss\nTail-aware L1\n+ hard weighting",
        fontsize=8.7,
        lw=1.0,
    )
    ax.text(
        loss_rect.get_x() + loss_rect.get_width() / 2.0,
        loss_rect.get_y() - 0.028,
        r"$L=|e|+\lambda_{tail}\max(0,|e|-\tau)^2$" "\n" r"$\tau=0.01,\ \lambda_{tail}=3.0$",
        ha="center",
        va="top",
        fontsize=7.5,
    )
    draw_arrow(
        ax,
        cnn_rect.get_x() + cnn_rect.get_width(),
        cnn_rect.get_y() + 0.70 * cnn_rect.get_height(),
        loss_rect.get_x(),
        loss_rect.get_y() + 0.70 * loss_rect.get_height(),
        lw=0.85,
    )
    return loss_rect


def main() -> None:
    configure_style()

    repo_root = Path(__file__).resolve().parents[3]
    figure_root = repo_root / "paper_figures" / "Fig_system_overview_ofdr_cnn_actual"
    _, outputs_dir, data_dir = ensure_dirs(figure_root)

    fig = plt.figure(figsize=(15.2, 4.8))
    ax = fig.add_axes([0.025, 0.08, 0.95, 0.84])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    y = 0.58
    h = 0.24
    blocks = [
        (0.01, 0.10, "Integrated\nOFDR unit"),
        (0.14, 0.17, "Identical UWFBG array\n(100 m, 10 gratings,\nlow reflectivity)"),
        (0.34, 0.10, "Backscattered\nsignal"),
        (0.47, 0.14, "Interferometric\nsignal acquisition"),
        (0.64, 0.13, "FFT / distance-domain\nprocessing"),
        (0.80, 0.15, "Grating localization\n+ local spectral\nwindow extraction"),
    ]

    rects: list[Rectangle] = []
    for x, w, text in blocks:
        rects.append(draw_block(ax, x, y, w, h, text))

    cnn_rect = draw_block(
        ax,
        0.18,
        0.11,
        0.34,
        0.21,
        "Actual baseline CNN\nInput: local spectrum (1 × 512)\nConv1d(32,k7)+ReLU+MaxPool\nConv1d(64,k5)+ReLU+MaxPool\nConv1d(128,k5)+ReLU\nFlatten → FC(128) → FC(1)",
        fontsize=8.5,
        lw=1.0,
    )
    output_rect = draw_block(
        ax,
        0.57,
        0.11,
        0.12,
        0.21,
        "Output:\nΔλ /\nTemperature",
        fontsize=9.0,
        lw=1.0,
    )
    loss_rect = add_loss_panel(ax, cnn_rect, output_rect)

    add_unit_icon(ax, rects[0])
    add_array_icon(ax, rects[1])
    add_fft_icon(ax, rects[4])
    add_window_icon(ax, rects[5])
    add_actual_cnn_icon(ax, cnn_rect)
    add_output_icon(ax, output_rect)

    for r0, r1 in zip(rects[:-1], rects[1:]):
        draw_arrow(ax, r0.get_x() + r0.get_width(), y + h / 2.0, r1.get_x(), y + h / 2.0)

    draw_arrow(
        ax,
        rects[5].get_x() + 0.42 * rects[5].get_width(),
        rects[5].get_y(),
        cnn_rect.get_x() + 0.12 * cnn_rect.get_width(),
        cnn_rect.get_y() + cnn_rect.get_height(),
        lw=0.9,
    )
    draw_arrow(
        ax,
        cnn_rect.get_x() + cnn_rect.get_width(),
        cnn_rect.get_y() + 0.34 * cnn_rect.get_height(),
        output_rect.get_x(),
        output_rect.get_y() + 0.50 * output_rect.get_height(),
        lw=0.9,
    )

    add_zoom_spectra(ax, fig, rects[5])

    ax.text(
        rects[1].get_x() + rects[1].get_width() / 2.0,
        y - 0.05,
        "Physical sensing front-end: identical UWFBG array",
        ha="center",
        va="top",
        fontsize=8.2,
    )
    ax.text(
        cnn_rect.get_x() + cnn_rect.get_width() / 2.0,
        cnn_rect.get_y() - 0.04,
        "CNN-based demodulation used in the current project",
        ha="center",
        va="top",
        fontsize=8.2,
    )
    ax.text(
        loss_rect.get_x() + loss_rect.get_width() / 2.0,
        loss_rect.get_y() + loss_rect.get_height() + 0.03,
        "Training only",
        ha="center",
        va="bottom",
        fontsize=7.9,
    )

    spec_text = """Actual-consistent OFDR-CNN system overview

Key corrections vs simplified draft:
1. Local spectrum input length is 512, not 256.
2. Baseline CNN architecture is:
   Conv1d(1->32,k7)+ReLU+MaxPool
   Conv1d(32->64,k5)+ReLU+MaxPool
   Conv1d(64->128,k5)+ReLU
   Flatten -> Linear(128 hidden) -> Linear(1)
3. Tail-aware loss is a training objective, not an inference block.
4. Main tail-aware config:
   tau = 0.01
   lambda_tail = 3.0
   hard weighting tau = 0.01, alpha = 1.5

Sources:
- src/ofdr/models/phase3_cnn.py
- src/ofdr/training/phase3_train_utils.py
- config/phase4a_shift004_linewidth_l3_method_tailaware_hard.yaml
- Real/stage1_three_grating_ideal_ofdr_bridge.py
"""
    (data_dir / "diagram_spec.txt").write_text(spec_text, encoding="utf-8")

    png_path = outputs_dir / "ofdr_cnn_system_diagram_actual.png"
    pdf_path = outputs_dir / "ofdr_cnn_system_diagram_actual.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
