from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Rectangle
import numpy as np


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
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


def draw_block(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, *, fontsize: float = 9.3, lw: float = 0.95) -> Rectangle:
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
        mutation_scale=8.5,
        linewidth=lw,
        color="black",
        shrinkA=0.0,
        shrinkB=0.0,
    )
    ax.add_patch(arrow)


def add_array_icon(ax: plt.Axes, rect: Rectangle) -> None:
    x, y = rect.get_xy()
    w, h = rect.get_width(), rect.get_height()
    y_mid = y + 0.27 * h
    x0 = x + 0.10 * w
    x1 = x + 0.90 * w
    ax.plot([x0, x1], [y_mid, y_mid], color="black", linewidth=0.8)
    tick_x = np.linspace(x + 0.16 * w, x + 0.84 * w, 10)
    for tx in tick_x:
        ax.plot([tx, tx], [y_mid - 0.045 * h, y_mid + 0.075 * h], color="black", linewidth=0.75)


def add_unit_icon(ax: plt.Axes, rect: Rectangle) -> None:
    x, y = rect.get_xy()
    w, h = rect.get_width(), rect.get_height()
    xx = np.linspace(x + 0.12 * w, x + 0.88 * w, 120)
    yy = y + 0.27 * h + 0.055 * h * np.sin(np.linspace(0, 4 * np.pi, 120))
    ax.plot(xx, yy, color="black", linewidth=0.8)
    ax.plot([x + 0.12 * w, x + 0.12 * w], [y + 0.18 * h, y + 0.36 * h], color="black", linewidth=0.8)


def add_fft_icon(ax: plt.Axes, rect: Rectangle) -> None:
    x, y = rect.get_xy()
    w, h = rect.get_width(), rect.get_height()
    bars = np.array([0.10, 0.24, 0.48, 0.78, 0.42, 0.18])
    xs = np.linspace(x + 0.13 * w, x + 0.87 * w, len(bars))
    for bx, bh in zip(xs, bars):
        ax.plot([bx, bx], [y + 0.16 * h, y + (0.16 + 0.28 * bh) * h], color="black", linewidth=1.0)


def add_window_icon(ax: plt.Axes, rect: Rectangle) -> None:
    x, y = rect.get_xy()
    w, h = rect.get_width(), rect.get_height()
    xx = np.linspace(x + 0.10 * w, x + 0.90 * w, 160)
    curve = (
        0.22
        + 0.55 * np.exp(-0.5 * ((xx - (x + 0.47 * w)) / (0.10 * w)) ** 2)
        + 0.18 * np.exp(-0.5 * ((xx - (x + 0.60 * w)) / (0.05 * w)) ** 2)
    )
    ax.plot(xx, y + curve * h, color="black", linewidth=0.9)
    ax.plot([x + 0.33 * w, x + 0.33 * w], [y + 0.15 * h, y + 0.82 * h], color="black", linestyle="--", linewidth=0.7)
    ax.plot([x + 0.69 * w, x + 0.69 * w], [y + 0.15 * h, y + 0.82 * h], color="black", linestyle="--", linewidth=0.7)


def add_cnn_icon(ax: plt.Axes, rect: Rectangle) -> None:
    x, y = rect.get_xy()
    w, h = rect.get_width(), rect.get_height()
    left = [(x + 0.18 * w, y + 0.25 * h + i * 0.11 * h) for i in range(4)]
    mid = [(x + 0.48 * w, y + 0.30 * h + i * 0.15 * h) for i in range(3)]
    right = [(x + 0.78 * w, y + 0.42 * h)]
    for p in left:
        circ = plt.Circle(p, 0.013 * h, fill=False, color="black", linewidth=0.7)
        ax.add_patch(circ)
    for p in mid:
        circ = plt.Circle(p, 0.013 * h, fill=False, color="black", linewidth=0.7)
        ax.add_patch(circ)
    for p in right:
        circ = plt.Circle(p, 0.015 * h, fill=False, color="black", linewidth=0.85)
        ax.add_patch(circ)
    for p1 in left:
        for p2 in mid:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black", linewidth=0.35, alpha=0.8)
    for p1 in mid:
        for p2 in right:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black", linewidth=0.35, alpha=0.8)


def add_output_icon(ax: plt.Axes, rect: Rectangle) -> None:
    x, y = rect.get_xy()
    w, h = rect.get_width(), rect.get_height()
    xx = np.linspace(x + 0.14 * w, x + 0.86 * w, 120)
    yy = y + 0.25 * h + 0.45 * h / (1 + np.exp(-(xx - (x + 0.50 * w)) / (0.08 * w)))
    ax.plot(xx, yy, color="black", linewidth=0.9)


def add_zoom_spectra(ax: plt.Axes, fig: plt.Figure, anchor_rect: Rectangle) -> None:
    inset = fig.add_axes([0.515, 0.63, 0.28, 0.23])
    x = np.linspace(0.0, 1.0, 600)

    base = 0.06 + 0.015 * np.sin(2 * np.pi * 2.1 * x + 0.4)
    target = 0.95 * np.exp(-0.5 * ((x - 0.50) / 0.07) ** 2)
    left_neighbor = 0.42 * np.exp(-0.5 * ((x - 0.43) / 0.055) ** 2)
    right_neighbor = 0.34 * np.exp(-0.5 * ((x - 0.59) / 0.060) ** 2)
    asym = 0.10 * np.exp(-0.5 * ((x - 0.56) / 0.11) ** 2)
    ripple = 0.028 * np.sin(2 * np.pi * 11.0 * x + 0.7)
    spike = 0.11 * np.exp(-0.5 * ((x - 0.69) / 0.012) ** 2)

    y = base + target + left_neighbor + right_neighbor + asym + ripple + spike
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)

    inset.plot(x, y, color="black", linewidth=1.0)
    inset.set_xlim(0, 1)
    inset.set_ylim(-0.02, 1.05)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_linewidth(0.9)

    inset.text(0.05, 0.93, "Zoom-in: distorted local spectra", transform=inset.transAxes, fontsize=8.7, ha="left", va="top")
    inset.annotate(
        "neighbor shift",
        xy=(0.42, 0.68),
        xytext=(0.08, 0.62),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=0.7, color="black"),
        fontsize=8.0,
    )
    inset.annotate(
        "linewidth asymmetry",
        xy=(0.55, 0.88),
        xytext=(0.44, 0.93),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=0.7, color="black"),
        fontsize=8.0,
        ha="center",
    )
    inset.annotate(
        "system artifact",
        xy=(0.69, 0.66),
        xytext=(0.73, 0.28),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=0.7, color="black"),
        fontsize=8.0,
    )

    con = ConnectionPatch(
        xyA=(0.20, 0.00),
        coordsA=inset.transAxes,
        xyB=(anchor_rect.get_x() + 0.65 * anchor_rect.get_width(), anchor_rect.get_y() + anchor_rect.get_height()),
        coordsB=ax.transData,
        linestyle="--",
        linewidth=0.7,
        color="black",
        alpha=0.8,
    )
    fig.add_artist(con)


def main() -> None:
    configure_style()

    repo_root = Path(__file__).resolve().parents[3]
    figure_root = repo_root / "paper_figures" / "Fig_system_overview_ofdr_cnn"
    _, outputs_dir, data_dir = ensure_dirs(figure_root)

    fig = plt.figure(figsize=(15.0, 4.6))
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.84])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    y = 0.32
    h = 0.24
    blocks = [
        (0.02, 0.11, "Integrated\nOFDR Unit"),
        (0.16, 0.16, "Identical UWFBG Array\n(100 m, 10 gratings,\nlow reflectivity)"),
        (0.36, 0.11, "Backscattered\nSignal"),
        (0.50, 0.14, "Interferometric Signal\nAcquisition"),
        (0.68, 0.13, "FFT / Distance-Domain\nProcessing"),
        (0.84, 0.13, "Local Spectral Window\nExtraction"),
    ]

    rects: list[Rectangle] = []
    for x, w, text in blocks:
        rects.append(draw_block(ax, x, y, w, h, text))

    cnn_rect = draw_block(ax, 0.16, 0.03, 0.23, 0.18, "CNN-based demodulation\n1D CNN + Tail-aware Loss", fontsize=9.4, lw=1.0)
    out_rect = draw_block(ax, 0.48, 0.03, 0.16, 0.18, "Output:\nΔλ / Temperature", fontsize=9.5, lw=1.0)

    add_unit_icon(ax, rects[0])
    add_array_icon(ax, rects[1])
    add_fft_icon(ax, rects[4])
    add_window_icon(ax, rects[5])
    add_cnn_icon(ax, cnn_rect)
    add_output_icon(ax, out_rect)

    for r0, r1 in zip(rects[:-1], rects[1:]):
        draw_arrow(
            ax,
            r0.get_x() + r0.get_width(),
            y + h / 2.0,
            r1.get_x(),
            y + h / 2.0,
        )

    draw_arrow(ax, rects[5].get_x() + 0.45 * rects[5].get_width(), y, cnn_rect.get_x() + 0.12 * cnn_rect.get_width(), cnn_rect.get_y() + cnn_rect.get_height(), lw=0.9)
    draw_arrow(ax, cnn_rect.get_x() + cnn_rect.get_width(), cnn_rect.get_y() + cnn_rect.get_height() / 2.0, out_rect.get_x(), out_rect.get_y() + out_rect.get_height() / 2.0, lw=0.9)

    ax.text(rects[1].get_x() + rects[1].get_width() / 2.0, y - 0.05, "emphasis: identical UWFBG", ha="center", va="top", fontsize=8.6)
    ax.text(cnn_rect.get_x() + cnn_rect.get_width() / 2.0, cnn_rect.get_y() - 0.03, "emphasis: CNN-based demodulation", ha="center", va="top", fontsize=8.6)

    add_zoom_spectra(ax, fig, rects[5])

    spec_text = """Figure type: SCI-style OFDR system overview diagram
Style: black and white, thin arrows, rectangular modules
Flow:
Integrated OFDR Unit -> Identical UWFBG Array -> Backscattered Signal ->
Interferometric Signal Acquisition -> FFT / Distance-Domain Processing ->
Local Spectral Window Extraction -> 1D CNN + Tail-aware Loss -> Output
Zoom-in annotation terms:
- neighbor shift
- linewidth asymmetry
- system artifact
"""
    (data_dir / "diagram_spec.txt").write_text(spec_text, encoding="utf-8")

    png_path = outputs_dir / "ofdr_cnn_system_diagram.png"
    pdf_path = outputs_dir / "ofdr_cnn_system_diagram.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
