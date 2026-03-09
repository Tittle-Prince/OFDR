from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath


def main() -> None:
    # Dense system schematic needs a taller canvas than simple pipeline.
    dpi = 300
    width_px, height_px = 2400, 1000
    fig_w, fig_h = width_px / dpi, height_px / dpi

    plt.rcParams["font.family"] = "Arial"

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def add_box(x: float, y: float, w: float, h: float, text: str, fs: float = 7.2) -> None:
        ax.add_patch(Rectangle((x, y), w, h, linewidth=1.9, edgecolor="black", facecolor="white"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def add_arrow(p1: tuple[float, float], p2: tuple[float, float], lw: float = 1.5) -> None:
        ax.add_patch(
            FancyArrowPatch(
                p1,
                p2,
                arrowstyle="->",
                mutation_scale=11,
                linewidth=lw,
                color="black",
                shrinkA=0,
                shrinkB=0,
            )
        )

    # Main optical chain (left to center)
    y_main = 24
    add_box(6, 20.5, 9.5, 7, "FDML\nLaser", fs=8.2)
    add_box(17.5, 20.5, 9.5, 7, "Optical\nAmplifier", fs=8.2)
    add_box(29, 20.5, 9.5, 7, "Fiber\nSwitch", fs=8.2)
    ax.add_patch(Circle((44, y_main), 1.5, facecolor="white", edgecolor="black", linewidth=1.9))
    ax.add_patch(Arc((44, y_main), 2.0, 2.0, theta1=-50, theta2=230, linewidth=1.2, color="black"))
    add_box(42.9, 17.8, 2.2, 2.2, "", fs=7.0)
    ax.add_patch(Circle((52, y_main), 1.0, facecolor="#e5e5e5", edgecolor="black", linewidth=1.5))
    ax.text(44, 26.7, "Circulator", ha="center", va="bottom", fontsize=7.1)
    ax.text(52, 21.2, "Coupler", ha="center", va="top", fontsize=7.0)
    ax.text(45, 16.9, "Detector", ha="left", va="center", fontsize=7.0)

    # Red optical links on main trunk
    for x1, x2 in [(15.5, 17.5), (27.0, 29.0), (38.5, 42.5), (45.5, 51.0)]:
        ax.plot([x1, x2], [y_main, y_main], color="red", linewidth=1.8)

    # Small monitor waveforms (top-left)
    ax.plot([7, 7], [28.5, 34], color="black", linewidth=1.0)
    ax.plot([7, 14], [28.5, 28.5], color="black", linewidth=1.0)
    add_arrow((7, 34), (7, 34.8), lw=1.0)
    add_arrow((14, 28.5), (14.7, 28.5), lw=1.0)
    x = [7.4 + i * 0.15 for i in range(44)]
    y = [30 + 2.8 * (1 - ((xx - 10.6) / 2.3) ** 2) for xx in x]
    y = [max(29.5, yy) for yy in y]
    ax.plot(x, y, color="#1240d8", linewidth=1.5)
    ax.text(6.1, 30.7, "λ", fontsize=7.0, rotation=90)
    ax.text(11.0, 27.2, "t", fontsize=7.0)

    ax.plot([19, 19], [28.5, 34], color="black", linewidth=1.0)
    ax.plot([19, 26], [28.5, 28.5], color="black", linewidth=1.0)
    add_arrow((19, 34), (19, 34.8), lw=1.0)
    add_arrow((26, 28.5), (26.7, 28.5), lw=1.0)
    x2 = [19.4 + i * 0.12 for i in range(36)]
    y2 = [29.9 + 3.1 * (1 - ((xx - 22.0) / 2.0) ** 2) for xx in x2]
    y2 = [max(29.2, yy) for yy in y2]
    ax.plot(x2, y2, color="#1240d8", linewidth=1.4)
    ax.plot([22.8, 25.7], [32.2, 30.0], color="#8a8a8a", linestyle=":", linewidth=0.8)
    ax.add_patch(Rectangle((19.2, 29.0), 4.8, 0.9, facecolor="#d7d4ff", edgecolor="#6d6cb0", linewidth=0.6))
    ax.text(20.8, 30.1, "tPW", fontsize=6.2)
    ax.text(18.1, 30.7, "λ", fontsize=7.0, rotation=90)
    ax.text(23.8, 27.2, "t", fontsize=7.0)

    # Driver electronics blocks
    add_box(6.8, 14.3, 7.8, 4.3, "Electric\nAmplifier", fs=7.4)
    add_box(31.0, 14.5, 6.8, 4.0, "Driver", fs=7.4)
    add_arrow((10.7, 18.6), (10.7, 20.4))
    add_arrow((34.4, 18.5), (34.4, 20.4))

    # Bottom digital processing chain
    add_box(28.8, 3.4, 10.0, 4.6, "Function\nGenerator", fs=7.5)
    add_box(48.0, 4.2, 12.0, 4.8, "Analog-Digital\nConverter", fs=7.2)
    add_box(63.0, 4.4, 13.5, 4.6, "Peak Detection\nwith CNN", fs=7.3)
    add_box(80.2, 4.4, 11.5, 4.6, "Wavelength\nConversion", fs=7.3)
    add_arrow((38.8, 6.0), (48.0, 6.0))
    add_arrow((60.0, 6.6), (63.0, 6.6))
    add_arrow((76.5, 6.6), (80.2, 6.6))

    # Control and acquisition signals
    ax.plot([33.8, 33.8], [8.0, 14.5], color="black", linewidth=1.2)
    add_arrow((33.8, 14.5), (33.8, 14.9))
    ax.text(34.6, 12.2, r"$V_P$", fontsize=8)

    ax.plot([31.2, 10.5], [5.6, 5.6], color="black", linewidth=1.2)
    ax.plot([10.5, 10.5], [5.6, 14.3], color="black", linewidth=1.2)
    add_arrow((10.5, 14.3), (10.5, 14.7))
    ax.text(11.3, 12.3, r"$V_S$", fontsize=8)

    ax.plot([44.0, 44.0], [20.0, 9.0], color="black", linewidth=1.2)
    add_arrow((44.0, 9.0), (48.0, 9.0))
    ax.text(44.6, 8.9, r"$V_D$", fontsize=8)

    ax.plot([37.8, 53.8], [2.0, 2.0], color="black", linewidth=1.2)
    ax.plot([53.8, 53.8], [2.0, 4.2], color="black", linewidth=1.2)
    add_arrow((53.8, 2.2), (53.8, 4.0))
    ax.text(39.5, 3.2, r"$V_T$", fontsize=8)
    ax.text(55.9, 2.2, r"$V_R$", fontsize=8)

    # Right-side UWBFG array (red optical branches)
    branch_y = [30.0, 24.0, 18.0]
    start = (52.8, 24.0)
    for yy in branch_y:
        verts = [
            start,
            (56.0, 24.0),
            (58.0, yy),
            (62.0, yy),
        ]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="none", edgecolor="red", linewidth=1.8))
        ax.plot([62.0, 88.0], [yy, yy], color="red", linewidth=1.8)
        ax.add_patch(Arc((88.7, yy), 3.0, 3.4, theta1=-90, theta2=90, color="red", linewidth=1.8))

    fbg_labels = [
        ("FBG$_{11}$", "λ$_{B,11}$\n(Δλ$_{B,11}$)", 67.0, 30.0),
        ("FBG$_{21}$", "λ$_{B,21}$\n(Δλ$_{B,21}$)", 80.8, 30.0),
        ("FBG$_{12}$", "λ$_{B,12}$\n(Δλ$_{B,12}$)", 67.0, 24.0),
        ("FBG$_{22}$", "λ$_{B,22}$\n(Δλ$_{B,22}$)", 80.8, 24.0),
        ("FBG$_{13}$", "λ$_{B,13}$\n(Δλ$_{B,13}$)", 67.0, 18.0),
        ("FBG$_{23}$", "λ$_{B,23}$\n(Δλ$_{B,23}$)", 80.8, 18.0),
    ]
    for name, val, x, y in fbg_labels:
        ax.add_patch(Rectangle((x - 1.25, y - 0.7), 2.5, 1.4, linewidth=1.2, edgecolor="black", facecolor="white"))
        ax.text(x, y + 1.1, name, ha="center", va="bottom", fontsize=7.0)
        ax.text(x, y - 2.0, val, ha="center", va="center", fontsize=6.6)

    # Multiplexing annotations
    add_arrow((57.0, 35.8), (91.0, 35.8), lw=1.0)
    add_arrow((91.0, 35.8), (57.0, 35.8), lw=1.0)
    ax.text(74.0, 36.8, "Different-Wavelength Multiplexing", ha="center", fontsize=8.0, fontstyle="italic")
    add_arrow((92.5, 14.2), (92.5, 34.5), lw=1.0)
    add_arrow((92.5, 34.5), (92.5, 14.2), lw=1.0)
    ax.text(
        95.2,
        24.4,
        "Same Wavelength Multiplexing\nwith Different FWHMs",
        rotation=-90,
        ha="center",
        va="center",
        fontsize=7.5,
        fontstyle="italic",
    )

    out_path = Path("results/paper_figures/Fig1_system_overview.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
