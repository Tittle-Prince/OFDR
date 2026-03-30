from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class AlignmentDiagnosisConfig:
    dataset_path: str = "data/processed/dataset_c_phase4a_shift004_linewidth_l3.npz"
    bridge_path: str = "Real/stage1_three_grating_ideal_ofdr_bridge_outputs/bridge_model_ready_examples.npz"
    output_dir: str = "Real/stage1_bridge_phase4a_alignment_outputs"
    label_pool_halfwidth_nm: float = 0.03


def _sample_features(x: np.ndarray) -> np.ndarray:
    idx = np.arange(x.size, dtype=np.float64)
    weight = np.sum(x) + 1e-12
    center = np.sum(idx * x) / weight
    spread = np.sqrt(np.sum(((idx - center) ** 2) * x) / weight)
    return np.array(
        [
            float(np.mean(x)),
            float(np.std(x)),
            float(np.quantile(x, 0.90)),
            float(np.mean(x > 0.50)),
            float(center),
            float(spread),
        ],
        dtype=np.float64,
    )


def load_inputs(repo_root: Path, cfg: AlignmentDiagnosisConfig) -> dict[str, np.ndarray]:
    data = np.load(repo_root / cfg.dataset_path)
    bridge = np.load(repo_root / cfg.bridge_path)

    x_local = data["X_local"].astype(np.float64)
    y_nm = data["Y_dlambda_target"].astype(np.float64)
    wavelengths_nm = data["wavelengths"].astype(np.float64)

    x_bridge = bridge["X_bridge"][:, 0, :].astype(np.float64)
    y_bridge_pm = bridge["Y_delta_target_pm"].astype(np.float64)
    y_bridge_nm = y_bridge_pm * 1e-3

    return {
        "X_local": x_local,
        "Y_nm": y_nm,
        "wavelengths_nm": wavelengths_nm,
        "X_bridge": x_bridge,
        "Y_bridge_pm": y_bridge_pm,
        "Y_bridge_nm": y_bridge_nm,
    }


def find_nearest_real_samples(
    x_local: np.ndarray,
    y_nm: np.ndarray,
    x_bridge: np.ndarray,
    y_bridge_nm: np.ndarray,
    halfwidth_nm: float,
) -> list[dict[str, float | int]]:
    matches: list[dict[str, float | int]] = []
    for i in range(x_bridge.shape[0]):
        target_nm = float(y_bridge_nm[i])
        mask = np.abs(y_nm - target_nm) <= halfwidth_nm
        candidate_idx = np.flatnonzero(mask)
        if candidate_idx.size == 0:
            candidate_idx = np.arange(x_local.shape[0])

        diffs = x_local[candidate_idx] - x_bridge[i][None, :]
        mse = np.mean(diffs**2, axis=1)
        best_local = int(np.argmin(mse))
        best_idx = int(candidate_idx[best_local])
        matches.append(
            {
                "bridge_index": i,
                "real_index": best_idx,
                "target_delta_nm": target_nm,
                "real_delta_nm": float(y_nm[best_idx]),
                "mse": float(mse[best_local]),
            }
        )
    return matches


def plot_distribution_alignment(
    x_local: np.ndarray,
    x_bridge: np.ndarray,
    output_path: Path,
) -> None:
    real_feats = np.stack([_sample_features(x) for x in x_local], axis=0)
    bridge_feats = np.stack([_sample_features(x) for x in x_bridge], axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)

    panels = [
        (0, "Mean intensity", real_feats[:, 0], bridge_feats[:, 0]),
        (1, "Std intensity", real_feats[:, 1], bridge_feats[:, 1]),
        (3, "Fraction above 0.5", real_feats[:, 3], bridge_feats[:, 3]),
        (5, "Energy center index", real_feats[:, 4], bridge_feats[:, 4]),
    ]
    for ax, (feat_idx, title, real_values, bridge_values) in zip(axes.ravel(), panels):
        ax.hist(real_values, bins=50, density=True, color="#d9d9d9", edgecolor="#7f7f7f", alpha=0.95)
        for j, val in enumerate(bridge_values):
            ax.axvline(val, color="#1f77b4", linewidth=1.4, alpha=0.85)
            ax.text(val, ax.get_ylim()[1] * (0.92 - 0.08 * min(j, 3)), f"B{j+1}", rotation=90, va="top", ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_overlay_matches(
    wavelengths_nm: np.ndarray,
    x_local: np.ndarray,
    y_nm: np.ndarray,
    x_bridge: np.ndarray,
    y_bridge_pm: np.ndarray,
    matches: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 10.0), constrained_layout=True, sharex=True, sharey=True)
    flat_axes = axes.ravel()

    for ax, match in zip(flat_axes, matches):
        b_idx = int(match["bridge_index"])
        r_idx = int(match["real_index"])
        ax.plot(wavelengths_nm, x_local[r_idx], color="#7f7f7f", linestyle="--", linewidth=1.8, label="Nearest real phase4a")
        ax.plot(wavelengths_nm, x_bridge[b_idx], color="#1f77b4", linewidth=1.8, label="Bridge sample")
        ax.set_title(
            f"Bridge {b_idx + 1} | target={y_bridge_pm[b_idx]:+.1f} pm | "
            f"nearest real={y_nm[r_idx] * 1e3:+.1f} pm\nMSE={float(match['mse']):.4e}",
            fontsize=10,
        )
        ax.grid(True, alpha=0.25)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    for ax in axes[-1, :]:
        ax.set_xlabel("Wavelength (nm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized intensity")

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_summary_text(
    repo_root: Path,
    cfg: AlignmentDiagnosisConfig,
    matches: list[dict[str, float | int]],
    x_local: np.ndarray,
    x_bridge: np.ndarray,
) -> Path:
    out_path = repo_root / cfg.output_dir / "alignment_summary.txt"
    real_feats = np.stack([_sample_features(x) for x in x_local], axis=0)
    bridge_feats = np.stack([_sample_features(x) for x in x_bridge], axis=0)

    lines: list[str] = []
    lines.append("Stage1.5 bridge vs phase4a alignment summary")
    lines.append("")
    lines.append("Real phase4a aggregate:")
    lines.append(f"- mean intensity mean: {real_feats[:, 0].mean():.6f}")
    lines.append(f"- std intensity mean: {real_feats[:, 1].mean():.6f}")
    lines.append(f"- frac(x>0.5) mean: {real_feats[:, 3].mean():.6f}")
    lines.append(f"- center index mean: {real_feats[:, 4].mean():.3f}")
    lines.append("")
    lines.append("Bridge examples:")
    for i in range(x_bridge.shape[0]):
        lines.append(
            f"- Bridge {i + 1}: mean={bridge_feats[i, 0]:.6f}, std={bridge_feats[i, 1]:.6f}, "
            f"frac(x>0.5)={bridge_feats[i, 3]:.6f}, center={bridge_feats[i, 4]:.3f}"
        )
    lines.append("")
    lines.append("Nearest real matches:")
    for m in matches:
        lines.append(
            f"- Bridge {int(m['bridge_index']) + 1} -> real index {int(m['real_index'])}, "
            f"target delta={float(m['target_delta_nm']) * 1e3:+.1f} pm, "
            f"real delta={float(m['real_delta_nm']) * 1e3:+.1f} pm, "
            f"MSE={float(m['mse']):.6e}"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    cfg = AlignmentDiagnosisConfig()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_inputs(repo_root, cfg)
    matches = find_nearest_real_samples(
        x_local=payload["X_local"],
        y_nm=payload["Y_nm"],
        x_bridge=payload["X_bridge"],
        y_bridge_nm=payload["Y_bridge_nm"],
        halfwidth_nm=cfg.label_pool_halfwidth_nm,
    )

    plot_distribution_alignment(
        x_local=payload["X_local"],
        x_bridge=payload["X_bridge"],
        output_path=output_dir / "bridge_vs_phase4a_distribution_alignment.png",
    )
    plot_overlay_matches(
        wavelengths_nm=payload["wavelengths_nm"],
        x_local=payload["X_local"],
        y_nm=payload["Y_nm"],
        x_bridge=payload["X_bridge"],
        y_bridge_pm=payload["Y_bridge_pm"],
        matches=matches,
        output_path=output_dir / "bridge_vs_phase4a_nearest_overlays.png",
    )

    summary_path = save_summary_text(
        repo_root=repo_root,
        cfg=cfg,
        matches=matches,
        x_local=payload["X_local"],
        x_bridge=payload["X_bridge"],
    )

    np.savez(
        output_dir / "bridge_vs_phase4a_alignment_data.npz",
        X_bridge=payload["X_bridge"].astype(np.float32),
        Y_bridge_pm=payload["Y_bridge_pm"].astype(np.float32),
        nearest_real_indices=np.array([int(m["real_index"]) for m in matches], dtype=np.int64),
        nearest_real_mse=np.array([float(m["mse"]) for m in matches], dtype=np.float64),
    )

    print("Stage1.5 alignment diagnosis completed.")
    print(f"Saved: {output_dir / 'bridge_vs_phase4a_distribution_alignment.png'}")
    print(f"Saved: {output_dir / 'bridge_vs_phase4a_nearest_overlays.png'}")
    print(f"Saved: {summary_path}")
    print("")
    print("Nearest real matches:")
    for m in matches:
        print(
            f"Bridge {int(m['bridge_index']) + 1} -> real index {int(m['real_index'])} | "
            f"target={float(m['target_delta_nm']) * 1e3:+.1f} pm | "
            f"real={float(m['real_delta_nm']) * 1e3:+.1f} pm | "
            f"MSE={float(m['mse']):.5e}"
        )


if __name__ == "__main__":
    main()
