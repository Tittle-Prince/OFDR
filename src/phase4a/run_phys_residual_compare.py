from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase2.baselines import (
    estimate_center_by_parametric_fit,
    estimate_shift_by_cross_correlation,
    gaussian_spectrum,
    normalize_minmax,
)
from phase3.common import metrics_dict, set_seed
from phase4a.common import load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Physics-prior residual CNN comparison")
    p.add_argument("--config", type=str, default="config/phase4_array_se_hard.yaml")
    return p.parse_args()


def load_phase4a_dataset(cfg: dict) -> dict:
    data_path = resolve_project_path(cfg["phase4a"]["dataset_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset missing: {data_path}")
    data = np.load(data_path)
    needed = ["X_local", "Y_dlambda_target", "wavelengths", "idx_train", "idx_val", "idx_test"]
    miss = [k for k in needed if k not in data]
    if miss:
        raise KeyError(f"Dataset missing keys: {miss}")
    return {
        "x": data["X_local"].astype(np.float32),
        "y": data["Y_dlambda_target"].astype(np.float32),
        "wavelengths": data["wavelengths"].astype(np.float32),
        "idx_train": data["idx_train"].astype(np.int64),
        "idx_val": data["idx_val"].astype(np.int64),
        "idx_test": data["idx_test"].astype(np.int64),
    }


def _gaussian_kernel1d(sigma_points: float) -> np.ndarray:
    sigma = float(max(1e-6, sigma_points))
    radius = int(max(1, round(4.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= np.sum(k)
    return k


def _conv1d_same(y: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.convolve(y, kernel, mode="same")


def preprocess_spectra(x: np.ndarray, cfg: dict) -> np.ndarray:
    pcfg = cfg.get("signal_processing", {})
    if not bool(pcfg.get("enabled", False)):
        return x
    baseline_sigma = float(pcfg.get("baseline_sigma_points", 18.0))
    denoise_sigma = float(pcfg.get("denoise_sigma_points", 1.0))
    base_strength = float(pcfg.get("baseline_subtract_strength", 1.0))
    edge_alpha = float(pcfg.get("edge_enhance_alpha", 0.0))
    p_low = float(pcfg.get("p_low", 1.0))
    p_high = float(pcfg.get("p_high", 99.0))

    k_base = _gaussian_kernel1d(baseline_sigma)
    k_denoise = _gaussian_kernel1d(denoise_sigma)
    out = np.zeros_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        y = x[i].astype(np.float64)
        baseline = _conv1d_same(y, k_base)
        y_det = y - base_strength * baseline
        y_smooth = _conv1d_same(y_det, k_denoise)
        y_proc = y_smooth + edge_alpha * (y_det - y_smooth)
        lo = float(np.percentile(y_proc, p_low))
        hi = float(np.percentile(y_proc, p_high))
        y_proc = (y_proc - lo) / (hi - lo + 1e-12)
        out[i] = np.clip(y_proc, 0.0, 1.0).astype(np.float32)
    return out


def build_physics_coarse(ds: dict, cfg: dict, x_used: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wl = ds["wavelengths"]
    step_nm = float(wl[1] - wl[0])
    lambda0 = float(cfg["array"]["lambda0_nm"])
    sigma = float(cfg["array"]["linewidth_sigma_nm"])
    fit_window_points = int(cfg["compare"]["fit_window_points"])
    baseline_percentile = float(cfg["compare"]["baseline_percentile"])

    ref = gaussian_spectrum(wl, center_nm=lambda0, sigma_nm=sigma, amplitude=1.0, baseline=0.0)
    if str(cfg["local_window"]["normalize"]) == "minmax_per_sample":
        ref = normalize_minmax(ref)

    n = x_used.shape[0]
    pred_cc = np.zeros(n, dtype=np.float32)
    pred_pf = np.zeros(n, dtype=np.float32)
    for i in range(n):
        pred_cc[i] = estimate_shift_by_cross_correlation(ref, x_used[i], step_nm)
        c = estimate_center_by_parametric_fit(
            wl,
            x_used[i],
            fit_window_points=fit_window_points,
            baseline_percentile=baseline_percentile,
        )
        pred_pf[i] = np.float32(c - lambda0)

    pcfg = cfg.get("physics_train", {})
    tau = float(pcfg.get("disagreement_tau_nm", 0.03))
    w_pf = float(pcfg.get("coarse_pf_weight", 0.75))
    diff = np.abs(pred_pf - pred_cc)
    conf = np.exp(-diff / max(1e-9, tau)).astype(np.float32)
    # Dynamic blending: when disagreement is small, average both; otherwise trust PF more.
    alpha = (0.5 * conf + (1.0 - conf) * w_pf).astype(np.float32)
    coarse = alpha * pred_pf + (1.0 - alpha) * pred_cc
    return coarse.astype(np.float32), conf.astype(np.float32), diff.astype(np.float32)


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w


class ResidualFusionNet(nn.Module):
    def __init__(self, input_dim: int, use_se: bool):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        b2: list[nn.Module] = [nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU()]
        if use_se:
            b2.append(SEBlock1D(64))
        b2.append(nn.MaxPool1d(2))
        self.block2 = nn.Sequential(*b2)
        b3: list[nn.Module] = [nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU()]
        if use_se:
            b3.append(SEBlock1D(128))
        self.block3 = nn.Sequential(*b3)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            feat_dim = int(np.prod(self._forward_features(dummy).shape[1:]))

        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.res_head = nn.Sequential(
            nn.Linear(65, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.dt_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward(self, x: torch.Tensor, coarse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self._forward_features(x.unsqueeze(1))
        emb = self.embed(feat)
        fused = torch.cat([emb, coarse.unsqueeze(1)], dim=1)
        residual = self.res_head(fused).squeeze(-1)
        dtemp = self.dt_head(emb).squeeze(-1)
        final = coarse + residual
        return final, residual, dtemp


@dataclass
class PhysicsTrainCfg:
    lr: float
    weight_decay: float
    epochs: int
    patience: int
    alpha_prior: float
    beta_temp: float
    k_t: float
    prior_beta: float
    grad_clip: float
    conf_gate: float
    conf_power: float
    prior_warmup_epochs: int


def get_phys_train_cfg(cfg: dict) -> PhysicsTrainCfg:
    pcfg = cfg.get("physics_train", {})
    return PhysicsTrainCfg(
        lr=float(pcfg.get("lr", 8e-4)),
        weight_decay=float(pcfg.get("weight_decay", 1e-5)),
        epochs=int(pcfg.get("epochs", 60)),
        patience=int(pcfg.get("patience", 12)),
        alpha_prior=float(pcfg.get("alpha_prior", 0.18)),
        beta_temp=float(pcfg.get("beta_temp", 0.08)),
        k_t=float(pcfg.get("k_t_nm_per_c", 0.01)),
        prior_beta=float(pcfg.get("prior_beta", 0.015)),
        grad_clip=float(pcfg.get("grad_clip", 1.0)),
        conf_gate=float(pcfg.get("conf_gate", 0.55)),
        conf_power=float(pcfg.get("conf_power", 2.0)),
        prior_warmup_epochs=int(pcfg.get("prior_warmup_epochs", 12)),
    )


def train_phys_model(
    model: ResidualFusionNet,
    ds: dict,
    coarse: np.ndarray,
    conf: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    cfg: PhysicsTrainCfg,
    device: torch.device,
) -> ResidualFusionNet:
    x_train = torch.tensor(ds["x_proc"][idx_train], dtype=torch.float32)
    y_train = torch.tensor(ds["y"][idx_train], dtype=torch.float32)
    c_train = torch.tensor(coarse[idx_train], dtype=torch.float32)
    w_train = torch.tensor(conf[idx_train], dtype=torch.float32)
    x_val = torch.tensor(ds["x_proc"][idx_val], dtype=torch.float32)
    y_val = torch.tensor(ds["y"][idx_val], dtype=torch.float32)
    c_val = torch.tensor(coarse[idx_val], dtype=torch.float32)
    w_val = torch.tensor(conf[idx_val], dtype=torch.float32)

    batch_size = int(ds.get("batch_size", 128))
    train_loader = DataLoader(TensorDataset(x_train, y_train, c_train, w_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val, c_val, w_val), batch_size=batch_size, shuffle=False)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()
    smooth = nn.SmoothL1Loss(beta=cfg.prior_beta, reduction="none")

    best_state = None
    best_val = float("inf")
    stale = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb, cb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            cb = cb.to(device)
            wb = wb.to(device)
            opt.zero_grad()
            pred, _, dtemp = model(xb, cb)
            data_loss = mse(pred, yb)
            w_conf = torch.clamp(wb, 0.0, 1.0) ** cfg.conf_power
            conf_mask = (w_conf >= cfg.conf_gate).float()
            prior_raw = smooth(pred, cb)
            if float(torch.sum(conf_mask).item()) > 0.5:
                prior_loss = torch.sum(w_conf * conf_mask * prior_raw) / (torch.sum(conf_mask) + 1e-12)
            else:
                prior_loss = torch.zeros((), device=xb.device)
            temp_loss = mse(pred, dtemp * cfg.k_t)
            warm = min(1.0, float(epoch) / max(1, cfg.prior_warmup_epochs))
            alpha_eff = cfg.alpha_prior * warm
            loss = data_loss + alpha_eff * prior_loss + cfg.beta_temp * temp_loss
            loss.backward()
            if cfg.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, cb, _ in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                cb = cb.to(device)
                pred, _, _ = model(xb, cb)
                val_losses.append(float(torch.mean((pred - yb) ** 2).item()))
        val_mse = float(np.mean(val_losses))
        if epoch % 5 == 0 or epoch == 1:
            print(f"[Phys] Epoch {epoch:03d} | val_mse={val_mse:.6f}")
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                print(f"[Phys] Early stop at epoch {epoch}, best val_mse={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_phys(model: ResidualFusionNet, ds: dict, coarse: np.ndarray, idx: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    x = torch.tensor(ds["x_proc"][idx], dtype=torch.float32, device=device)
    c = torch.tensor(coarse[idx], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred, _, _ = model(x, c)
    return pred.cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["phase4a"]["seed"]))
    ds = load_phase4a_dataset(cfg)
    ds["x_proc"] = preprocess_spectra(ds["x"], cfg)
    ds["batch_size"] = int(cfg["train"]["batch_size"])

    coarse, conf, diff = build_physics_coarse(ds, cfg, ds["x_proc"])
    y_true = ds["y"][ds["idx_test"]]
    coarse_test = coarse[ds["idx_test"]]
    coarse_metrics = metrics_dict(y_true, coarse_test)
    print(
        f"Coarse prior | RMSE={coarse_metrics['rmse']:.6f} | "
        f"MAE={coarse_metrics['mae']:.6f} | R2={coarse_metrics['r2']:.6f}"
    )
    print(f"Coarse disagreement mean={float(np.mean(diff)):.6f} nm")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg = get_phys_train_cfg(cfg)

    set_seed(int(cfg["phase4a"]["seed"]) + 121)
    model_cnn = ResidualFusionNet(input_dim=ds["x"].shape[1], use_se=False)
    model_cnn = train_phys_model(model_cnn, ds, coarse, conf, ds["idx_train"], ds["idx_val"], tcfg, device)
    pred_phys_cnn = predict_phys(model_cnn, ds, coarse, ds["idx_test"], device)
    m_phys_cnn = metrics_dict(y_true, pred_phys_cnn)

    set_seed(int(cfg["phase4a"]["seed"]) + 131)
    model_se = ResidualFusionNet(input_dim=ds["x"].shape[1], use_se=True)
    model_se = train_phys_model(model_se, ds, coarse, conf, ds["idx_train"], ds["idx_val"], tcfg, device)
    pred_phys_se = predict_phys(model_se, ds, coarse, ds["idx_test"], device)
    m_phys_se = metrics_dict(y_true, pred_phys_se)

    rows = [
        ("Coarse prior", coarse_metrics),
        ("Phys-CNN", m_phys_cnn),
        ("Phys-CNN+SE", m_phys_se),
    ]
    out_dir = resolve_project_path(cfg["phase4a"]["results_dir"]) / "phys_residual_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics_table.csv", "w", encoding="utf-8") as f:
        f.write("Method,RMSE_nm,MAE_nm,R2\n")
        for name, m in rows:
            f.write(f"{name},{m['rmse']:.8f},{m['mae']:.8f},{m['r2']:.8f}\n")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({name: m for name, m in rows}, f, indent=2)

    np.savez(
        out_dir / "predictions.npz",
        y_true=y_true.astype(np.float32),
        pred_coarse=coarse_test.astype(np.float32),
        pred_phys_cnn=pred_phys_cnn.astype(np.float32),
        pred_phys_cnn_se=pred_phys_se.astype(np.float32),
        conf_test=conf[ds["idx_test"]].astype(np.float32),
    )

    print(f"Saved: {out_dir / 'metrics_table.csv'}")
    for name, m in rows:
        print(f"{name:<14} RMSE={m['rmse']:.6f} nm | MAE={m['mae']:.6f} nm | R2={m['r2']:.6f}")


if __name__ == "__main__":
    main()
