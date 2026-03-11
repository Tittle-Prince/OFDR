from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick GRU fusion verify over single-frame predictions")
    p.add_argument(
        "--pred",
        type=str,
        default="results/phase4_array_se_hard/dualinput_fusion/predictions.npz",
        help="npz containing y_true and pred_bias_corrected (or pred_raw)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="results/phase4_array_se_hard/gru_fusion_verify",
    )
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def kalman_1d(x: np.ndarray, q_scale: float = 0.01) -> np.ndarray:
    y = x.astype(np.float64)
    r = float(np.var(y) * 0.25 + 1e-12)
    q = float(r * q_scale)
    out = np.empty_like(y)
    xhat = float(y[0])
    p = float(r)
    out[0] = xhat
    for i in range(1, len(y)):
        p = p + q
        k = p / (p + r)
        xhat = xhat + k * (float(y[i]) - xhat)
        p = (1.0 - k) * p
        out[i] = xhat
    return out.astype(np.float32)


def build_sequence_dataset(y: np.ndarray, pred: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    # Build a smooth pseudo-trajectory by sorting labels.
    order = np.argsort(y)
    ys = y[order].astype(np.float32)
    ps = pred[order].astype(np.float32)

    # Feature channels:
    # 0 raw pred, 1 ema9, 2 kalman, 3 local delta of pred
    ema = np.empty_like(ps)
    alpha = 2.0 / (9.0 + 1.0)
    ema[0] = ps[0]
    for i in range(1, len(ps)):
        ema[i] = alpha * ps[i] + (1.0 - alpha) * ema[i - 1]
    kf = kalman_1d(ps, q_scale=0.01)
    d1 = np.concatenate([[0.0], np.diff(ps)]).astype(np.float32)
    feat = np.stack([ps, ema, kf, d1], axis=1).astype(np.float32)

    if len(ys) <= window:
        raise ValueError("Not enough samples for chosen window")

    xs, ts = [], []
    for i in range(window - 1, len(ys)):
        xs.append(feat[i - window + 1 : i + 1])
        ts.append(ys[i])
    x_seq = np.stack(xs, axis=0).astype(np.float32)  # [B, T, F]
    y_seq = np.asarray(ts, dtype=np.float32)
    return x_seq, y_seq


class GRUFusion(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=2,
            dropout=0.10,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        h = out[:, -1, :]
        return self.head(h).squeeze(-1)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pred_path = Path(args.pred)
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    d = np.load(pred_path)
    y = d["y_true"].astype(np.float32).reshape(-1)
    if "pred_bias_corrected" in d:
        p = d["pred_bias_corrected"].astype(np.float32).reshape(-1)
    elif "pred_raw" in d:
        p = d["pred_raw"].astype(np.float32).reshape(-1)
    else:
        raise KeyError("pred file must contain pred_bias_corrected or pred_raw")

    x_seq, y_seq = build_sequence_dataset(y, p, window=int(args.window))
    n = len(y_seq)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    idx = np.arange(n)
    idx_train = idx[:n_train]
    idx_val = idx[n_train : n_train + n_val]
    idx_test = idx[n_train + n_val :]

    x_train = torch.tensor(x_seq[idx_train], dtype=torch.float32)
    y_train = torch.tensor(y_seq[idx_train], dtype=torch.float32)
    x_val = torch.tensor(x_seq[idx_val], dtype=torch.float32)
    y_val = torch.tensor(y_seq[idx_val], dtype=torch.float32)
    x_test = torch.tensor(x_seq[idx_test], dtype=torch.float32)
    y_test = y_seq[idx_test]

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=int(args.batch_size), shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=int(args.batch_size), shuffle=False)
    total_batches = len(train_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUFusion(in_dim=x_seq.shape[2], hidden=int(args.hidden)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss(beta=0.01)

    best_state = None
    best_val = float("inf")
    stale = 0
    print(
        f"Start GRU training | epochs={int(args.epochs)} | window={int(args.window)} | "
        f"batches/epoch={total_batches} | batch_size={int(args.batch_size)} | lr={float(args.lr)}"
    )
    for ep in range(1, int(args.epochs) + 1):
        model.train()
        train_losses = []
        for bidx, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_losses.append(float(loss.item()))

            if bidx == 1 or bidx == total_batches or (total_batches >= 4 and bidx % max(1, total_batches // 4) == 0):
                bprog = 100.0 * float(bidx) / float(max(1, total_batches))
                print(f"  Epoch {ep:03d} batch {bidx:03d}/{total_batches:03d} ({bprog:5.1f}%)")

        model.eval()
        vals = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pv = model(xb)
                vals.append(float(torch.mean((pv - yb) ** 2).item()))
        vm = float(np.mean(vals))
        trm = float(np.mean(train_losses)) if len(train_losses) > 0 else 0.0
        eprog = 100.0 * float(ep) / float(max(1, int(args.epochs)))
        print(f"Epoch {ep:03d}/{int(args.epochs):03d} ({eprog:5.1f}%) | train_loss={trm:.6f} | val_mse={vm:.6f}")
        if vm < best_val:
            best_val = vm
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                print(f"Early stop at epoch {ep}, best val_mse={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_gru = model(x_test.to(device)).cpu().numpy().astype(np.float32)

    m_gru = {"rmse": rmse(y_test, p_gru), "mae": mae(y_test, p_gru), "r2": r2(y_test, p_gru)}

    # Baselines on same sorted sequence split:
    base_raw = x_seq[idx_test, -1, 0]
    base_kf = x_seq[idx_test, -1, 2]
    m_raw = {"rmse": rmse(y_test, base_raw), "mae": mae(y_test, base_raw), "r2": r2(y_test, base_raw)}
    m_kf = {"rmse": rmse(y_test, base_kf), "mae": mae(y_test, base_kf), "r2": r2(y_test, base_kf)}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics_table.csv", "w", encoding="utf-8") as f:
        f.write("Method,RMSE_nm,MAE_nm,R2\n")
        f.write(f"Single(raw),{m_raw['rmse']:.8f},{m_raw['mae']:.8f},{m_raw['r2']:.8f}\n")
        f.write(f"Kalman(feature),{m_kf['rmse']:.8f},{m_kf['mae']:.8f},{m_kf['r2']:.8f}\n")
        f.write(f"GRU-Fusion,{m_gru['rmse']:.8f},{m_gru['mae']:.8f},{m_gru['r2']:.8f}\n")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {"single_raw": m_raw, "kalman": m_kf, "gru_fusion": m_gru, "window": int(args.window)},
            f,
            indent=2,
        )
    np.savez(
        out_dir / "predictions.npz",
        y_true=y_test.astype(np.float32),
        pred_single=base_raw.astype(np.float32),
        pred_kalman=base_kf.astype(np.float32),
        pred_gru=p_gru.astype(np.float32),
    )

    print(f"Single(raw)    RMSE={m_raw['rmse']:.6f} | MAE={m_raw['mae']:.6f} | R2={m_raw['r2']:.6f}")
    print(f"Kalman(feature)RMSE={m_kf['rmse']:.6f} | MAE={m_kf['mae']:.6f} | R2={m_kf['r2']:.6f}")
    print(f"GRU-Fusion     RMSE={m_gru['rmse']:.6f} | MAE={m_gru['mae']:.6f} | R2={m_gru['r2']:.6f}")
    print(f"Saved: {out_dir / 'metrics_table.csv'}")


if __name__ == "__main__":
    main()
