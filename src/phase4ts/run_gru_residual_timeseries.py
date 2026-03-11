from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase3.models import build_model
from phase3.train_utils import make_loaders, train_model
from phase4a.common import load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Isolated GRU-residual route on real time-series trajectories")
    p.add_argument("--config", type=str, default="config/phase4_array_se_hard_ts.yaml")
    p.add_argument("--skip-single-train", action="store_true", help="Use cached single-frame predictions if present")
    p.add_argument("--q-scale", type=float, default=None, help="Override kalman.q_scale")
    p.add_argument("--r-scale", type=float, default=None, help="Override kalman.r_scale")
    p.add_argument("--window", type=int, default=None, help="Override train_gru.window")
    p.add_argument("--hidden", type=int, default=None, help="Override train_gru.hidden")
    p.add_argument("--epochs", type=int, default=None, help="Override train_gru.epochs")
    p.add_argument("--patience", type=int, default=None, help="Override train_gru.patience")
    p.add_argument("--run-tag", type=str, default="", help="Subfolder tag under results_dir/gru_residual")
    return p.parse_args()


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12)
    return float(1.0 - ss_res / ss_tot)


def _p95(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ae = np.abs(y_true - y_pred)
    return float(np.percentile(ae, 95))


def _p99(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ae = np.abs(y_true - y_pred)
    return float(np.percentile(ae, 99))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": _rmse(y_true, y_pred),
        "mae": _mae(y_true, y_pred),
        "p95": _p95(y_true, y_pred),
        "p99": _p99(y_true, y_pred),
        "r2": _r2(y_true, y_pred),
    }


def _predict(model: nn.Module, x: np.ndarray, idx: np.ndarray, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    out = np.zeros(len(idx), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            e = min(len(idx), s + batch_size)
            xb = torch.tensor(x[idx[s:e]], dtype=torch.float32, device=device)
            out[s:e] = model(xb).detach().cpu().numpy().astype(np.float32)
    return out


def _causal_kalman_1d(x: np.ndarray, q_scale: float = 0.01, r_scale: float = 0.25) -> np.ndarray:
    y = x.astype(np.float64)
    r = float(np.var(y) * r_scale + 1e-12)
    q = float(r * q_scale + 1e-12)
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


def _kalman_by_trajectory(
    pred_single: np.ndarray,
    traj_id: np.ndarray,
    t_index: np.ndarray,
    q_scale: float,
    r_scale: float,
) -> np.ndarray:
    out = np.zeros_like(pred_single, dtype=np.float32)
    for tid in np.unique(traj_id):
        idx = np.flatnonzero(traj_id == tid)
        order = np.argsort(t_index[idx])
        idx_s = idx[order]
        out[idx_s] = _causal_kalman_1d(pred_single[idx_s], q_scale=q_scale, r_scale=r_scale)
    return out


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a / (b + 1e-12)


def _build_features(
    x_local: np.ndarray,
    x_total: np.ndarray,
    wavelengths: np.ndarray,
    pred_single: np.ndarray,
    pred_kalman: np.ndarray,
    traj_id: np.ndarray,
    t_index: np.ndarray,
) -> np.ndarray:
    wl = wavelengths.astype(np.float64)[None, :]
    xl = x_local.astype(np.float64)
    xt = x_total.astype(np.float64)

    sum_l = np.sum(xl, axis=1, keepdims=True)
    sum_t = np.sum(xt, axis=1, keepdims=True)
    cent_l = _safe_div(np.sum(xl * wl, axis=1, keepdims=True), sum_l)
    cent_t = _safe_div(np.sum(xt * wl, axis=1, keepdims=True), sum_t)

    var_l = _safe_div(np.sum(((wl - cent_l) ** 2) * xl, axis=1, keepdims=True), sum_l)
    width_l = np.sqrt(np.clip(var_l, 0.0, None))
    peak_l = np.max(xl, axis=1, keepdims=True)
    peak_t = np.max(xt, axis=1, keepdims=True)

    d_single = np.zeros_like(pred_single, dtype=np.float32)
    d_kal = np.zeros_like(pred_kalman, dtype=np.float32)
    for tid in np.unique(traj_id):
        idx = np.flatnonzero(traj_id == tid)
        order = np.argsort(t_index[idx])
        idx_s = idx[order]
        ps = pred_single[idx_s]
        pk = pred_kalman[idx_s]
        ds = np.concatenate([[0.0], np.diff(ps)]).astype(np.float32)
        dk = np.concatenate([[0.0], np.diff(pk)]).astype(np.float32)
        d_single[idx_s] = ds
        d_kal[idx_s] = dk

    feat = np.column_stack(
        [
            pred_single.astype(np.float32),
            pred_kalman.astype(np.float32),
            (pred_single - pred_kalman).astype(np.float32),
            d_single.astype(np.float32),
            d_kal.astype(np.float32),
            cent_l.reshape(-1).astype(np.float32),
            cent_t.reshape(-1).astype(np.float32),
            peak_l.reshape(-1).astype(np.float32),
            peak_t.reshape(-1).astype(np.float32),
            width_l.reshape(-1).astype(np.float32),
        ]
    )
    return feat.astype(np.float32)


def _build_sequence_dataset(
    features: np.ndarray,
    y_true: np.ndarray,
    pred_kalman: np.ndarray,
    traj_id: np.ndarray,
    t_index: np.ndarray,
    traj_ids: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, frame_idx = [], [], []
    use_set = set(int(x) for x in traj_ids.reshape(-1))
    for tid in np.unique(traj_id):
        t_int = int(tid)
        if t_int not in use_set:
            continue
        idx = np.flatnonzero(traj_id == t_int)
        order = np.argsort(t_index[idx])
        idx_s = idx[order]
        if len(idx_s) < window:
            continue
        for t in range(window - 1, len(idx_s)):
            sl = idx_s[t - window + 1 : t + 1]
            end_idx = idx_s[t]
            xs.append(features[sl])
            ys.append(float(y_true[end_idx] - pred_kalman[end_idx]))
            frame_idx.append(int(end_idx))
    if len(xs) == 0:
        raise ValueError("No sequence samples produced. Check trajectory length/window.")
    x_arr = np.stack(xs, axis=0).astype(np.float32)
    y_arr = np.asarray(ys, dtype=np.float32)
    idx_arr = np.asarray(frame_idx, dtype=np.int64)
    return x_arr, y_arr, idx_arr


class GRUResidual(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_layers: int, dropout: float):
        super().__init__()
        dp = float(dropout) if int(num_layers) > 1 else 0.0
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=int(hidden),
            num_layers=int(num_layers),
            dropout=dp,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(int(hidden), int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o, _ = self.gru(x)
        h = o[:, -1, :]
        return self.head(h).squeeze(-1)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["phase4ts"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    data_path = resolve_project_path(cfg["phase4ts"]["dataset_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Time-series dataset missing: {data_path}")
    d = np.load(data_path)

    required = [
        "X_local",
        "X_total",
        "Y_dlambda_target",
        "wavelengths",
        "traj_id",
        "t_index",
        "idx_train",
        "idx_val",
        "idx_test",
        "traj_ids_train",
        "traj_ids_val",
        "traj_ids_test",
    ]
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(f"Dataset missing keys: {missing}")

    x_local = d["X_local"].astype(np.float32)
    x_total = d["X_total"].astype(np.float32)
    y_true = d["Y_dlambda_target"].astype(np.float32).reshape(-1)
    wl = d["wavelengths"].astype(np.float32).reshape(-1)
    traj_id = d["traj_id"].astype(np.int64).reshape(-1)
    t_index = d["t_index"].astype(np.int64).reshape(-1)
    idx_train = d["idx_train"].astype(np.int64).reshape(-1)
    idx_val = d["idx_val"].astype(np.int64).reshape(-1)
    idx_test = d["idx_test"].astype(np.int64).reshape(-1)
    traj_train = d["traj_ids_train"].astype(np.int64).reshape(-1)
    traj_val = d["traj_ids_val"].astype(np.int64).reshape(-1)
    traj_test = d["traj_ids_test"].astype(np.int64).reshape(-1)

    x_cat = np.concatenate([x_local, x_total], axis=1).astype(np.float32)
    run_tag = str(args.run_tag).strip()
    base_root = resolve_project_path(cfg["phase4ts"]["results_dir"]) / "gru_residual"
    out_root = base_root
    if len(run_tag) > 0:
        out_root = out_root / run_tag
    out_root.mkdir(parents=True, exist_ok=True)

    single_pred_cache = base_root / "single_frame_predictions_all.npz"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if bool(args.skip_single_train) and single_pred_cache.exists():
        cache = np.load(single_pred_cache)
        pred_single_bc_all = cache["pred_single_bias_corrected"].astype(np.float32).reshape(-1)
        bias_val = float(cache["val_bias_nm"][0])
        print(f"Loaded cached single-frame predictions: {single_pred_cache}")
    else:
        tr_cfg = cfg["train_single"]
        train_loader, val_loader = make_loaders(
            x_cat,
            y_true,
            idx_train,
            idx_val,
            batch_size=int(tr_cfg["batch_size"]),
        )
        model = build_model("cnn_se", input_dim=x_cat.shape[1]).to(device)
        model = train_model(model, train_loader, val_loader, tr_cfg, device)

        pred_train = _predict(model, x_cat, idx_train, device=device)
        pred_val = _predict(model, x_cat, idx_val, device=device)
        pred_test = _predict(model, x_cat, idx_test, device=device)
        bias_val = float(np.mean(y_true[idx_val] - pred_val))

        pred_single_bc_all = np.zeros_like(y_true, dtype=np.float32)
        pred_single_bc_all[idx_train] = pred_train + bias_val
        pred_single_bc_all[idx_val] = pred_val + bias_val
        pred_single_bc_all[idx_test] = pred_test + bias_val

        np.savez(
            single_pred_cache,
            pred_single_bias_corrected=pred_single_bc_all.astype(np.float32),
            val_bias_nm=np.array([bias_val], dtype=np.float32),
        )
        print(f"Saved single-frame cache: {single_pred_cache}")

    kf_cfg = cfg["kalman"]
    q_scale = float(args.q_scale) if args.q_scale is not None else float(kf_cfg["q_scale"])
    r_scale = float(args.r_scale) if args.r_scale is not None else float(kf_cfg["r_scale"])
    pred_kal_all = _kalman_by_trajectory(
        pred_single=pred_single_bc_all,
        traj_id=traj_id,
        t_index=t_index,
        q_scale=q_scale,
        r_scale=r_scale,
    )

    feat_all = _build_features(
        x_local=x_local,
        x_total=x_total,
        wavelengths=wl,
        pred_single=pred_single_bc_all,
        pred_kalman=pred_kal_all,
        traj_id=traj_id,
        t_index=t_index,
    )

    gcfg = dict(cfg["train_gru"])
    if args.window is not None:
        gcfg["window"] = int(args.window)
    if args.hidden is not None:
        gcfg["hidden"] = int(args.hidden)
    if args.epochs is not None:
        gcfg["epochs"] = int(args.epochs)
    if args.patience is not None:
        gcfg["patience"] = int(args.patience)

    window = int(gcfg["window"])
    x_tr, y_tr, idx_tr = _build_sequence_dataset(
        feat_all, y_true, pred_kal_all, traj_id, t_index, traj_train, window=window
    )
    x_va, y_va, idx_va = _build_sequence_dataset(
        feat_all, y_true, pred_kal_all, traj_id, t_index, traj_val, window=window
    )
    x_te, y_te, idx_te = _build_sequence_dataset(
        feat_all, y_true, pred_kal_all, traj_id, t_index, traj_test, window=window
    )

    # Feature normalization from training sequences only.
    mu = np.mean(x_tr.reshape(-1, x_tr.shape[-1]), axis=0, keepdims=True)
    sd = np.std(x_tr.reshape(-1, x_tr.shape[-1]), axis=0, keepdims=True) + 1e-6
    x_tr_n = ((x_tr - mu) / sd).astype(np.float32)
    x_va_n = ((x_va - mu) / sd).astype(np.float32)
    x_te_n = ((x_te - mu) / sd).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_tr_n), torch.tensor(y_tr)),
        batch_size=int(gcfg["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_va_n), torch.tensor(y_va)),
        batch_size=int(gcfg["batch_size"]),
        shuffle=False,
    )
    total_batches = len(train_loader)
    print(
        f"Start GRU-residual training | epochs={int(gcfg['epochs'])} | window={window} | "
        f"train_seq={len(x_tr_n)} | val_seq={len(x_va_n)} | test_seq={len(x_te_n)} | batches/epoch={total_batches}"
    )

    model_gru = GRUResidual(
        in_dim=x_tr_n.shape[2],
        hidden=int(gcfg["hidden"]),
        num_layers=int(gcfg["num_layers"]),
        dropout=float(gcfg["dropout"]),
    ).to(device)
    opt = torch.optim.AdamW(
        model_gru.parameters(),
        lr=float(gcfg["lr"]),
        weight_decay=float(gcfg["weight_decay"]),
    )
    loss_raw = nn.SmoothL1Loss(beta=float(gcfg["huber_beta"]), reduction="none")

    tail_thr = float(np.percentile(np.abs(y_tr), float(gcfg["tail_percentile"])))
    tail_w = float(gcfg["tail_weight"])
    best_val = float("inf")
    best_state = None
    stale = 0
    for ep in range(1, int(gcfg["epochs"]) + 1):
        model_gru.train()
        tr_losses = []
        for bidx, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pr = model_gru(xb)
            l = loss_raw(pr, yb)
            w = 1.0 + tail_w * (torch.abs(yb) > tail_thr).float()
            loss = torch.mean(w * l)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_gru.parameters(), max_norm=float(gcfg["grad_clip"]))
            opt.step()
            tr_losses.append(float(loss.item()))

            if bidx == 1 or bidx == total_batches or (total_batches >= 4 and bidx % max(1, total_batches // 4) == 0):
                prog = 100.0 * float(bidx) / float(max(1, total_batches))
                print(f"  Epoch {ep:03d} batch {bidx:03d}/{total_batches:03d} ({prog:5.1f}%)")

        model_gru.eval()
        vals = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pv = model_gru(xb)
                vals.append(float(torch.mean((pv - yb) ** 2).item()))
        vm = float(np.mean(vals))
        trm = float(np.mean(tr_losses)) if len(tr_losses) > 0 else 0.0
        eprog = 100.0 * float(ep) / float(max(1, int(gcfg["epochs"])))
        print(f"Epoch {ep:03d}/{int(gcfg['epochs']):03d} ({eprog:5.1f}%) | train_loss={trm:.6f} | val_mse={vm:.6f}")
        if vm < best_val:
            best_val = vm
            best_state = {k: v.detach().cpu().clone() for k, v in model_gru.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(gcfg["patience"]):
                print(f"Early stop at epoch {ep}, best val_mse={best_val:.6f}")
                break

    if best_state is not None:
        model_gru.load_state_dict(best_state)

    model_gru.eval()
    with torch.no_grad():
        pred_res = model_gru(torch.tensor(x_te_n, dtype=torch.float32, device=device)).cpu().numpy().astype(np.float32)

    y_eval = y_true[idx_te]
    p_single = pred_single_bc_all[idx_te]
    p_kal = pred_kal_all[idx_te]
    p_gru = (p_kal + pred_res).astype(np.float32)

    m_single = _metrics(y_eval, p_single)
    m_kal = _metrics(y_eval, p_kal)
    m_gru = _metrics(y_eval, p_gru)

    with open(out_root / "metrics_table.csv", "w", encoding="utf-8") as f:
        f.write("Method,RMSE_nm,MAE_nm,P95_nm,P99_nm,R2\n")
        f.write(
            f"SingleFrame(CNNSE+bias),{m_single['rmse']:.8f},{m_single['mae']:.8f},"
            f"{m_single['p95']:.8f},{m_single['p99']:.8f},{m_single['r2']:.8f}\n"
        )
        f.write(
            f"Kalman(Causal),{m_kal['rmse']:.8f},{m_kal['mae']:.8f},"
            f"{m_kal['p95']:.8f},{m_kal['p99']:.8f},{m_kal['r2']:.8f}\n"
        )
        f.write(
            f"GRUResidual(Kalman+res),{m_gru['rmse']:.8f},{m_gru['mae']:.8f},"
            f"{m_gru['p95']:.8f},{m_gru['p99']:.8f},{m_gru['r2']:.8f}\n"
        )

    with open(out_root / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "single_frame": m_single,
                "kalman": m_kal,
                "gru_residual": m_gru,
                "val_bias_nm": bias_val,
                "window": window,
                "kalman_q_scale": q_scale,
                "kalman_r_scale": r_scale,
                "gru_hidden": int(gcfg["hidden"]),
                "gru_epochs": int(gcfg["epochs"]),
                "gru_patience": int(gcfg["patience"]),
                "train_sequences": int(len(x_tr)),
                "val_sequences": int(len(x_va)),
                "test_sequences": int(len(x_te)),
            },
            f,
            indent=2,
        )

    np.savez(
        out_root / "predictions_test.npz",
        frame_idx=idx_te.astype(np.int64),
        y_true=y_eval.astype(np.float32),
        pred_single=p_single.astype(np.float32),
        pred_kalman=p_kal.astype(np.float32),
        pred_gru=p_gru.astype(np.float32),
    )

    print(
        f"SingleFrame(CNNSE+bias) RMSE={m_single['rmse']:.6f} | MAE={m_single['mae']:.6f} | "
        f"P95={m_single['p95']:.6f} | P99={m_single['p99']:.6f}"
    )
    print(
        f"Kalman(Causal)          RMSE={m_kal['rmse']:.6f} | MAE={m_kal['mae']:.6f} | "
        f"P95={m_kal['p95']:.6f} | P99={m_kal['p99']:.6f}"
    )
    print(
        f"GRUResidual(K+res)      RMSE={m_gru['rmse']:.6f} | MAE={m_gru['mae']:.6f} | "
        f"P95={m_gru['p95']:.6f} | P99={m_gru['p99']:.6f}"
    )
    print(f"Saved: {out_root / 'metrics_table.csv'}")


if __name__ == "__main__":
    main()
