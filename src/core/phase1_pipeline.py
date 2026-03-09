import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


@dataclass
class DatasetA:
    x: np.ndarray
    y_dlambda: np.ndarray
    y_dt: np.ndarray
    lambda_axis: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray


def gaussian_spectrum(
    lambda_axis: np.ndarray,
    center_nm: float,
    sigma_nm: float,
    amplitude: float,
    baseline: float,
) -> np.ndarray:
    return baseline + amplitude * np.exp(-0.5 * ((lambda_axis - center_nm) / sigma_nm) ** 2)


def normalize_minmax(x: np.ndarray) -> np.ndarray:
    xmin = x.min()
    xmax = x.max()
    return (x - xmin) / (xmax - xmin + 1e-12)


def generate_dataset_a(cfg: dict) -> DatasetA:
    dcfg = cfg["dataset"]
    seed = int(cfg["phase1"]["seed"])
    rng = np.random.default_rng(seed)

    n = int(dcfg["num_samples"])
    num_points = int(dcfg["num_points"])
    lambda_b0 = float(dcfg["lambda_b0_nm"])
    window = float(dcfg["lambda_window_nm"])
    sigma = float(dcfg["linewidth_nm"])
    amplitude = float(dcfg["peak_amplitude"])
    baseline = float(dcfg["baseline"])
    temp_min = float(dcfg["temp_min_c"])
    temp_max = float(dcfg["temp_max_c"])
    k_t = float(dcfg["k_t_nm_per_c"])
    noise_std = float(dcfg["noise_std"])
    normalize_mode = str(dcfg["normalize"])
    clip_nonnegative = bool(dcfg["clip_to_nonnegative"])

    lambda_axis = np.linspace(lambda_b0 - window / 2.0, lambda_b0 + window / 2.0, num_points, dtype=np.float64)
    y_dt = rng.uniform(temp_min, temp_max, size=n).astype(np.float64)
    y_dlambda = y_dt * k_t

    x = np.zeros((n, num_points), dtype=np.float32)
    for i in range(n):
        center = lambda_b0 + y_dlambda[i]
        clean = gaussian_spectrum(lambda_axis, center, sigma, amplitude, baseline)
        noisy = clean + rng.normal(0.0, noise_std, size=num_points)
        if clip_nonnegative:
            noisy = np.clip(noisy, 0.0, None)
        if normalize_mode == "minmax_per_sample":
            noisy = normalize_minmax(noisy)
        x[i] = noisy.astype(np.float32)

    idx = rng.permutation(n)
    train_ratio = float(dcfg["train_ratio"])
    val_ratio = float(dcfg["val_ratio"])
    test_ratio = float(dcfg["test_ratio"])
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    idx_train = idx[:n_train]
    idx_val = idx[n_train : n_train + n_val]
    idx_test = idx[n_train + n_val :]

    return DatasetA(
        x=x,
        y_dlambda=y_dlambda.astype(np.float32),
        y_dt=y_dt.astype(np.float32),
        lambda_axis=lambda_axis.astype(np.float32),
        idx_train=idx_train.astype(np.int64),
        idx_val=idx_val.astype(np.int64),
        idx_test=idx_test.astype(np.int64),
    )


def save_dataset(ds: DatasetA, data_path: Path) -> None:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        data_path,
        X=ds.x,
        Y_dlambda=ds.y_dlambda,
        Y_dT=ds.y_dt,
        wavelengths=ds.lambda_axis,
        idx_train=ds.idx_train,
        idx_val=ds.idx_val,
        idx_test=ds.idx_test,
    )


def load_dataset(data_path: Path) -> DatasetA:
    data = np.load(data_path)
    return DatasetA(
        x=data["X"],
        y_dlambda=data["Y_dlambda"],
        y_dt=data["Y_dT"],
        lambda_axis=data["wavelengths"],
        idx_train=data["idx_train"],
        idx_val=data["idx_val"],
        idx_test=data["idx_test"],
    )


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CNN1DRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            feat_dim = int(np.prod(self.features(dummy).shape[1:]))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        return self.head(self.features(x)).squeeze(-1)


def make_loaders(
    x: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    x_train = torch.tensor(x[idx_train], dtype=torch.float32)
    y_train = torch.tensor(y[idx_train], dtype=torch.float32)
    x_val = torch.tensor(x[idx_val], dtype=torch.float32)
    y_val = torch.tensor(y[idx_val], dtype=torch.float32)
    x_test = torch.tensor(x[idx_test], dtype=torch.float32)
    y_test = torch.tensor(y[idx_test], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, x_test, y_test


def evaluate_model(model: nn.Module, x: torch.Tensor, y_true: torch.Tensor, device: torch.device) -> dict:
    model.eval()
    with torch.no_grad():
        pred = model(x.to(device)).cpu().numpy()
    y_np = y_true.numpy()
    return {
        "rmse": rmse(y_np, pred),
        "mae": mae(y_np, pred),
        "r2": r2(y_np, pred),
        "pred": pred,
    }


def train_torch_regressor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg_train: dict,
    device: torch.device,
) -> nn.Module:
    lr = float(cfg_train["lr"])
    weight_decay = float(cfg_train["weight_decay"])
    epochs = int(cfg_train["epochs"])
    patience = int(cfg_train["patience"])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                val_losses.append(float(criterion(model(xb), yb).item()))
        mean_val = float(np.mean(val_losses))
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | val_mse={mean_val:.6f}")

        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stop at epoch {epoch}, best val_mse={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def estimate_shift_by_cross_correlation(reference: np.ndarray, sample: np.ndarray, step_nm: float) -> float:
    ref = reference - reference.mean()
    sig = sample - sample.mean()
    corr = np.correlate(sig, ref, mode="full")
    peak = int(np.argmax(corr))
    lag = peak - (len(reference) - 1)

    delta = 0.0
    if 0 < peak < len(corr) - 1:
        y1, y2, y3 = corr[peak - 1], corr[peak], corr[peak + 1]
        denom = y1 - 2.0 * y2 + y3
        if abs(denom) > 1e-12:
            delta = 0.5 * (y1 - y3) / denom

    return float((lag + delta) * step_nm)


def evaluate_cross_correlation(ds: DatasetA, cfg: dict) -> dict:
    dcfg = cfg["dataset"]
    lambda_b0 = float(dcfg["lambda_b0_nm"])
    sigma = float(dcfg["linewidth_nm"])
    amplitude = float(dcfg["peak_amplitude"])
    baseline = float(dcfg["baseline"])
    step_nm = float(ds.lambda_axis[1] - ds.lambda_axis[0])

    ref = gaussian_spectrum(ds.lambda_axis, lambda_b0, sigma, amplitude, baseline)
    if str(dcfg["normalize"]) == "minmax_per_sample":
        ref = normalize_minmax(ref)

    y_true = ds.y_dlambda[ds.idx_test]
    y_pred = np.zeros_like(y_true)
    for i, idx in enumerate(ds.idx_test):
        y_pred[i] = estimate_shift_by_cross_correlation(ref, ds.x[idx], step_nm)

    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred), "r2": r2(y_true, y_pred), "pred": y_pred}


def run_phase1(cfg: dict, regenerate: bool) -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / cfg["phase1"]["data_path"]
    results_dir = project_root / cfg["phase1"]["results_dir"]
    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg["phase1"]["seed"]))

    if regenerate or (not data_path.exists()):
        print(f"Generating Dataset_A at {data_path}")
        ds = generate_dataset_a(cfg)
        save_dataset(ds, data_path)
    else:
        print(f"Using existing Dataset_A: {data_path}")
        ds = load_dataset(data_path)

    batch_size = int(cfg["train"]["batch_size"])
    train_loader, val_loader, x_test, y_test = make_loaders(
        ds.x, ds.y_dlambda, ds.idx_train, ds.idx_val, ds.idx_test, batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Phase1 target: delta_lambda regression")

    cc_metrics = evaluate_cross_correlation(ds, cfg)
    print(
        f"Cross-correlation | RMSE={cc_metrics['rmse']:.6f} nm | "
        f"MAE={cc_metrics['mae']:.6f} nm | R2={cc_metrics['r2']:.6f}"
    )

    mlp = MLPRegressor(input_dim=ds.x.shape[1]).to(device)
    print("Training MLP ...")
    mlp = train_torch_regressor(mlp, train_loader, val_loader, cfg["train"], device)
    mlp_metrics = evaluate_model(mlp, x_test, y_test, device)
    torch.save(mlp.state_dict(), models_dir / "phase1_mlp.pt")
    print(
        f"MLP              | RMSE={mlp_metrics['rmse']:.6f} nm | "
        f"MAE={mlp_metrics['mae']:.6f} nm | R2={mlp_metrics['r2']:.6f}"
    )

    cnn = CNN1DRegressor(input_dim=ds.x.shape[1]).to(device)
    print("Training 1D CNN ...")
    cnn = train_torch_regressor(cnn, train_loader, val_loader, cfg["train"], device)
    cnn_metrics = evaluate_model(cnn, x_test, y_test, device)
    torch.save(cnn.state_dict(), models_dir / "phase1_cnn1d.pt")
    print(
        f"1D CNN           | RMSE={cnn_metrics['rmse']:.6f} nm | "
        f"MAE={cnn_metrics['mae']:.6f} nm | R2={cnn_metrics['r2']:.6f}"
    )

    metrics = {
        "cross_correlation": {k: v for k, v in cc_metrics.items() if k in {"rmse", "mae", "r2"}},
        "mlp": {k: v for k, v in mlp_metrics.items() if k in {"rmse", "mae", "r2"}},
        "cnn1d": {k: v for k, v in cnn_metrics.items() if k in {"rmse", "mae", "r2"}},
    }

    with open(results_dir / "phase1_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    table_lines = [
        "method,rmse_nm,mae_nm,r2",
        f"cross_correlation,{metrics['cross_correlation']['rmse']:.8f},{metrics['cross_correlation']['mae']:.8f},{metrics['cross_correlation']['r2']:.8f}",
        f"mlp,{metrics['mlp']['rmse']:.8f},{metrics['mlp']['mae']:.8f},{metrics['mlp']['r2']:.8f}",
        f"cnn1d,{metrics['cnn1d']['rmse']:.8f},{metrics['cnn1d']['mae']:.8f},{metrics['cnn1d']['r2']:.8f}",
    ]
    (results_dir / "phase1_metrics.csv").write_text("\n".join(table_lines), encoding="utf-8")
    print(f"Saved metrics to {results_dir / 'phase1_metrics.json'}")
    print(f"Saved table to {results_dir / 'phase1_metrics.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase1 pipeline: Dataset_A + baselines")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "config" / "phase1.yaml"),
        help="Path to phase1 yaml config",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate Dataset_A even if existing file is present",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    run_phase1(config, regenerate=args.regenerate)
