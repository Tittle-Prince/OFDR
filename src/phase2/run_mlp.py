from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase2.common import load_config, load_phase2_data, method_dir, save_method_outputs, set_seed
from phase2.nn_models import MLPRegressor
from phase2.nn_train import make_loaders, predict, train_regressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase2 baseline: MLP")
    parser.add_argument("--config", type=str, default="config/phase2.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["phase2"]["seed"]))
    data = load_phase2_data(cfg)

    train_loader, val_loader = make_loaders(
        data.x,
        data.y_dlambda,
        data.idx_train,
        data.idx_val,
        batch_size=int(cfg["train"]["batch_size"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressor(input_dim=data.x.shape[1]).to(device)
    model = train_regressor(model, train_loader, val_loader, cfg["train"], device)

    y_true = data.y_dlambda[data.idx_test]
    y_pred = predict(model, data.x, data.idx_test, device)
    metrics = save_method_outputs(cfg, "mlp", y_true, y_pred)

    out_dir = method_dir(cfg, "mlp")
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"Saved model to: {out_dir / 'model.pt'}")
    print(f"MLP | RMSE={metrics['rmse']:.6f} nm | MAE={metrics['mae']:.6f} nm | R2={metrics['r2']:.6f}")


if __name__ == "__main__":
    main()

