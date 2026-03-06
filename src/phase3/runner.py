from __future__ import annotations

import torch

from phase3.common import load_dataset_b, method_dir, save_method_outputs, set_seed
from phase3.models import build_model
from phase3.train_utils import make_loaders, predict, train_model


def run_single_method(cfg: dict, method_key: str, display_name: str) -> dict:
    set_seed(int(cfg["phase3"]["seed"]))
    data = load_dataset_b(cfg)

    train_loader, val_loader = make_loaders(
        data.x,
        data.y_dlambda,
        data.idx_train,
        data.idx_val,
        batch_size=int(cfg["train"]["batch_size"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(method_key=method_key, input_dim=data.x.shape[1]).to(device)
    model = train_model(model, train_loader, val_loader, cfg["train"], device)

    y_true = data.y_dlambda[data.idx_test]
    y_pred = predict(model, data.x, data.idx_test, device)
    metrics = save_method_outputs(cfg, method_key, y_true, y_pred)

    out = method_dir(cfg, method_key)
    torch.save(model.state_dict(), out / "model.pt")
    print(f"{display_name} | RMSE={metrics['rmse']:.6f} nm | MAE={metrics['mae']:.6f} nm | R2={metrics['r2']:.6f}")
    print(f"Saved model to: {out / 'model.pt'}")
    return metrics

