import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "artifacts" / "testing_diff_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def pick_file(preferred: str, fallback: str) -> Path:
    p = ROOT / preferred
    return p if p.exists() else ROOT / fallback


X_TRAIN_PATH = pick_file("X_train_diff.csv", "X_train.csv")
X_TEST_PATH = pick_file("X_test_diff.csv", "X_test.csv")
Y_TRAIN_PATH = pick_file("y_train_diff.csv", "y_train.csv")
Y_TEST_PATH = pick_file("y_test_diff.csv", "y_test.csv")
FUND_AVAIL_PATH = pick_file(
    "test_fundamentals_available_diff.csv",
    "test_fundamentals_available.csv",
)
RECENCY_PATH = pick_file(
    "test_fundamental_recency_days_diff.csv",
    "test_fundamental_recency_days.csv",
)


torch.manual_seed(42)
np.random.seed(42)


class VolatilityMLP(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residual = y_true - y_pred
    mse = float(np.mean(residual ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(residual ** 2) / denom) if denom > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_and_predict(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    y_train: np.ndarray,
    *,
    epochs: int = 300,
    lr: float = 1e-3,
) -> np.ndarray:
    X_train = torch.tensor(X_train_df.values.astype(np.float32))
    X_test = torch.tensor(X_test_df.values.astype(np.float32))
    y_train_t = torch.tensor(y_train.reshape(-1, 1).astype(np.float32))

    model = VolatilityMLP(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test).squeeze().cpu().numpy()
    return preds


def get_feature_sets(columns: list[str]) -> dict[str, list[str]]:
    quarterly_markers = [
        "total_cash",
        "lfcf",
        "fundamental",
        "days_since_total_cash_report",
        "days_since_lfcf_report",
    ]
    dynamic_keys = ["qoq", "yoy", "trend", "improving", "to_total_cash"]
    quarterly_base = [
        "total_cash",
        "lfcf",
        "fundamental",
        "days_since_total_cash_report",
        "days_since_lfcf_report",
    ]

    monthly_only = [c for c in columns if not any(k in c for k in quarterly_markers)]
    quarterly_levels = [
        c
        for c in columns
        if any(k in c for k in quarterly_base) and not any(d in c for d in dynamic_keys)
    ]
    quarterly_dynamics = [c for c in columns if any(d in c for d in dynamic_keys)]

    return {
        "monthly_only": monthly_only,
        "monthly_plus_quarterly_levels": sorted(set(monthly_only + quarterly_levels)),
        "monthly_plus_levels_plus_dynamics": sorted(
            set(monthly_only + quarterly_levels + quarterly_dynamics)
        ),
    }


def diagnostic_summary(
    X_train: pd.DataFrame,
    y_true: np.ndarray,
    pred_map: dict[str, np.ndarray],
    feature_sets: dict[str, list[str]],
) -> dict:
    summary = {}
    dynamic_cols = [
        c
        for c in X_train.columns
        if any(k in c for k in ["qoq", "yoy", "trend", "improving", "to_total_cash"])
    ]
    if dynamic_cols:
        dyn = X_train[dynamic_cols]
        with np.errstate(invalid="ignore"):
            z = (dyn - dyn.mean()) / (dyn.std(ddof=0) + 1e-12)
        extreme_rate = float((np.abs(z.values) > 5).mean())
        summary["dynamic_outlier_rate_gt_5std"] = extreme_rate
        summary["dynamic_feature_count"] = len(dynamic_cols)
    else:
        summary["dynamic_outlier_rate_gt_5std"] = None
        summary["dynamic_feature_count"] = 0

    if FUND_AVAIL_PATH.exists():
        fund_avail = pd.read_csv(FUND_AVAIL_PATH).iloc[:, 0].values
        summary["fundamentals_available_rate_test"] = float(np.mean(fund_avail >= 0.5))

    if RECENCY_PATH.exists():
        recency = pd.read_csv(RECENCY_PATH).iloc[:, 0].values
        recency = recency[~np.isnan(recency)]
        if recency.size:
            summary["recency_days_p50"] = float(np.percentile(recency, 50))
            summary["recency_days_p90"] = float(np.percentile(recency, 90))

    perf = {name: regression_metrics(y_true, pred) for name, pred in pred_map.items()}
    summary["metrics"] = perf
    best = min(perf, key=lambda n: perf[n]["rmse"])
    summary["best_variant"] = best
    summary["interpretation"] = (
        "Quarterly levels/dynamics can underperform when they are sparse, stale, or noisy, "
        "adding dimensionality that increases generalization error. Dynamic transforms are "
        "especially sensitive to outliers and imperfect fundamental availability."
    )
    return summary


def plot_metrics(metrics_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, ["rmse", "mae", "r2"]):
        vals = metrics_df[metric].values
        ax.bar(metrics_df["variant"], vals)
        ax.set_title(metric.upper())
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ablation_metrics.png", dpi=180)
    plt.close(fig)


def plot_scatter(y_true: np.ndarray, pred_map: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    keys = list(pred_map.keys())
    for ax, k in zip(axes, keys):
        y_pred = pred_map[k]
        ax.scatter(y_true, y_pred, s=8, alpha=0.35)
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
        ax.set_title(k)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "actual_vs_predicted_scatter.png", dpi=180)
    plt.close(fig)


def plot_residual_hist(y_true: np.ndarray, pred_map: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    keys = list(pred_map.keys())
    for ax, k in zip(axes, keys):
        resid = y_true - pred_map[k]
        ax.hist(resid, bins=60, alpha=0.8)
        ax.set_title(f"{k} residuals")
        ax.set_xlabel("y_true - y_pred")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "residual_distributions.png", dpi=180)
    plt.close(fig)


def plot_predictions_by_variant(y_true: np.ndarray, pred_map: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    keys = list(pred_map.keys())
    n = min(300, len(y_true))
    idx = np.arange(n)

    for ax, k in zip(axes, keys):
        y_pred = pred_map[k][:n]
        ax.plot(idx, y_true[:n], label="Actual", linewidth=1.8)
        ax.plot(idx, y_pred, label="Predicted", linewidth=1.6, alpha=0.85)
        ax.set_title(k)
        ax.set_ylabel("Volatility")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Test sample index")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "predictions_all_three_variants.png", dpi=180)
    plt.close(fig)


def main() -> None:
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).iloc[:, 0].values.astype(np.float32)
    y_test = pd.read_csv(Y_TEST_PATH).iloc[:, 0].values.astype(np.float32)

    if np.isnan(X_train.values).any() or np.isnan(X_test.values).any():
        raise ValueError("NaNs found in X data. Rebuild CSVs before running.")
    if np.isnan(y_train).any() or np.isnan(y_test).any():
        raise ValueError("NaNs found in y data. Rebuild CSVs before running.")

    feature_sets = get_feature_sets(list(X_train.columns))

    pred_map = {}
    rows = []
    for name, cols in feature_sets.items():
        cols = [c for c in cols if c in X_train.columns and c in X_test.columns]
        y_pred = train_and_predict(X_train[cols], X_test[cols], y_train)
        pred_map[name] = y_pred
        m = regression_metrics(y_test, y_pred)
        rows.append({"variant": name, "n_features": len(cols), **m})
        print(
            f"{name} | n_features={len(cols)} | "
            f"RMSE={m['rmse']:.6f} | MAE={m['mae']:.6f} | R2={m['r2']:.6f}"
        )

    metrics_df = pd.DataFrame(rows).sort_values("rmse")
    metrics_df.to_csv(OUT_DIR / "ablation_metrics.csv", index=False)

    summary = diagnostic_summary(X_train, y_test, pred_map, feature_sets)
    with open(OUT_DIR / "diagnostic_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_metrics(metrics_df)
    plot_scatter(y_test, pred_map)
    plot_residual_hist(y_test, pred_map)
    plot_predictions_by_variant(y_test, pred_map)

    best = metrics_df.iloc[0]
    print("\nBest variant:", best["variant"])
    print("Outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
