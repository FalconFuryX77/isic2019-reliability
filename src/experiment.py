"""
Experiment helpers: unique experiment IDs, dirs, config snapshots.
"""
import shutil
import uuid
from pathlib import Path

from src.config import load_config
from src.utils import save_json


def create_experiment_dir(cfg, exp_id, copy_config=True):
    """Create experiments/<exp_id> and copy the resolved config in."""
    base = Path(cfg.paths.experiments_dir)
    exp_dir = base / exp_id
    if exp_dir.exists():
        exp_dir = base / f"{exp_id}_{uuid.uuid4().hex[:4]}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    if copy_config:
        save_json(cfg.to_dict(), exp_dir / "experiment_config.json")
    return exp_dir


def save_training_curves(history_csv, exp_dir, model_name=""):
    """Plot loss/accuracy/macro-F1 curves from training_history.csv."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col_train, col_val, title in [
        (axes[0], "train_loss", "val_loss", "Loss"),
        (axes[1], "train_accuracy", "val_accuracy", "Accuracy"),
        (axes[2], None, "val_f1_macro", "Macro F1"),
    ]:
        if col_train is not None:
            ax.plot(df["epoch"], df[col_train], label="Train")
        ax.plot(df["epoch"], df[col_val], label="Validation")
        ax.set_title(f"{title} — {model_name}".strip())
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.axes[1].figure.savefig(Path(exp_dir) / "accuracy_curve.png", dpi=150)
    fig.axes[0].figure.savefig(Path(exp_dir) / "loss_curve.png", dpi=150)
    fig.axes[2].figure.savefig(Path(exp_dir) / "macro_f1_curve.png", dpi=150)
    plt.close(fig)
    print(f"  Training curves saved to {exp_dir}")