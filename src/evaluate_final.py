from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from src.metrics import evaluate
from src.split import load_splits, xy
from src.train_baseline import BASELINE_PATH
from src.train_lgbm import LGBM_PATH

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def slice_table(df, preds):
    """Log loss per game situation, for each model."""
    secs = df["game_seconds_remaining"]
    margin = df["score_differential"].abs()
    slices = {
        "all plays": secs >= 0,
        "Q1-Q3": secs > 900,
        "Q4": secs <= 900,
        "Q4, one-score game": (secs <= 900) & (margin <= 8),
        "final 2 min, one-score": (secs <= 120) & (margin <= 8),
        "blowouts (17+)": margin >= 17,
    }
    rows = []
    for name, mask in slices.items():
        row = {"slice": name, "plays": int(mask.sum())}
        for model, p in preds.items():
            row[model] = evaluate(df["label"][mask], p[mask.to_numpy()])["log_loss"]
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    REPORTS_DIR.mkdir(exist_ok=True)

    _, _, test = load_splits()

    # Every model must be scored on exactly the same plays, so drop any
    # play where nflfastR has no prediction.
    test = test.dropna(subset=["vegas_wp"])
    X_test, y_test = xy(test)
    print(f"Test: {len(test):,} plays (2025)\n")

    # nflfastR occasionally outputs exactly 0 or 1. If that's ever wrong,
    # log loss becomes infinite, so nudge those values just inside (0, 1).
    nflfastr = np.clip(test["vegas_wp"].to_numpy(), 1e-6, 1 - 1e-6)

    preds = {
        "logistic": joblib.load(BASELINE_PATH).predict_proba(X_test)[:, 1],
        "lightgbm": joblib.load(LGBM_PATH).predict_proba(X_test)[:, 1],
        "nflfastR": nflfastr,
    }

    results = pd.DataFrame({name: evaluate(y_test, p) for name, p in preds.items()}).T
    print("Overall (2025 test):")
    print(results.round(4), "\n")

    print("By game situation (log loss, lower is better):")
    print(slice_table(test, preds).round(4).to_string(index=False), "\n")

    # Calibration: 10 equal-width bins, average prediction vs. actual win rate
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    for name, p in preds.items():
        actual, predicted = calibration_curve(y_test, p, n_bins=10)
        ax.plot(predicted, actual, marker="o", label=name)
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Actual win rate")
    ax.set_title("Calibration on the 2025 season")
    ax.legend()

    out = REPORTS_DIR / "calibration_2025.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved calibration plot to {out}")