"""
Train the logistic regression baseline and score it on the validation season.

Run with:  python -m src.train_baseline
Output:    models/baseline.joblib
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLS
from src.metrics import evaluate
from src.split import load_splits, xy

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
BASELINE_PATH = MODELS_DIR / "baseline.joblib"


if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)
    train, val, _ = load_splits()
    print(f"Train: {len(train):,} plays (2010-2023)")
    print(f"Val:   {len(val):,} plays (2024)")

    X_train, y_train = xy(train)
    X_val, y_val = xy(val)


    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    p_train = model.predict_proba(X_train)[:, 1]
    p_val = model.predict_proba(X_val)[:, 1]

    coin_flip = np.full(len(y_val), 0.5)
    base_rate = np.full(len(y_val), y_train.mean())

    results = pd.DataFrame({
        "coin flip (val)": evaluate(y_val, coin_flip),
        "base rate (val)": evaluate(y_val, base_rate),
        "logistic (train)": evaluate(y_train, p_train),
        "logistic (val)": evaluate(y_val, p_val),
    }).T
    print("\n", results.round(4))

    # bigger magnitude = bigger influence.
    # Positive = pushes win probability UP; negative = pushes it DOWN.
    coefs = pd.Series(model[-1].coef_[0], index=FEATURE_COLS)
    coefs = coefs.sort_values(key=abs, ascending=False)
    print("\nCoefficients (scaled features, biggest influence first):")
    print(coefs.round(3))

    joblib.dump(model, BASELINE_PATH)
    print(f"\nSaved model to {BASELINE_PATH}")