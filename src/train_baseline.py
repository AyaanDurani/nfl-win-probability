"""
Train the logistic regression baseline and score it on the validation season.

Run with:  python -m src.train_baseline
Output:    models/baseline.joblib
"""

from pathlib import Path

import joblib                     # saves/loads trained models (installed with scikit-learn)
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

    # The underscore means "we're deliberately ignoring this".
    # The test set stays locked until CP6.
    train, val, _ = load_splits()
    print(f"Train: {len(train):,} plays (2010-2023)")
    print(f"Val:   {len(val):,} plays (2024)")

    X_train, y_train = xy(train)
    X_val, y_val = xy(val)

    # Pipeline = scaler then model, treated as one object.
    # .fit() learns the scaler's means/stds AND the model's weights from
    # training data only; .predict_proba() applies both in order.
    # max_iter=1000 gives the solver enough steps to converge.
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    # predict_proba returns two columns: [P(lose), P(win)].
    # [:, 1] means "all rows, second column", i.e. P(offense wins).
    p_train = model.predict_proba(X_train)[:, 1]
    p_val = model.predict_proba(X_val)[:, 1]

    # Reference predictions. A model is only as impressive as
    # the dumb thing it beats.
    coin_flip = np.full(len(y_val), 0.5)             # always say 50%
    base_rate = np.full(len(y_val), y_train.mean())  # always say ~50.7%

    # One row per approach, metrics as columns. .T flips rows/columns.
    results = pd.DataFrame({
        "coin flip (val)": evaluate(y_val, coin_flip),
        "base rate (val)": evaluate(y_val, base_rate),
        "logistic (train)": evaluate(y_train, p_train),
        "logistic (val)": evaluate(y_val, p_val),
    }).T
    print("\n", results.round(4))

    # What did the model learn? model[-1] is the last pipeline step
    # (the LogisticRegression). Because features were scaled, coefficient
    # sizes are comparable: bigger magnitude = bigger influence.
    # Positive = pushes win probability UP; negative = pushes it DOWN.
    coefs = pd.Series(model[-1].coef_[0], index=FEATURE_COLS)
    coefs = coefs.sort_values(key=abs, ascending=False)
    print("\nCoefficients (scaled features, biggest influence first):")
    print(coefs.round(3))

    joblib.dump(model, BASELINE_PATH)
    print(f"\nSaved model to {BASELINE_PATH}")