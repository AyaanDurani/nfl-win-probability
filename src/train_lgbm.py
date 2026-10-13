import itertools

import joblib
import lightgbm as lgb
import pandas as pd

from src.features import FEATURE_COLS
from src.metrics import evaluate
from src.split import load_splits, xy
from src.train_baseline import BASELINE_PATH, MODELS_DIR

LGBM_PATH = MODELS_DIR / "lgbm.joblib"

# Small grid: 3 tree sizes x 2 leaf minimums = 6 models
GRID = {
    "num_leaves": [15, 31, 63],
    "min_child_samples": [100, 500],
}


def train_one(params, X_train, y_train, X_val, y_val):
    model = lgb.LGBMClassifier(
        n_estimators=3000,        # upper limit; early stopping picks the real number
        learning_rate=0.05,
        verbose=-1,               # silence LightGBM's training logs
        **params,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


if __name__ == "__main__":
    train, val, _ = load_splits()    # test stays locked until CP6
    X_train, y_train = xy(train)
    X_val, y_val = xy(val)

    # Try every combination in the grid
    runs = []
    for num_leaves, min_child in itertools.product(*GRID.values()):
        params = {"num_leaves": num_leaves, "min_child_samples": min_child}
        model = train_one(params, X_train, y_train, X_val, y_val)
        scores = evaluate(y_val, model.predict_proba(X_val)[:, 1])
        runs.append({**params, "trees": model.best_iteration_, **scores, "model": model})
        print(f"{params} -> trees={model.best_iteration_}, val log loss={scores['log_loss']:.4f}")

    # Pick the run with the lowest validation log loss
    best = min(runs, key=lambda r: r["log_loss"])
    best_model = best["model"]
    print(f"\nBest: num_leaves={best['num_leaves']}, "
    f"min_child_samples={best['min_child_samples']}, trees={best['trees']}")

    # Head-to-head with the baseline on the same 2024 plays
    baseline = joblib.load(BASELINE_PATH)
    comparison = pd.DataFrame({
        "logistic (val)": evaluate(y_val, baseline.predict_proba(X_val)[:, 1]),
        "lightgbm (val)": evaluate(y_val, best_model.predict_proba(X_val)[:, 1]),
        "lightgbm (train)": evaluate(y_train, best_model.predict_proba(X_train)[:, 1]),
    }).T
    print("\n", comparison.round(4))

    # Importance by "gain": how much each feature reduced the error in total
    gain = best_model.booster_.feature_importance(importance_type="gain")
    importance = pd.Series(gain, index=FEATURE_COLS)
    importance = (importance / importance.sum()).sort_values(ascending=False)
    print("\nFeature importance (share of total gain):")
    print(importance.round(3))

    joblib.dump(best_model, LGBM_PATH)
    print(f"\nSaved model to {LGBM_PATH}")