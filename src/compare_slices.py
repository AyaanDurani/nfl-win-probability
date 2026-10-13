import joblib
import pandas as pd

from src.metrics import evaluate
from src.split import load_splits, xy
from src.train_baseline import BASELINE_PATH
from src.train_lgbm import LGBM_PATH

_, val, _ = load_splits()
X_val, y_val = xy(val)

val = val.assign(
    p_logistic=joblib.load(BASELINE_PATH).predict_proba(X_val)[:, 1],
    p_lgbm=joblib.load(LGBM_PATH).predict_proba(X_val)[:, 1],
)

secs = val["game_seconds_remaining"]
margin = val["score_differential"].abs()

# Each slice is a True/False filter over the validation plays
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
    s = val[mask]
    logistic = evaluate(s["label"], s["p_logistic"])["log_loss"]
    lgbm = evaluate(s["label"], s["p_lgbm"])["log_loss"]
    rows.append({
        "slice": name,
        "plays": len(s),
        "logistic": logistic,
        "lightgbm": lgbm,
        "lgbm better by": logistic - lgbm,   # positive = LightGBM wins
    })

print(pd.DataFrame(rows).round(4).to_string(index=False))