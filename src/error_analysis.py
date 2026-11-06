import joblib
import numpy as np
import pandas as pd

from src.split import load_splits, xy
from src.train_lgbm import LGBM_PATH

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 70)


def play_log_loss(y, p):
    """Log loss for each play individually (not averaged)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


if __name__ == "__main__":
    _, val, _ = load_splits()
    val = val.dropna(subset=["vegas_wp"]).copy()
    X, y = xy(val)

    val["p_lgbm"] = joblib.load(LGBM_PATH).predict_proba(X)[:, 1]
    val["p_nflfastr"] = val["vegas_wp"].clip(1e-6, 1 - 1e-6)
    val["loss_lgbm"] = play_log_loss(val["label"], val["p_lgbm"])
    val["loss_nflfastr"] = play_log_loss(val["label"], val["p_nflfastr"])

    # 1. How is overtime encoded? Check before trusting any OT numbers.
    ot = val[val["qtr"] == 5]
    print(f"Overtime plays: {len(ot)} across {ot['game_id'].nunique()} games")
    if len(ot):
        print(f"  game_seconds_remaining in OT ranges "
            f"{ot['game_seconds_remaining'].min():.0f} to {ot['game_seconds_remaining'].max():.0f}\n")

    # 2. Situations: is the model worse than nflfastR anywhere?
    half = val["half_seconds_remaining"]
    margin = val["score_differential"].abs()
    situations = {
        "all plays": val["qtr"] > 0,
        "end of 1st half (Q2, last 2 min)": (val["qtr"] == 2) & (half <= 120),
        "end of game (Q4, last 2 min)": (val["qtr"] == 4) & (half <= 120),
        "Q4 one-score": (val["qtr"] == 4) & (margin <= 8),
        "overtime": val["qtr"] == 5,
        "blowouts (17+)": margin >= 17,
    }
    rows = []
    for name, mask in situations.items():
        s = val[mask]
        rows.append({
            "situation": name,
            "plays": len(s),
            "lightgbm": s["loss_lgbm"].mean(),
            "nflfastR": s["loss_nflfastr"].mean(),
            "we're worse by": s["loss_lgbm"].mean() - s["loss_nflfastr"].mean(),
        })
    print("Log loss by situation (positive last column = nflfastR does better):")
    print(pd.DataFrame(rows).round(4).to_string(index=False), "\n")

    # 3. The 15 individual plays the model got most wrong
    cols = ["game_id", "qtr", "game_seconds_remaining", "posteam", "score_differential",
            "down", "ydstogo", "yardline_100", "p_lgbm", "p_nflfastr", "label", "desc"]
    print("15 worst plays (highest individual log loss):")
    print(val.nlargest(15, "loss_lgbm")[cols].round(3).to_string(index=False), "\n")

    # 4. The 10 games the model understood worst, on average
    games = (
        val.groupby("game_id")
        .agg(plays=("loss_lgbm", "size"),
            lgbm=("loss_lgbm", "mean"),
            nflfastR=("loss_nflfastr", "mean"))
        .nlargest(10, "lgbm")
    )
    print("10 hardest games (average log loss):")
    print(games.round(3))