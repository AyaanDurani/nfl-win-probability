"""
Build the model's feature table from raw play-by-play.

One row per real play, every column from the OFFENSE's point of view,
plus the label: did the team with the ball win the game?

Run with:  python -m src.features
Output:    data/processed/features.parquet
"""

from pathlib import Path

import polars as pl

from src.data import load_pbp


#find the project root from this file's location
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"

# The inputs the model will see.
# If training and scoring ever used different feature lists, the model would silently break.
FEATURE_COLS = [
    "score_differential",          # offense score minus defense score
    "game_seconds_remaining",      # 3600 at kickoff, 0 at the end
    "half_seconds_remaining",      # matters for end-of-half situations
    "down",
    "ydstogo",
    "yardline_100",                # yards to the opponent's end zone
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "is_home",                     # 1 if the offense is the home team
    "posteam_spread",              # pregame spread, offense's perspective
    "receive_2h_ko",               # 1 if the offense gets the 2nd-half kickoff
    "spread_time",                 # engineered: spread fading over the game
    "diff_time_ratio",             # engineered: lead growing in importance
]

# Not model inputs, but we keep them:
#   ID columns let us split by season (CP4) and look up specific games
#   "wp" is nflfastR's prediction
ID_COLS = ["game_id", "season", "play_id", "posteam", "home_team", "away_team"]
BENCHMARK_COLS = ["wp"]
LABEL_COL = "label"


# second-half kickoff

def add_second_half_receiver(pbp: pl.DataFrame) -> pl.DataFrame:
    opening_receiver = (
        pbp
        # Only first-quarter kickoffs. The opening kick is one of these.
        .filter((pl.col("play_type") == "kickoff") & (pl.col("qtr") == 1))
        # For each game, sort its kickoffs by play_id (play order) and take
        # the first one's receiving team.
        .group_by("game_id")
        .agg(pl.col("posteam").sort_by("play_id").first().alias("opening_receiver"))
    )

    # Attach opening_receiver to every play of its game.
    # how="left" keeps every play even if a game somehow has no match.
    pbp = pbp.join(opening_receiver, on="game_id", how="left")

    # Flag is 1 only in the first half (qtr 1-2), for the team that did NOT
    # receive the opening kick. After halftime, the advantage has been used.
    return pbp.with_columns(
        (
            (pl.col("qtr") <= 2)
            & (pl.col("posteam") != pl.col("opening_receiver"))
        )
        .fill_null(False)          # if opening_receiver was missing, treat as 0
        .cast(pl.Int8)             # True/False -> 1/0, which models need
        .alias("receive_2h_ko")
    )


# filter, flip perspective, engineer

def build_features(pbp: pl.DataFrame) -> pl.DataFrame:
    """Turn raw play-by-play into the model-ready feature table."""

    # Step 1 (see above). Must come first.
    df = add_second_half_receiver(pbp)

    # Step 2: keep only real plays from games that had a winner.
    df = df.filter(
        pl.col("down").is_not_null()          # no down = not a play
        & pl.col("posteam").is_not_null()     # need to know who has the ball
        & (pl.col("result") != 0)             # ties have no winner
    )

    # A reusable expression: "is the offense the home team?"
    # Defining it once avoids typing the comparison three times.
    is_home = pl.col("posteam") == pl.col("home_team")

    # Step 3: flip home-perspective columns into the offense's perspective.
    df = df.with_columns(
        is_home.cast(pl.Int8).alias("is_home"),

        # spread_line is positive when HOME is favored.
        # Home offense: keep it. Away offense: flip the sign.
        pl.when(is_home)
        .then(pl.col("spread_line"))
        .otherwise(-pl.col("spread_line"))
        .alias("posteam_spread"),

        # The label. result = home score - away score.
        # Home offense won if result > 0; away offense won if result < 0.
        pl.when(is_home)
        .then(pl.col("result") > 0)
        .otherwise(pl.col("result") < 0)
        .cast(pl.Int8)
        .alias(LABEL_COL),

        # Fraction of the game played: 0.0 at kickoff, 1.0 at the end.
        # clip() keeps it in [0, 1] so overtime can't push it out of range.
        ((3600 - pl.col("game_seconds_remaining")) / 3600)
        .clip(0, 1)
        .alias("elapsed_share"),
    )

    # Step 4: engineered features. This is a SEPARATE with_columns because
    # Polars computes everything inside one with_columns at the same time,
    # so they can't use posteam_spread or elapsed_share, which were only
    # created in the block above.
    decay = (-4 * pl.col("elapsed_share")).exp()   # 1.0 at kickoff, ~0.02 at the end
    df = df.with_columns(
        (pl.col("posteam_spread") * decay).alias("spread_time"),
        (pl.col("score_differential") / decay).alias("diff_time_ratio"),
    )

    # Keep only the columns we need, in a fixed order
    return df.select(ID_COLS + FEATURE_COLS + BENCHMARK_COLS + [LABEL_COL])


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw play-by-play...")
    pbp = load_pbp()

    print("Building features...")
    features = build_features(pbp)

    # Drop rows with a missing feature or label, and SAY how many.
    # Silently dropping rows is how datasets quietly shrink without anyone
    # noticing. Always report it.
    before = features.height
    features = features.drop_nulls(subset=FEATURE_COLS + [LABEL_COL])
    print(f"Dropped {before - features.height:,} rows with missing values")

    # Sanity checks: fail loudly if something is obviously wrong
    assert features.height > 0, "Feature table is empty"
    label_rate = features[LABEL_COL].mean()
    assert 0.4 < label_rate < 0.6, f"Label rate {label_rate:.3f} looks wrong"

    features.write_parquet(FEATURES_PATH)
    print(f"Saved {features.height:,} plays x {features.width} columns")
    print(f"Label rate (share of plays where offense won): {label_rate:.3f}")

    # Expect TB plays labeled 1 and HOU plays labeled 0.
    print("\nSpot check: 2025_02_TB_HOU")
    print(
        features.filter(pl.col("game_id") == "2025_02_TB_HOU")
        .group_by("posteam")
        .agg(pl.col(LABEL_COL).mean().alias("label"), pl.len().alias("plays"))
    )