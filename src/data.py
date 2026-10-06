from pathlib import Path

import nflreadpy as nfl
import polars as pl


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

SEASONS = range(2010, 2026)

def season_path(season: int) -> Path:
    """Return the file path for one season, e.g. data/raw/pbp_2015.parquet."""
    return RAW_DIR / f"pbp_{season}.parquet"

def download_season(season: int, force: bool = False) -> None:
    return RAW_DIR / f"pbp_{season}.parquet"


def download_season(season: int, force: bool = False) -> None:
    path = season_path(season)

    if path.exists() and not force:
        print(f"{season}: cached, skipping")
        return

    print(f"{season}: downloading...")

    df = nfl.load_pbp(seasons=[season])

    # Save it to disk as Parquet (compressed, keeps column types)
    df.write_parquet(path)

    print(f"{season}: saved {df.height:,} plays")


def load_pbp(seasons=SEASONS) -> pl.DataFrame:
    """Load the cached seasons from disk and stack them into one DataFrame.

    Every later step (features, modeling, the pipeline) calls this
    instead of downloading anything.
    """
    # Read each season's file into its own DataFrame (a list comprehension:
    # "for each season s, read its file" builds a list of DataFrames)
    frames = [pl.read_parquet(season_path(s)) for s in seasons]

    # Stack them vertically into one big table.
    # nflverse has added/changed columns over the years, so 2010 and 2025
    # don't have identical columns. A normal concat would crash on that.
    # "diagonal_relaxed" matches columns by name, fills missing ones with
    # nulls, and reconciles small type differences (e.g. int vs float).
    return pl.concat(frames, how="diagonal_relaxed")


# This block only runs when you execute the file directly
# (python -m src.data). It does NOT run when another script does
# "from src.data import load_pbp", which is what we want.
if __name__ == "__main__":
    # Create data/raw if it doesn't exist yet.
    # parents=True also creates data/ if needed; exist_ok=True means
    # "don't error if the folder is already there".
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Download each season (skips any that are already cached)
    for season in SEASONS:
        download_season(season)

    # Load everything back as a sanity check and report the size.
    # df.width = number of columns
    df = load_pbp()
    print(f"Total: {df.height:,} plays x {df.width} columns")