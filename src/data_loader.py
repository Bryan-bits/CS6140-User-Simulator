from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


RATINGS_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
MOVIES_COLUMNS = ["movie_id", "title", "genres"]
USERS_COLUMNS = ["user_id", "gender", "age", "occupation", "zip_code"]


@dataclass
class MovieLensBundle:
    ratings: pd.DataFrame
    movies: pd.DataFrame
    users: pd.DataFrame


def _read_dat(path: Path, columns: List[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=columns,
        encoding="latin-1",
    )


def load_movielens_1m(data_dir: str | Path = "data/ml-1m") -> MovieLensBundle:
    """Load MovieLens 1M raw files from the standard folder layout."""
    data_path = Path(data_dir)
    ratings = _read_dat(data_path / "ratings.dat", RATINGS_COLUMNS)
    movies = _read_dat(data_path / "movies.dat", MOVIES_COLUMNS)
    users = _read_dat(data_path / "users.dat", USERS_COLUMNS)

    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings["like"] = (ratings["rating"] >= 4).astype(int)
    ratings = ratings.sort_values(["user_id", "timestamp", "movie_id"]).reset_index(drop=True)

    return MovieLensBundle(ratings=ratings, movies=movies, users=users)


def build_interactions(bundle: MovieLensBundle) -> pd.DataFrame:
    """Join ratings with movie and user metadata."""
    interactions = (
        bundle.ratings.merge(bundle.movies, on="movie_id", how="left")
        .merge(bundle.users, on="user_id", how="left")
        .sort_values(["user_id", "timestamp", "movie_id"])
        .reset_index(drop=True)
    )
    interactions["genres"] = interactions["genres"].fillna("(no genres listed)")
    return interactions


def train_valid_test_split(
    interactions: pd.DataFrame,
    min_history: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Split each user's history chronologically.

    Train uses all but the final two interactions, validation is the second-to-last,
    and test is the final interaction.
    """
    eligible = interactions.groupby("user_id").filter(lambda frame: len(frame) >= min_history).copy()
    eligible["rank_desc"] = eligible.groupby("user_id")["timestamp"].rank(method="first", ascending=False)

    train = eligible[eligible["rank_desc"] > 2].drop(columns="rank_desc").reset_index(drop=True)
    valid = eligible[eligible["rank_desc"] == 2].drop(columns="rank_desc").reset_index(drop=True)
    test = eligible[eligible["rank_desc"] == 1].drop(columns="rank_desc").reset_index(drop=True)
    return {"train": train, "valid": valid, "test": test}


def build_user_histories(interactions: pd.DataFrame) -> Dict[int, List[Tuple[int, int]]]:
    """Return ordered user histories as (movie_id, like) pairs."""
    ordered = interactions.sort_values(["user_id", "timestamp", "movie_id"])
    return {
        int(user_id): list(zip(frame["movie_id"].astype(int), frame["like"].astype(int)))
        for user_id, frame in ordered.groupby("user_id")
    }
