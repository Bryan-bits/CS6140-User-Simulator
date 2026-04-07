from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _split_genres(value: str) -> List[str]:
    if not isinstance(value, str) or not value:
        return ["unknown"]
    return value.split("|")


def build_genre_vocabulary(movies: pd.DataFrame) -> List[str]:
    return sorted({genre for value in movies["genres"].fillna("unknown") for genre in _split_genres(value)})


def _genre_indicator(genres: str, vocabulary: List[str]) -> Dict[str, int]:
    values = set(_split_genres(genres))
    return {f"genre_{genre}": int(genre in values) for genre in vocabulary}


@dataclass
class StaticFeatureArtifacts:
    feature_columns: List[str]
    genre_vocabulary: List[str]


def build_static_feature_frame(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    pooling: str = "mean",
) -> Tuple[pd.DataFrame, StaticFeatureArtifacts]:
    """Create static per-interaction features for the XGBoost baseline."""
    genre_vocabulary = build_genre_vocabulary(movies_df)
    user_profile = _build_user_profiles(train_df, pooling=pooling)

    merged = target_df.copy()
    merged = merged.merge(user_profile, on="user_id", how="left")
    merged = merged.fillna(
        {
            "user_like_rate": train_df["like"].mean(),
            "user_avg_rating": train_df["rating"].mean(),
            "user_history_len": 0,
            "user_recent_like_rate": train_df["like"].mean(),
        }
    )

    movie_stats = (
        train_df.groupby("movie_id")
        .agg(movie_avg_rating=("rating", "mean"), movie_like_rate=("like", "mean"), movie_popularity=("movie_id", "size"))
        .reset_index()
    )
    merged = merged.merge(movie_stats, on="movie_id", how="left")
    merged = merged.fillna(
        {
            "movie_avg_rating": train_df["rating"].mean(),
            "movie_like_rate": train_df["like"].mean(),
            "movie_popularity": 0,
        }
    )

    user_meta_columns = ["gender", "age", "occupation"]
    available_meta_columns = [column for column in user_meta_columns if column in train_df.columns]

    if available_meta_columns:
        user_meta = train_df[["user_id"] + available_meta_columns].drop_duplicates("user_id")
        merged = merged.merge(user_meta, on="user_id", how="left")

    if "gender" not in merged.columns:
        merged["gender"] = -1
    else:
        merged["gender"] = merged["gender"].map({"F": 0, "M": 1}).fillna(-1)

    if "age" not in merged.columns:
        merged["age"] = 0
    else:
        merged["age"] = merged["age"].fillna(train_df["age"].median() if "age" in train_df.columns else 0)

    if "occupation" not in merged.columns:
        merged["occupation"] = -1
    else:
        merged["occupation"] = merged["occupation"].fillna(-1)

    genre_frame = pd.DataFrame(
        [_genre_indicator(genres, genre_vocabulary) for genres in merged["genres"]],
        index=merged.index,
    )
    merged = pd.concat([merged, genre_frame], axis=1)

    feature_columns = [
        "user_like_rate",
        "user_avg_rating",
        "user_history_len",
        "user_recent_like_rate",
        "movie_avg_rating",
        "movie_like_rate",
        "movie_popularity",
        "gender",
        "age",
        "occupation",
    ] + list(genre_frame.columns)

    return merged[feature_columns + ["like"]].copy(), StaticFeatureArtifacts(feature_columns, genre_vocabulary)


def _build_user_profiles(train_df: pd.DataFrame, pooling: str = "mean") -> pd.DataFrame:
    grouped = train_df.sort_values(["user_id", "timestamp"]).groupby("user_id")
    profiles = []

    for user_id, frame in grouped:
        likes = frame["like"].to_numpy(dtype=float)
        ratings = frame["rating"].to_numpy(dtype=float)
        if pooling == "recency_weighted":
            weights = np.linspace(1.0, 2.0, num=len(frame))
            like_rate = float(np.average(likes, weights=weights))
            avg_rating = float(np.average(ratings, weights=weights))
        else:
            like_rate = float(likes.mean())
            avg_rating = float(ratings.mean())

        recent_window = likes[-5:] if len(likes) >= 5 else likes
        profiles.append(
            {
                "user_id": user_id,
                "user_like_rate": like_rate,
                "user_avg_rating": avg_rating,
                "user_history_len": int(len(frame)),
                "user_recent_like_rate": float(recent_window.mean()),
            }
        )

    return pd.DataFrame(profiles)


def build_training_sequence_examples(
    train_df: pd.DataFrame,
    max_seq_len: int = 50,
) -> Tuple[List[List[int]], List[int], Dict[int, int]]:
    """Create autoregressive training examples from chronological user histories."""
    ordered = train_df.sort_values(["user_id", "timestamp", "movie_id"]).reset_index(drop=True)
    movie_ids = sorted(ordered["movie_id"].unique().tolist())
    movie_to_index = {movie_id: idx + 1 for idx, movie_id in enumerate(movie_ids)}

    histories: Dict[int, List[int]] = {}
    sequences: List[List[int]] = []
    labels: List[int] = []

    for _, row in ordered.iterrows():
        user_id = int(row["user_id"])
        history = histories.get(user_id, [])
        if history:
            sequences.append(history[-max_seq_len:].copy())
            labels.append(int(row["like"]))
        histories.setdefault(user_id, []).append(movie_to_index[int(row["movie_id"])])

    return sequences, labels, movie_to_index


def build_eval_sequence_examples(
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
    movie_to_index: Dict[int, int],
    max_seq_len: int = 50,
) -> Tuple[List[List[int]], List[int]]:
    """Create validation or test sequences using train histories as context."""
    ordered_history = history_df.sort_values(["user_id", "timestamp", "movie_id"])
    histories: Dict[int, List[int]] = {}
    sequences: List[List[int]] = []
    labels: List[int] = []

    for _, row in ordered_history.iterrows():
        histories.setdefault(int(row["user_id"]), []).append(movie_to_index.get(int(row["movie_id"]), 0))

    for _, row in target_df.sort_values(["user_id", "timestamp", "movie_id"]).iterrows():
        user_id = int(row["user_id"])
        history = [item for item in histories.get(user_id, []) if item != 0]
        sequences.append(history[-max_seq_len:].copy())
        labels.append(int(row["like"]))

    return sequences, labels
