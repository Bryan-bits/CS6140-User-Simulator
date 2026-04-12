from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import torch

from data_loader import build_interactions, load_movielens_1m, train_valid_test_split
from feature_engineering import (
    build_eval_sequence_examples,
    build_static_feature_frame,
    build_training_sequence_examples,
)
from gru4rec_simulator import GRU4RecConfig, GRU4RecUserSimulator
from xgboost_simulator import XGBoostConfig, XGBoostUserSimulator


RESULT_COLUMNS = [
    "model",
    "pooling",
    "max_depth",
    "hidden_size",
    "max_seq_len",
    "device",
    "valid_auc",
    "valid_accuracy",
    "valid_f1",
    "valid_positive_rate",
    "test_auc",
    "test_accuracy",
    "test_f1",
    "test_positive_rate",
]


def run_xgboost_experiment(data_dir: str, pooling: str, max_depth: int) -> Dict[str, float]:
    bundle = load_movielens_1m(data_dir)
    interactions = build_interactions(bundle)
    splits = train_valid_test_split(interactions)

    train_frame, artifacts = build_static_feature_frame(splits["train"], splits["train"], bundle.movies, pooling=pooling)
    valid_frame, _ = build_static_feature_frame(splits["train"], splits["valid"], bundle.movies, pooling=pooling)
    test_frame, _ = build_static_feature_frame(splits["train"], splits["test"], bundle.movies, pooling=pooling)

    x_train = train_frame[artifacts.feature_columns]
    y_train = train_frame["like"]
    x_valid = valid_frame[artifacts.feature_columns]
    y_valid = valid_frame["like"]
    x_test = test_frame[artifacts.feature_columns]
    y_test = test_frame["like"]

    simulator = XGBoostUserSimulator(XGBoostConfig(max_depth=max_depth))
    simulator.fit(x_train, y_train)

    metrics = {f"valid_{key}": value for key, value in simulator.evaluate(x_valid, y_valid).items()}
    metrics.update({f"test_{key}": value for key, value in simulator.evaluate(x_test, y_test).items()})
    return metrics


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return device


def run_gru4rec_experiment(
    data_dir: str,
    hidden_size: int,
    max_seq_len: int,
    device: str,
    epochs: int,
    batch_size: int,
) -> Dict[str, float]:
    bundle = load_movielens_1m(data_dir)
    interactions = build_interactions(bundle)
    splits = train_valid_test_split(interactions)

    train_sequences, train_labels, movie_to_index = build_training_sequence_examples(
        splits["train"],
        max_seq_len=max_seq_len,
    )
    valid_sequences, valid_labels = build_eval_sequence_examples(
        splits["train"],
        splits["valid"],
        movie_to_index,
        max_seq_len=max_seq_len,
    )
    test_sequences, test_labels = build_eval_sequence_examples(
        splits["train"],
        splits["test"],
        movie_to_index,
        max_seq_len=max_seq_len,
    )

    simulator = GRU4RecUserSimulator(
        num_items=len(movie_to_index),
        config=GRU4RecConfig(
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
        ),
    )
    simulator.fit(train_sequences, train_labels)

    metrics = {f"valid_{key}": value for key, value in simulator.evaluate(valid_sequences, valid_labels).items()}
    metrics.update({f"test_{key}": value for key, value in simulator.evaluate(test_sequences, test_labels).items()})
    return metrics


def append_result_row(results_path: str | Path, row: Dict[str, object]) -> None:
    results_file = Path(results_path)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = results_file.exists() and results_file.stat().st_size > 0
    normalized_row = {column: row.get(column, "") for column in RESULT_COLUMNS}

    with results_file.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(normalized_row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train user simulator baselines.")
    parser.add_argument("--model", choices=["xgboost", "gru4rec"], required=True)
    parser.add_argument("--data-dir", default="data/ml-1m")
    parser.add_argument("--results-path", default="experiments/results.csv")
    parser.add_argument("--pooling", choices=["mean", "recency_weighted"], default="mean")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.model == "xgboost":
        metrics = run_xgboost_experiment(args.data_dir, args.pooling, args.max_depth)
        row = {"model": "xgboost", "pooling": args.pooling, "max_depth": args.max_depth, **metrics}
    else:
        resolved_device = resolve_device(args.device)
        print(f"Running GRU4Rec with device={resolved_device}, hidden_size={args.hidden_size}, max_seq_len={args.max_seq_len}, epochs={args.epochs}, batch_size={args.batch_size}")
        metrics = run_gru4rec_experiment(
            args.data_dir,
            args.hidden_size,
            args.max_seq_len,
            resolved_device,
            args.epochs,
            args.batch_size,
        )
        row = {
            "model": "gru4rec",
            "hidden_size": args.hidden_size,
            "max_seq_len": args.max_seq_len,
            "device": resolved_device,
            **metrics,
        }

    append_result_row(args.results_path, row)
    print(row)


if __name__ == "__main__":
    main()
