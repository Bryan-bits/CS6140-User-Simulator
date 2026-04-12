from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRICS = ["test_auc", "test_accuracy", "test_f1"]


def build_run_label(row: pd.Series) -> str:
    if row["model"] == "xgboost":
        pooling = row["pooling"] if pd.notna(row["pooling"]) and str(row["pooling"]).strip() else "unknown"
        depth = int(float(row["max_depth"])) if pd.notna(row["max_depth"]) and str(row["max_depth"]).strip() else "?"
        return f"XGB\n{pooling}\ndepth={depth}"

    hidden = int(float(row["hidden_size"])) if pd.notna(row["hidden_size"]) and str(row["hidden_size"]).strip() else "?"
    seq_len = int(float(row["max_seq_len"])) if pd.notna(row["max_seq_len"]) and str(row["max_seq_len"]).strip() else "?"
    return f"GRU\nh={hidden}\nseq={seq_len}"


def load_results(results_path: Path) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    if df.empty:
        raise ValueError("results.csv is empty. Run experiments first.")

    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    df = df.dropna(subset=METRICS, how="all").copy()
    df["run_label"] = df.apply(build_run_label, axis=1)
    return df


def save_metric_bar_chart(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    ordered = df.sort_values(metric, ascending=False).copy()

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=ordered, x="run_label", y=metric, hue="model", dodge=False, palette="Set2")
    ax.set_title(f"{metric.replace('_', ' ').title()} by Experiment")
    ax.set_xlabel("Run")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=0)

    for patch, value in zip(ax.patches, ordered[metric]):
        ax.annotate(
            f"{value:.3f}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}.png", dpi=200)
    plt.close()


def save_metric_line_chart(df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = df.copy().reset_index(drop=True)
    plot_df["run_id"] = plot_df.index + 1
    long_df = plot_df.melt(
        id_vars=["run_id", "model", "run_label"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    )

    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=long_df, x="run_id", y="value", hue="metric", style="model", markers=True, dashes=False)
    ax.set_title("Metric Trends Across Runs")
    ax.set_xlabel("Run Order in results.csv")
    ax.set_ylabel("Score")
    ax.set_xticks(plot_df["run_id"])
    ax.set_ylim(0.5, min(1.0, long_df["value"].max() + 0.05))
    plt.tight_layout()
    plt.savefig(output_dir / "metric_trends.png", dpi=200)
    plt.close()


def save_best_model_chart(df: pd.DataFrame, output_dir: Path) -> None:
    best_rows = df.sort_values("test_auc", ascending=False).groupby("model", as_index=False).first()
    best_long = best_rows.melt(
        id_vars=["model", "run_label"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    )

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=best_long, x="metric", y="value", hue="model", palette="Set1")
    ax.set_title("Best Run per Model")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")

    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:.3f}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "best_model_comparison.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot experiment metrics from results.csv")
    parser.add_argument("--results-path", default="experiments/results.csv")
    parser.add_argument("--output-dir", default="experiments/plots")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")

    results_path = Path(args.results_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(results_path)

    for metric in METRICS:
        save_metric_bar_chart(df, metric, output_dir)

    save_metric_line_chart(df, output_dir)
    save_best_model_chart(df, output_dir)

    print(f"Saved plots to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
