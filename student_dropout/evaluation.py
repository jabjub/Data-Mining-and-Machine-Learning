"""
Model evaluation: metrics aggregation, comparison plots, ROC curves,
confusion matrices, and Wilcoxon statistical tests (project plan §9, §11).

Wilcoxon signed-rank test (course Exercise-3 pattern):
  For each pair of competing models, we test whether the per-fold F1-macro
  distributions come from the same distribution (H₀).  A p-value < α=0.05
  allows rejection of H₀ — i.e., the performance difference is statistically
  significant and not due to random fold variance.
"""

import logging
import os
from itertools import combinations
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
)

from student_dropout.config import EXPERIMENT_DIR, OUTER_CV_FOLDS, RESULTS_DIR

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

ALPHA = 0.05


def _savefig(subdir: str, name: str) -> None:
    path_dir = os.path.join(RESULTS_DIR, subdir)
    os.makedirs(path_dir, exist_ok=True)
    path = os.path.join(path_dir, name)
    plt.savefig(path)
    plt.close()
    logger.info("Saved: %s", path)


# ── Model comparison bar chart ─────────────────────────────────────────────────
def plot_model_comparison(df: pd.DataFrame) -> None:
    """
    Grouped bar chart: mean F1-macro per (model, strategy) with ±1 std error bars.
    This is the central plot for the defence — visually answers which pipeline is best.
    """
    df = df.copy()
    df["label"] = df["model_name"] + "\n(" + df["strategy"] + ")"
    agg = df.groupby("label")["f1_macro"].agg(["mean", "std"]).reset_index()
    agg = agg.sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(agg)))
    bars = ax.bar(
        agg["label"], agg["mean"],
        yerr=agg["std"], capsize=5,
        color=colors, edgecolor="black", width=0.6,
        error_kw={"elinewidth": 1.5, "ecolor": "black"},
    )

    for bar, mean_val in zip(bars, agg["mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{mean_val:.4f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    ax.set_ylabel("Mean F1-Macro (± std over outer folds)", fontsize=11)
    ax.set_title(
        f"Nested CV Model Comparison ({OUTER_CV_FOLDS}-fold outer, F1-Macro)",
        fontweight="bold", fontsize=13,
    )
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random baseline (0.5)")
    ax.legend()
    plt.xticks(rotation=15, ha="right", fontsize=8)
    fig.tight_layout()
    _savefig("experiments", "model_f1_comparison.png")


def plot_f1_boxplot(df: pd.DataFrame) -> None:
    """
    Boxplot of per-fold F1-macro scores – shows variance as well as mean.
    Mirrors the course's Exercise-3 boxplot pattern exactly.
    """
    df = df.copy()
    df["label"] = df["model_name"] + "\n(" + df["strategy"] + ")"
    order = (
        df.groupby("label")["f1_macro"].mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=df, x="label", y="f1_macro", order=order, ax=ax,
        color="steelblue", linewidth=1.2,
    )
    # Highlight median line in red (mirroring course Exercise-3 style)
    for patch in ax.patches:
        patch.set_alpha(0.7)
    ax.set_title("Per-fold F1-Macro Distribution by Pipeline", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("F1-Macro")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    _savefig("experiments", "f1_boxplot.png")


# ── ROC curves ────────────────────────────────────────────────────────────────
def plot_roc_curves(all_results: list[dict[str, Any]]) -> None:
    """
    One ROC curve per (model, strategy) using the predictions accumulated
    across all outer folds (concatenated y_test and y_proba).
    """
    from sklearn.metrics import roc_curve, auc as auc_score

    groups: dict[str, dict] = {}
    for r in all_results:
        if r.get("y_proba") is None:
            continue
        key = f"{r['model_name']} ({r['strategy']})"
        if key not in groups:
            groups[key] = {"y_test": [], "y_proba": []}
        groups[key]["y_test"].extend(r["y_test"].tolist())
        groups[key]["y_proba"].extend(r["y_proba"].tolist())

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 1], [0, 1], "k--", label="Random classifier (AUC=0.50)")

    for label, data in groups.items():
        fpr, tpr, _ = roc_curve(data["y_test"], data["y_proba"])
        roc_auc = auc_score(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.5, label=f"{label}  (AUC={roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves – All Pipelines (Concatenated Outer Folds)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(linestyle="--", alpha=0.6)
    fig.tight_layout()
    _savefig("experiments", "roc_curves.png")


# ── Confusion matrices ────────────────────────────────────────────────────────
def plot_confusion_matrices(all_results: list[dict[str, Any]]) -> None:
    """
    Aggregate confusion matrix per (model, strategy) across all outer folds.
    """
    from sklearn.metrics import confusion_matrix

    out_dir = os.path.join(RESULTS_DIR, "experiments", "confusion_matrices")
    os.makedirs(out_dir, exist_ok=True)

    groups: dict[str, dict] = {}
    for r in all_results:
        key = f"{r['model_name']}_{r['strategy']}"
        if key not in groups:
            groups[key] = {"y_test": [], "y_pred": []}
        groups[key]["y_test"].extend(r["y_test"].tolist())
        groups[key]["y_pred"].extend(r["y_pred"].tolist())

    for key, data in groups.items():
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            data["y_test"],
            data["y_pred"],
            display_labels=["Non-Dropout", "Dropout"],
            cmap="Blues",
            ax=ax,
            values_format="d",
        )
        ax.set_title(key.replace("_", " | "), fontsize=10)
        path = os.path.join(out_dir, f"cm_{key}.png")
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        logger.info("Saved: %s", path)


# ── Wilcoxon statistical tests ─────────────────────────────────────────────────
def run_wilcoxon_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Wilcoxon signed-rank test over per-fold F1-macro scores.

    Mirrors Exercise-3 from course: tests H₀ that two pipelines produce
    identically distributed F1 scores across folds.  p < 0.05 → significant.

    Returns a DataFrame of all pairwise comparisons.
    """
    df = df.copy()
    df["label"] = df["model_name"] + " (" + df["strategy"] + ")"

    pivot = df.pivot_table(index="outer_fold", columns="label", values="f1_macro")
    labels = pivot.columns.tolist()

    records = []
    for a, b in combinations(labels, 2):
        scores_a = pivot[a].values
        scores_b = pivot[b].values
        try:
            stat, pval = wilcoxon(scores_a, scores_b)
        except ValueError:
            stat, pval = np.nan, np.nan
        records.append({
            "Pipeline A":   a,
            "Pipeline B":   b,
            "Wilcoxon W":   round(stat, 4) if not np.isnan(stat) else "N/A",
            "p-value":      round(pval, 6) if not np.isnan(pval) else "N/A",
            "Significant":  "Yes" if (not np.isnan(pval) and pval < ALPHA) else "No",
            "Mean F1 A":    round(pivot[a].mean(), 4),
            "Mean F1 B":    round(pivot[b].mean(), 4),
        })

    result_df = pd.DataFrame(records)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    result_df.to_csv(os.path.join(EXPERIMENT_DIR, "wilcoxon_tests.csv"), index=False)
    logger.info("Wilcoxon test results saved.")
    return result_df


def print_summary_table(agg_df: pd.DataFrame) -> None:
    logger.info("\n=== AGGREGATED RESULTS (mean ± std across %d outer folds) ===", OUTER_CV_FOLDS)
    with pd.option_context("display.max_rows", 40, "display.max_columns", 20, "display.width", 120):
        logger.info("\n%s", agg_df.to_string())


def plot_metric_heatmap(df: pd.DataFrame) -> None:
    """
    Heatmap of mean metrics per pipeline – gives an at-a-glance overview
    of the precision/recall/F1/AUC trade-offs across all configurations.
    """
    df = df.copy()
    df["label"] = df["model_name"] + " (" + df["strategy"] + ")"
    pivot = df.groupby("label")[["f1_macro", "f1_dropout", "roc_auc", "precision_macro", "recall_macro"]].mean()
    pivot = pivot.sort_values("f1_macro", ascending=False).round(4)

    fig, ax = plt.subplots(figsize=(10, max(5, len(pivot) * 0.5 + 2)))
    sns.heatmap(
        pivot, annot=True, fmt=".4f", cmap="YlOrRd",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Score"},
    )
    ax.set_title("Pipeline Metric Heatmap (mean over outer folds)", fontweight="bold")
    ax.set_ylabel("")
    ax.set_xticklabels(
        ["F1-Macro", "F1-Dropout", "ROC-AUC", "Precision-Macro", "Recall-Macro"],
        rotation=25,
    )
    fig.tight_layout()
    _savefig("experiments", "metric_heatmap.png")


def run_evaluation(df_results: pd.DataFrame, all_results: list[dict]) -> pd.DataFrame:
    """
    Orchestrate all evaluation plots and tables.

    Returns the aggregated summary DataFrame.
    """
    from student_dropout.experiments import aggregate_results
    logger.info("=== Running Evaluation ===")

    agg_df = aggregate_results(df_results)
    print_summary_table(agg_df)

    plot_model_comparison(df_results)
    plot_f1_boxplot(df_results)
    plot_roc_curves(all_results)
    plot_confusion_matrices(all_results)
    plot_metric_heatmap(df_results)

    wilcoxon_df = run_wilcoxon_tests(df_results)
    logger.info("=== Evaluation complete ===")
    return agg_df
