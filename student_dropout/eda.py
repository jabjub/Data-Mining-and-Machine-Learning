"""
Exploratory Data Analysis (project plan §3).

All plots are saved to disk under EDA_DIR so they can be embedded directly
in the defence presentation.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from student_dropout.config import EDA_DIR, NUMERICAL_FEATURES, POSITIVE_CLASS

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


def _savefig(name: str) -> None:
    os.makedirs(EDA_DIR, exist_ok=True)
    path = os.path.join(EDA_DIR, name)
    plt.savefig(path)
    plt.close()
    logger.info("Saved: %s", path)


# ── 3.1  Class distribution ────────────────────────────────────────────────────
def plot_class_distribution(y: pd.Series) -> None:
    counts = y.value_counts().sort_index()
    labels = ["Non-Dropout (0)", "Dropout (1)"]
    colors = ["#4CAF50", "#F44336"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(labels, counts.values, color=colors, edgecolor="black", width=0.5)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 30, str(v), ha="center", fontweight="bold")
    axes[0].set_title("Absolute class frequencies")
    axes[0].set_ylabel("Count")

    axes[1].pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": "white"},
    )
    axes[1].set_title("Class proportion")

    fig.suptitle("Target Class Distribution (Binary: Dropout vs Non-Dropout)", fontweight="bold")
    _savefig("01_class_distribution.png")


# ── 3.2  Correlation heatmap (numerical features only) ────────────────────────
def plot_correlation_heatmap(X: pd.DataFrame) -> None:
    num_cols = [c for c in NUMERICAL_FEATURES if c in X.columns]
    corr = X[num_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Pearson Correlation – Numerical Features", fontweight="bold", pad=12)
    _savefig("02_correlation_heatmap.png")


# ── 3.3  Histograms coloured by class ─────────────────────────────────────────
def plot_feature_histograms(X: pd.DataFrame, y: pd.Series) -> None:
    num_cols = [c for c in NUMERICAL_FEATURES if c in X.columns]
    n = len(num_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    df_plot = X[num_cols].copy()
    df_plot["_target"] = y.values

    for i, col in enumerate(num_cols):
        for cls, color, label in [(0, "#4CAF50", "Non-Dropout"), (1, "#F44336", "Dropout")]:
            subset = df_plot.loc[df_plot["_target"] == cls, col].dropna()
            axes[i].hist(subset, bins=30, alpha=0.6, color=color, label=label, density=True)
        axes[i].set_title(col, fontsize=9)
        axes[i].legend(fontsize=7)
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Class", fontweight="bold", y=1.01)
    fig.tight_layout()
    _savefig("03_feature_histograms.png")


# ── 3.4  Boxplots by class ─────────────────────────────────────────────────────
def plot_boxplots(X: pd.DataFrame, y: pd.Series) -> None:
    key_cols = [
        "Admission grade",
        "Age at enrollment",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
    ]
    key_cols = [c for c in key_cols if c in X.columns]

    df_plot = X[key_cols].copy()
    df_plot["Class"] = y.map({0: "Non-Dropout", 1: "Dropout"})

    n = len(key_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(key_cols):
        sns.boxplot(
            data=df_plot, x="Class", y=col,
            hue="Class", palette={"Non-Dropout": "#4CAF50", "Dropout": "#F44336"},
            ax=axes[i], order=["Non-Dropout", "Dropout"], width=0.4, legend=False,
        )
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Key Feature Boxplots by Class", fontweight="bold")
    fig.tight_layout()
    _savefig("04_boxplots.png")


# ── 3.5  Categorical feature vs dropout rate ──────────────────────────────────
def plot_categorical_dropout_rates(X: pd.DataFrame, y: pd.Series) -> None:
    cat_cols_plot = [
        "Scholarship holder",
        "Debtor",
        "Tuition fees up to date",
        "Gender",
        "Displaced",
        "International",
    ]
    cat_cols_plot = [c for c in cat_cols_plot if c in X.columns]

    df_plot = X[cat_cols_plot].copy()
    df_plot["target"] = y.values

    ncols = 3
    nrows = int(np.ceil(len(cat_cols_plot) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols_plot):
        rates = df_plot.groupby(col)["target"].mean() * 100
        axes[i].bar(rates.index.astype(str), rates.values, color="#5C9BD6", edgecolor="black", width=0.4)
        axes[i].set_title(f"Dropout rate by {col}", fontsize=9)
        axes[i].set_ylabel("Dropout %")
        axes[i].set_ylim(0, 100)
        for xi, v in zip(rates.index, rates.values):
            axes[i].text(str(xi), v + 1.5, f"{v:.1f}%", ha="center", fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Dropout Rate by Binary/Categorical Feature", fontweight="bold")
    fig.tight_layout()
    _savefig("05_categorical_dropout_rates.png")


# ── Orchestrator ───────────────────────────────────────────────────────────────
def run_eda(X: pd.DataFrame, y: pd.Series) -> None:
    logger.info("=== Running EDA ===")
    plot_class_distribution(y)
    plot_correlation_heatmap(X)
    plot_feature_histograms(X, y)
    plot_boxplots(X, y)
    plot_categorical_dropout_rates(X, y)
    logger.info("EDA complete. Plots saved to: %s", EDA_DIR)
