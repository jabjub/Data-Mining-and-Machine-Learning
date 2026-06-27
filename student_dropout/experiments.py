"""
Nested Cross-Validation with RandomizedSearchCV (project plan §7).

Architecture
────────────
Outer loop  : StratifiedKFold(n=5)  – unbiased estimate of generalisation performance.
Inner loop  : RandomizedSearchCV(cv=StratifiedKFold(n=3)) – hyperparameter selection.

Why nested CV?
  A single CV loop used for both hyperparameter tuning AND performance estimation
  produces optimistically biased results because the same data informs both decisions.
  Nested CV separates these concerns: the outer fold score is a true out-of-sample
  estimate of the model *selected* by the inner search.

Why StratifiedKFold?
  Class imbalance (~32% Dropout) means random splitting could create folds with
  very different class ratios, destabilising performance estimates.  Stratified
  splitting preserves the population-level ratio in every fold.

Why RandomizedSearchCV over GridSearchCV?
  The combined search space across all models has thousands of combinations.
  Random sampling of n_iter=30 configurations (Bergstra & Bengio, 2012) achieves
  near-optimal results at a fraction of the cost, making nested CV feasible.

Result schema (one dict per outer fold × model × strategy):
  model_name, strategy, outer_fold, f1_macro, roc_auc,
  precision_macro, recall_macro, best_params
"""

import logging
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd

# Suppress harmless sklearn warnings that clutter the log
warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The total space of parameters .* is smaller than n_iter",
    category=UserWarning,
)
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from student_dropout.config import (
    EXPERIMENT_DIR,
    INNER_CV_FOLDS,
    OUTER_CV_FOLDS,
    PARAM_GRIDS,
    RANDOM_STATE,
    RANDOMIZED_SEARCH_ITER,
)
from student_dropout.preprocessing import (
    build_pipeline,
    get_model_strategy_pairs,
)

import os

logger = logging.getLogger(__name__)


def _outer_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=OUTER_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def _inner_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def _evaluate_fold(
    model_name: str,
    strategy: str,
    pipeline,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fold_idx: int,
) -> dict[str, Any]:
    """
    Run the inner RandomizedSearchCV on the training fold, then evaluate the
    best estimator on the held-out test fold.  SMOTE (when present) is applied
    only during the inner fit(), never touching the test fold.
    """
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=RANDOMIZED_SEARCH_ITER,
        cv=_inner_cv(),
        scoring="f1_macro", #TODO Revert back
        refit=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        error_score="raise",
    )

    t0 = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    try:
        y_proba = best_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except AttributeError:
        auc = np.nan

    result = {
        "model_name":       model_name,
        "strategy":         strategy,
        "outer_fold":       fold_idx,
        "f1_macro":         f1_score(y_test, y_pred, average="macro"),
        "f1_dropout":       f1_score(y_test, y_pred, pos_label=1, average="binary"),
        "roc_auc":          auc,
        "precision_macro":  precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro":     recall_score(y_test, y_pred, average="macro", zero_division=0),
        "best_params":      search.best_params_,
        "fit_time_s":       round(elapsed, 2),
        "best_model":       best_model,
        "y_test":           y_test,
        "y_pred":           y_pred,
        "y_proba":          y_proba if not np.isnan(auc) else None,
    }

    logger.debug(
        "[Fold %d] %s | %s -> F1=%.4f  AUC=%.4f  (%.1fs)",
        fold_idx, model_name, strategy, result["f1_macro"], auc, elapsed,
    )
    return result


def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    num_features: list[str],
    cat_features: list[str],
) -> list[dict[str, Any]]:
    """
    Execute the full nested CV benchmark over all (model, strategy) pairs.

    Returns
    -------
    all_results : list of dicts
        One entry per (model, strategy, outer_fold).
    """
    pairs = get_model_strategy_pairs()
    outer_cv = _outer_cv()
    # Keep X as DataFrame so ColumnTransformer can resolve string column names.
    y_arr = y.values

    all_results: list[dict] = []
    total = len(pairs) * OUTER_CV_FOLDS

    logger.info(
        "=== Starting Nested CV: %d model-strategy pairs x %d outer folds = %d total ===",
        len(pairs), OUTER_CV_FOLDS, total,
    )

    done = 0
    for model_name, strategy in pairs:
        param_grid = PARAM_GRIDS.get(model_name, {})
        fold_results = []

        logger.info(">> %s | strategy=%s", model_name, strategy)

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_arr)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]

            # Build a fresh pipeline for every fold to avoid state contamination
            pipeline = build_pipeline(model_name, strategy, num_features, cat_features)

            try:
                result = _evaluate_fold(
                    model_name, strategy, pipeline, param_grid,
                    X_train, y_train, X_test, y_test, fold_idx,
                )
                fold_results.append(result)
                all_results.append(result)
            except Exception as exc:
                logger.error(
                    "FAILED: %s | %s | fold %d - %s", model_name, strategy, fold_idx, exc
                )

            done += 1
            logger.info("  Progress: %d / %d", done, total)

        if fold_results:
            mean_f1 = np.mean([r["f1_macro"] for r in fold_results])
            std_f1  = np.std([r["f1_macro"] for r in fold_results])
            mean_auc = np.mean([r["roc_auc"] for r in fold_results if not np.isnan(r["roc_auc"])])
            logger.info(
                "  [OK] %s | %s -> F1=%.4f +/- %.4f  AUC=%.4f",
                model_name, strategy, mean_f1, std_f1, mean_auc,
            )

    logger.info("=== Nested CV complete: %d results collected ===", len(all_results))
    return all_results


def results_to_dataframe(all_results: list[dict]) -> pd.DataFrame:
    """
    Convert the raw results list into a tidy DataFrame (excluding heavy objects).
    """
    rows = []
    for r in all_results:
        rows.append({
            "model_name":       r["model_name"],
            "strategy":         r["strategy"],
            "outer_fold":       r["outer_fold"],
            "f1_macro":         r["f1_macro"],
            "f1_dropout":       r["f1_dropout"],
            "roc_auc":          r["roc_auc"],
            "precision_macro":  r["precision_macro"],
            "recall_macro":     r["recall_macro"],
            "fit_time_s":       r["fit_time_s"],
            "best_params":      str(r["best_params"]),
        })
    return pd.DataFrame(rows)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean ± std across outer folds for each (model, strategy) pair.
    This is the table that goes directly into the defence presentation.
    """
    numeric_cols = ["f1_macro", "f1_dropout", "roc_auc", "precision_macro", "recall_macro"]
    agg = (
        df.groupby(["model_name", "strategy"])[numeric_cols]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.sort_values("f1_dropout_mean", ascending=False)   # TODO REVERT BACK TO f1_macro_mean


def find_best_configuration(agg_df: pd.DataFrame) -> tuple[str, str]:
    """
    Selection rule (project plan §9):
      1. Highest mean F1-Balanced
      2. ROC-AUC as tie-breaker
    """
    top = agg_df.sort_values(
        ["f1_dropout_mean", "roc_auc_mean"], ascending=[False, False] # TODO REVERT BACK TO f1_macro_mean
    ).iloc[0]
    idx = top.name  # (model_name, strategy)
    logger.info("Best configuration: model=%s  strategy=%s", idx[0], idx[1])
    return idx[0], idx[1]


def save_results(df: pd.DataFrame, agg_df: pd.DataFrame) -> None:
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    df.to_csv(os.path.join(EXPERIMENT_DIR, "nested_cv_raw.csv"), index=False)
    agg_df.to_csv(os.path.join(EXPERIMENT_DIR, "nested_cv_aggregated.csv"))
    logger.info("Experiment results saved to %s", EXPERIMENT_DIR)
