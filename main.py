"""
Entry point for the Student Dropout Prediction experiment framework.

Execution order
───────────────
1.  Load & prepare data          → data_loader.py
2.  Exploratory Data Analysis    → eda.py
3.  Nested CV benchmark          → experiments.py
4.  Evaluation & plots           → evaluation.py
5.  XAI on best pipeline         → explainability.py

Usage
─────
    python main.py                      # fetches dataset via ucimlrepo
    python main.py --csv data/data.csv  # uses local CSV (sep=';')
    python main.py --skip-eda           # skip EDA (faster re-runs)
    python main.py --skip-cv            # skip CV, load cached results
"""

import argparse
import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

# ── Logging ────────────────────────────────────────────────────────────────────
os.makedirs("results/logs", exist_ok=True)
_fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.stream.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format=_fmt,
    handlers=[
        _stream_handler,
        logging.FileHandler("results/logs/experiment.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Student Dropout Prediction Pipeline")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to local dataset CSV (semicolon-separated).")
    parser.add_argument("--skip-eda", action="store_true",
                        help="Skip EDA plots.")
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip nested CV; load cached results from results/experiments/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()

    logger.info("=" * 70)
    logger.info("  STUDENT DROPOUT PREDICTION - DMML Project Framework")
    logger.info("=" * 70)

    # ── 1. Data loading & preparation ─────────────────────────────────────────
    from student_dropout.data_loader import prepare_data
    X, y, num_features, cat_features, all_features = prepare_data(local_csv=args.csv)
    y_arr = y.values

    logger.info(
        "Dataset: %d samples | %d features | Dropout rate: %.2f%%",
        len(X), X.shape[1], 100 * y.mean(),
    )

    # ── 2. EDA ────────────────────────────────────────────────────────────────
    if not args.skip_eda:
        from student_dropout.eda import run_eda
        run_eda(X, y)
    else:
        logger.info("EDA skipped (--skip-eda).")

    # ── 3. Nested CV ──────────────────────────────────────────────────────────
    results_csv = os.path.join("results", "experiments", "nested_cv_raw.csv")
    cache_pkl   = os.path.join("results", "experiments", "all_results.pkl")

    if args.skip_cv and os.path.exists(cache_pkl):
        logger.info("Loading cached CV results from %s", cache_pkl)
        with open(cache_pkl, "rb") as f:
            all_results = pickle.load(f)
        df_results = pd.read_csv(results_csv)
    else:
        from student_dropout.experiments import (
            run_nested_cv, results_to_dataframe, aggregate_results, save_results
        )
        all_results = run_nested_cv(X, y, num_features, cat_features)

        df_results = results_to_dataframe(all_results)
        agg_df = aggregate_results(df_results)
        save_results(df_results, agg_df)

        # Cache raw results (with model objects) for re-use
        os.makedirs("results/experiments", exist_ok=True)
        with open(cache_pkl, "wb") as f:
            pickle.dump(all_results, f)
        logger.info("Results cached to %s", cache_pkl)

    # ── 4. Evaluation ─────────────────────────────────────────────────────────
    from student_dropout.evaluation import run_evaluation
    agg_df = run_evaluation(df_results, all_results)

    # ── 5. Identify best configuration ────────────────────────────────────────
    from student_dropout.experiments import find_best_configuration
    best_model_name, best_strategy = find_best_configuration(agg_df)

    logger.info(
        "\n>>> BEST PIPELINE: model=%s  strategy=%s",
        best_model_name, best_strategy,
    )

    # ── 6. Retrain best pipeline on full dataset ───────────────────────────────
    logger.info("Retraining best pipeline on full dataset for XAI...")
    from student_dropout.preprocessing import build_pipeline
    from student_dropout.config import PARAM_GRIDS, RANDOM_STATE
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    best_pipeline = build_pipeline(best_model_name, best_strategy, num_features, cat_features)

    # Find the most common best_params from the nested CV outer folds
    best_fold_results = [
        r for r in all_results
        if r["model_name"] == best_model_name and r["strategy"] == best_strategy
    ]
    # Use the fold with the highest F1 to get representative params
    best_fold = max(best_fold_results, key=lambda r: r["f1_macro"])
    best_params = best_fold["best_params"]
    logger.info("Using best-fold params for XAI refit: %s", best_params)

    try:
        best_pipeline.set_params(**best_params)
    except Exception as exc:
        logger.warning("Could not set best_params on pipeline: %s. Fitting with defaults.", exc)

    best_pipeline.fit(X, y_arr)

    # ── 7. XAI ────────────────────────────────────────────────────────────────
    from student_dropout.explainability import run_explainability
    run_explainability(best_pipeline, X, y_arr, best_model_name)

    # ── 8. Also run XAI for RandomForest if it wasn't the best (for feature importance) ──
    if best_model_name != "RandomForest":
        logger.info("Running feature-importance-only XAI for RandomForest...")
        rf_results = [
            r for r in all_results
            if r["model_name"] == "RandomForest"
        ]
        if rf_results:
            from student_dropout.explainability import plot_feature_importance, _transform_X
            from student_dropout.preprocessing import get_feature_names_out

            rf_pipeline = build_pipeline("RandomForest", "smote", num_features, cat_features)
            rf_best_fold = max(rf_results, key=lambda r: r["f1_macro"])
            try:
                rf_pipeline.set_params(**rf_best_fold["best_params"])
            except Exception:
                pass
            rf_pipeline.fit(X, y_arr)
            rf_feat_names = get_feature_names_out(rf_pipeline)
            rf_X_transformed = _transform_X(rf_pipeline, X)
            plot_feature_importance(rf_pipeline, rf_feat_names, "RandomForest",
                                    rf_X_transformed, y_arr)

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    logger.info("=" * 70)
    logger.info("  EXPERIMENT COMPLETE  (total runtime: %.1f min)", elapsed / 60)
    logger.info("  Best model  : %s", best_model_name)
    logger.info("  Strategy    : %s", best_strategy)
    logger.info("  Mean F1-Mac : %.4f", agg_df.loc[(best_model_name, best_strategy), "f1_macro_mean"])
    logger.info("  Mean AUC    : %.4f", agg_df.loc[(best_model_name, best_strategy), "roc_auc_mean"])
    logger.info("  Results dir : results/")
    logger.info("=" * 70)

    # Print defence-ready summary table to stdout
    print("\n" + "=" * 70)
    print("  AGGREGATED RESULTS (for defence)")
    print("=" * 70)
    print(agg_df[["f1_macro_mean", "f1_macro_std", "roc_auc_mean", "f1_dropout_mean"]].to_string())
    print("=" * 70)


if __name__ == "__main__":
    main()
