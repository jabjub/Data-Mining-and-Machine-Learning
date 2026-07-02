"""
Data loading and preparation for the UCI Student Dropout dataset (ID 697).

Binary reformulation (project plan §1, §4.0):
    Dropout  → 1  (positive class – the intervention target)
    Graduate + Enrolled → 0  (Non-Dropout)

Second-semester features are removed immediately after loading (project plan §4.1)
to ensure the model is usable as an early-intervention tool after the first semester.
"""

import logging
import os

import numpy as np
import pandas as pd

from student_dropout.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    SCALE_FEATURES,
    SECOND_SEM_FEATURES,
    TARGET_COL, DROPOUT_CLASS, INCLUDE_SECOND_SEM_FEATURES,
)

logger = logging.getLogger(__name__)


def load_raw_dataframe(local_csv: str | None = None) -> pd.DataFrame:
    """
    Load the raw dataset, preferring ucimlrepo; falls back to a local CSV.
    The local CSV is expected to use ';' as separator (UCI standard export).
    """
    if local_csv and os.path.exists(local_csv):
        logger.info("Loading dataset from local file: %s", local_csv)
        df = pd.read_csv(local_csv, sep=";")
        df.columns = df.columns.str.strip()
        return df

    try:
        from ucimlrepo import fetch_ucirepo
        logger.info("Fetching dataset from UCI ML Repository (ID 697)...")
        repo = fetch_ucirepo(id=697)
        df = pd.concat([repo.data.features, repo.data.targets], axis=1)
        df.columns = df.columns.str.strip()
        logger.info("Dataset loaded: %d rows × %d cols", *df.shape)
        return df
    except Exception as exc:
        raise RuntimeError(
            "Could not fetch the dataset from ucimlrepo and no local_csv was supplied. "
            "Either `pip install ucimlrepo` or download the dataset from "
            "https://archive.ics.uci.edu/dataset/697 and pass local_csv='path/to/file.csv'."
        ) from exc


def _binarise_target(series: pd.Series) -> pd.Series:
    """
    Convert three-class target to binary.

    Justification (project plan §1): the goal is early dropout identification,
    not differentiation between enrolled and graduated students.
    """
    return series.apply(lambda v: 1 if v == DROPOUT_CLASS else 0)


def _validate_columns(df: pd.DataFrame) -> None:
    expected = set(SCALE_FEATURES + CATEGORICAL_FEATURES + SECOND_SEM_FEATURES + [TARGET_COL])
    missing = expected - set(df.columns)
    if missing:
        # Strip any residual whitespace from column names and retry
        df.columns = df.columns.str.strip()
        missing = expected - set(df.columns)
    if missing:
        logger.warning(
            "The following expected columns were not found and will be skipped: %s", missing
        )


def prepare_data(
    local_csv: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], list[str]]:
    """
    Full preparation pipeline executed *once* before any CV fold is created.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with 2nd-semester columns removed.
    y : pd.Series
        Binary target (1 = Dropout, 0 = Non-Dropout).
    num_features : list[str]
        Numerical + binary feature names present in X.
    cat_features : list[str]
        Categorical feature names present in X.
    all_features : list[str]
        Ordered list [num_features + cat_features] – column order in X.
    """
    df = load_raw_dataframe(local_csv)
    df.columns = df.columns.str.strip()

    _validate_columns(df)

    # ── 4.0 Binary target ────────────────────────────────────────────────────
    y = _binarise_target(df[TARGET_COL])

    # ── 4.1 Drop 2nd-semester features ───────────────────────────────────────
    cols_to_drop = [] if INCLUDE_SECOND_SEM_FEATURES   else [c for c in SECOND_SEM_FEATURES if c in df.columns]
    df = df.drop(columns=cols_to_drop + [TARGET_COL])

    # ── Verify feature presence ───────────────────────────────────────────────
    num_features = [f for f in SCALE_FEATURES if f in df.columns]
    cat_features = [f for f in CATEGORICAL_FEATURES if f in df.columns]

    missing_num = set(SCALE_FEATURES) - set(num_features)
    missing_cat = set(CATEGORICAL_FEATURES) - set(cat_features)
    if missing_num:
        logger.warning("Missing numerical/binary features: %s", missing_num)
    if missing_cat:
        logger.warning("Missing categorical features: %s", missing_cat)

    # ── 4.2 Cast categoricals to str so OHE treats them correctly ────────────
    for col in cat_features:
        df[col] = df[col].astype(str)

    # ── 4.4 Check for missing values ─────────────────────────────────────────
    n_missing = df[num_features + cat_features].isnull().sum().sum()
    if n_missing > 0:
        logger.warning("%d missing values detected – SimpleImputer is included in pipeline.", n_missing)
    else:
        logger.info("No missing values detected in the feature matrix.")

    X = df[num_features + cat_features].copy()

    logger.info(
        "Data ready: %d samples | %d numerical | %d categorical | "
        "class distribution: Dropout=%.1f%% Non-Dropout=%.1f%%",
        len(X),
        len(num_features),
        len(cat_features),
        100 * y.mean(),
        100 * (1 - y.mean()),
    )

    return X, y, num_features, cat_features, num_features + cat_features
