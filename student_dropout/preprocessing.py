"""
Pipeline construction (project plan §4, §5, §6, §8).

Defence-critical design decisions
──────────────────────────────────
1.  imblearn.pipeline.Pipeline is used instead of sklearn's Pipeline because
    sklearn's Pipeline does not support Resampler steps.  imblearn's version is
    a drop-in replacement for all other steps.

2.  SMOTE is placed AFTER the ColumnTransformer (encoding + scaling).  At that
    point, all features are numerical, so standard SMOTE is mathematically
    equivalent to SMOTE-NC on the pre-encoded space — no synthetic category
    interpolation problem arises because OHE has already mapped every category
    to a distinct binary dimension.

3.  Fitting the ColumnTransformer (scaler + OHE) only on training folds and
    transforming test folds with those fitted parameters is automatically
    enforced by the Pipeline API — this is the primary mechanism for preventing
    data leakage (project plan §8).

4.  SMOTE is a step inside the same pipeline, so it can only access
    training-fold rows during fit().  predict() never calls the resampler,
    so the test/validation fold is never contaminated (project plan §5).

Strategies
──────────
  'baseline'     – no imbalance handling
  'class_weight' – class_weight='balanced' on classifiers that support it
  'smote'        – SMOTE oversampling inside the pipeline
"""

import logging

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from student_dropout.config import RANDOM_STATE

logger = logging.getLogger(__name__)

STRATEGIES = ("baseline", "class_weight", "smote")

# Classifiers that accept class_weight='balanced'
_SUPPORTS_CLASS_WEIGHT = {"LogisticRegression", "RandomForest"}


def build_preprocessor(num_features: list[str], cat_features: list[str]) -> ColumnTransformer:
    """
    ColumnTransformer that applies:
      - SimpleImputer (median) + StandardScaler  → numerical / binary features
      - SimpleImputer (most_frequent) + OHE      → nominal categorical features

    Fitting happens only inside pipeline.fit(), so scaler and encoder parameters
    are derived exclusively from training data — no leakage.
    """
    numerical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipe, num_features),
            ("cat", categorical_pipe, cat_features),
        ],
        remainder="drop",
    )


def _make_classifier(model_name: str, strategy: str):
    """Instantiate classifier, injecting class_weight when strategy requires it."""
    balanced = (strategy == "class_weight")

    if model_name == "LogisticRegression":
        return LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=2000,
            class_weight="balanced" if balanced else None,
        )
    if model_name == "KNN":
        return KNeighborsClassifier()
    if model_name == "RandomForest":
        return RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced" if balanced else None,
        )
    if model_name == "AdaBoost":
        base = DecisionTreeClassifier(
            max_depth=1,
            class_weight="balanced" if balanced else None,
            random_state=RANDOM_STATE,
        )
        return AdaBoostClassifier(estimator=base, random_state=RANDOM_STATE)
    if model_name == "NaiveBayes":
        return GaussianNB()

    raise ValueError(f"Unknown model_name: {model_name}")


def build_pipeline(
    model_name: str,
    strategy: str,
    num_features: list[str],
    cat_features: list[str],
) -> Pipeline:
    """
    Build a complete imblearn Pipeline for a given model and imbalance strategy.

    The step named 'clf' is what RandomizedSearchCV targets via 'clf__<param>'.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got '{strategy}'")
    if strategy == "class_weight" and model_name not in _SUPPORTS_CLASS_WEIGHT:
        raise ValueError(
            f"'{model_name}' does not natively support class_weight='balanced'. "
            "Use 'baseline' or 'smote' instead."
        )

    preprocessor = build_preprocessor(num_features, cat_features)
    clf = _make_classifier(model_name, strategy)

    steps = [("preprocessor", preprocessor)]

    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))

    steps.append(("clf", clf))

    return Pipeline(steps)


def get_model_strategy_pairs() -> list[tuple[str, str]]:
    """
    Return all valid (model_name, strategy) combinations.

    KNN and NaiveBayes do not support class_weight, so only baseline + smote
    are tested for those models.
    """
    pairs = []
    all_models = ["LogisticRegression", "KNN", "RandomForest", "AdaBoost", "NaiveBayes"]
    for model in all_models:
        for strategy in STRATEGIES:
            if strategy == "class_weight" and model not in _SUPPORTS_CLASS_WEIGHT:
                continue
            pairs.append((model, strategy))
    return pairs


def get_feature_names_out(pipeline: Pipeline) -> list[str]:
    """
    Extract human-readable feature names from a fitted pipeline's preprocessor,
    stripping the 'num__' / 'cat__' prefixes added by ColumnTransformer.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    raw = preprocessor.get_feature_names_out()
    return [n.split("__", 1)[-1] for n in raw]
