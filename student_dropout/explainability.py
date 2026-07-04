"""
Explainable AI module - ADMML5 recommended XAI workflow:

  1. Feature importance (MDI + Permutation)  - global quick view
  2. SHAP beeswarm                           - global: direction + feature values
  3. SHAP bar                                - global: magnitude ranking
  4. SHAP waterfall                          - local: one high-risk student

SHAP explainer selection (model_output='probability'):
  RandomForest / AdaBoost  -> TreeExplainer (interventional perturbation)
  LogisticRegression       -> LinearExplainer
  KNN / NaiveBayes         -> KernelExplainer (wrapped into Explanation object)
"""

import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance as sk_perm_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from student_dropout.config import XAI_DIR

logger = logging.getLogger(__name__)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


def _savefig(name: str) -> None:
    os.makedirs(XAI_DIR, exist_ok=True)
    path = os.path.join(XAI_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)


def _transform_X(pipeline, X) -> np.ndarray:
    """Apply all pipeline steps except classifier and SMOTE (fit-only)."""
    steps = list(pipeline.named_steps.items())
    clf_step_name = steps[-1][0]
    from imblearn.pipeline import Pipeline as ImbPipeline
    pre_steps = [
        (name, step) for name, step in steps
        if name not in (clf_step_name, "smote")
    ]
    return ImbPipeline(pre_steps).transform(X)


def _safe_feature_names(feature_names, n_features):
    """Return feature names trimmed/padded to exactly n_features."""
    if len(feature_names) == n_features:
        return list(feature_names)
    return [feature_names[i] if i < len(feature_names) else f"feat_{i}"
            for i in range(n_features)]


def _attach_names(sv, feature_names):
    """Force-set feature names on SHAP Explanation."""
    if sv is None:
        return sv
    n = sv.values.shape[1] if sv.values.ndim >= 2 else sv.values.shape[0]
    # Always overwrite — SHAP may have set generic placeholders already
    sv.feature_names = _safe_feature_names(feature_names, n)
    return sv


# ── 1. Feature importance: MDI + Permutation ─────────────────────────────────

def plot_feature_importance(
    pipeline,
    feature_names: list,
    model_name: str,
    X_transformed: np.ndarray = None,
    y: np.ndarray = None,
) -> None:
    """
    MDI importance for tree models (fast, built-in).
    Permutation importance for all models (slower, model-agnostic, more reliable).
    Both shown side-by-side per ADMML5 slide 6.
    """
    clf = pipeline.named_steps["clf"]


    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        if len(importances) == len(feature_names):
            idx = np.argsort(importances)[::-1]
            top_n = min(15, len(importances))
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(
                [feature_names[i] for i in idx[:top_n]][::-1],
                importances[idx[:top_n]][::-1],
                color="steelblue", edgecolor="black",
            )
            ax.set_xlabel("Mean Decrease in Impurity", fontsize=11)
            ax.set_title(f"Feature Importance (MDI) - {model_name}", fontweight="bold")
            fig.tight_layout()
            _savefig(f"feature_importance_mdi_{model_name}.png")

    if X_transformed is not None and y is not None:
        logger.info("Computing permutation importance for %s...", model_name)
        try:
            perm = sk_perm_importance(
                clf, X_transformed, y,
                n_repeats=10, random_state=42, scoring="f1_macro", n_jobs=-1,
            )

            idx = np.argsort(perm.importances_mean)[::-1]
            top_n = min(15, len(idx))
            names = _safe_feature_names(feature_names, len(idx))
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(
                [names[i] for i in idx[:top_n]] [::-1],
                perm.importances_mean[idx[:top_n]] [::-1],
                xerr=perm.importances_std[idx[:top_n]][::-1],
                color="darkorange", edgecolor="black",
            )
            # ax.set_xlim(0, 0.3)
            ax.set_xticks(np.arange(0, 0.20, 0.05))
            ax.axvline(0, color="red", linestyle="--", alpha=0.5, label="No effect baseline")
            ax.set_xlabel("Mean F1-macro drop when feature is shuffled", fontsize=11)
            ax.set_title(f"Permutation Importance - {model_name}", fontweight="bold")
            ax.legend(fontsize=9)
            fig.tight_layout()
            _savefig(f"feature_importance_permutation_{model_name}.png")
        except Exception as exc:
            logger.warning("Permutation importance failed for %s: %s", model_name, exc)


# ── 2-5. SHAP (new Explanation-object API, model_output='probability') ─────────

def _build_shap_explanation(clf, X_transformed: np.ndarray, model_name: str):
    """
    Build a SHAP Explanation object using the new SHAP API.

    model_output='probability' is set per ADMML5 slide 12 recommendation:
    SHAP values stay on the probability scale, not log-odds, so a value of
    +0.10 means the feature raises Dropout probability by ~10 percentage points.
    """
    import shap

    n = len(X_transformed)
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(n, size=min(100, n), replace=False)
    bg = X_transformed[bg_idx]
    n_explain = min(25, n)
    X_explain = X_transformed[:n_explain]

    try:
        if model_name in ("RandomForest", "AdaBoost"):
            explainer = shap.TreeExplainer(
                clf, data=bg,
                feature_perturbation="interventional",
                model_output="probability",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sv = explainer(X_explain, check_additivity=False)
            # sv.values shape: (n, p, 2) for binary -> slice Dropout class
            return sv[:, :, 1]

        if model_name == "LogisticRegression":
            explainer = shap.LinearExplainer(clf, bg)
            sv = explainer(X_explain)
            if sv.values.ndim == 3:
                return sv[:, :, 1] # Return the XAI results for the Drop class only
            return sv

        # KNN / NaiveBayes: model-agnostic KernelExplainer (slow)
        logger.info("KernelExplainer for %s (bg=50, explain=200).", model_name)
        explain_idx = rng.choice(n, size=min(200, n), replace=False)
        X_explain_k = X_transformed[explain_idx]
        bg_k = X_transformed[rng.choice(n, size=min(50, n), replace=False)]
        explainer = shap.KernelExplainer(clf.predict_proba, bg_k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_vals = explainer.shap_values(X_explain_k)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        elif shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]
        base_val = float(clf.predict_proba(bg_k)[:, 1].mean())
        return shap.Explanation(
            values=shap_vals,
            base_values=np.full(len(shap_vals), base_val),
            data=X_explain_k,
        )

    except Exception as exc:
        logger.error("SHAP build failed for %s: %s", model_name, exc, exc_info=True)
        return None


def plot_shap_beeswarm(sv_class, feature_names: list, model_name: str) -> None:
    """Global SHAP beeswarm: importance, direction, and feature-value colour."""
    import shap
    if sv_class is None:
        return
    _attach_names(sv_class, feature_names)
    shap.plots.beeswarm(sv_class, max_display=20, show=False)
    plt.title(
        f"SHAP Beeswarm - {model_name} | Dropout (probability scale)",
        fontweight="bold",
    )
    _savefig(f"shap_beeswarm_{model_name}.png")


def plot_shap_bar(sv_class, feature_names: list, model_name: str,count_y=20) -> None:
    mean_abs = np.abs(sv_class.values).mean(axis=0)

    idx = np.argsort(mean_abs)[::-1][:count_y]

    fig, ax = plt.subplots(figsize=(15, 12))

    ax.barh(
        [feature_names[i] for i in idx][::-1],
        mean_abs[idx][::-1],
        color="deeppink"
    )

    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title(f"SHAP Mean |SHAP| - {model_name}")

    plt.tight_layout()
    _savefig(f"shap_bar_{model_name}.png")


def plot_shap_waterfall(sv_class, feature_names: list, model_name: str) -> int:
    """
    SHAP waterfall: local explanation for the highest-risk student.

    why did the model give THIS student a high dropout probability?
    The student with the highest predicted probability is chosen as the most
    actionable case for early intervention.
    """
    import shap
    if sv_class is None:
        return 0
    _attach_names(sv_class, feature_names)

    # Predicted probability = base value + sum of SHAP contributions
    predicted_prob = sv_class.base_values + sv_class.values.sum(axis=1)
    high_risk_idx = int(np.argmax(predicted_prob))

    shap.plots.waterfall(sv_class[high_risk_idx], max_display=15, show=False)
    plt.title(
        f"SHAP Waterfall - {model_name} | Student #{high_risk_idx} "
        f"(P_dropout={predicted_prob[high_risk_idx]:.3f})",
        fontweight="bold",
    )
    _savefig(f"shap_waterfall_{model_name}.png")
    logger.info(
        "Waterfall: student #%d | P(Dropout)=%.3f",
        high_risk_idx, predicted_prob[high_risk_idx],
    )
    return high_risk_idx

# ── Orchestrator ───────────────────────────────────────────────────────────────

def run_explainability(pipeline, X, y: np.ndarray, model_name: str) -> None:
    """
    Full XAI pipeline following ADMML5 recommended workflow.

    Parameters
    ----------
    pipeline   : fitted imblearn Pipeline
    X          : full feature DataFrame (as returned by prepare_data)
    y          : full binary target array
    model_name : classifier name string
    """
    logger.info("=== Explainability: %s ===", model_name)

    from student_dropout.preprocessing import get_feature_names_out
    feature_names = get_feature_names_out(pipeline)

    X_transformed = _transform_X(pipeline, X)

    clf = pipeline.named_steps["clf"]

    # Step 1: MDI + Permutation importance
    plot_feature_importance(pipeline, feature_names, model_name, X_transformed, y)

    # Steps 2-5: SHAP
    sv_class = None
    high_risk_idx = 0
    try:
        sv_class = _build_shap_explanation(clf, X_transformed, model_name)
        if sv_class is not None:
            _attach_names(sv_class, feature_names)
            plot_shap_beeswarm(sv_class, feature_names, model_name)
            plot_shap_bar(sv_class, feature_names, model_name,25)
            high_risk_idx = plot_shap_waterfall(sv_class, feature_names, model_name)
    except Exception as exc:
        logger.error("SHAP failed for %s: %s", model_name, exc, exc_info=True)

    logger.info("=== Explainability complete ===")
