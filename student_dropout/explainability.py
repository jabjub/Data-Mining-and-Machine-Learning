"""
Explainable AI module - ADMML5 recommended XAI workflow:

  1. Feature importance (MDI + Permutation)  - global quick view
  2. SHAP beeswarm                           - global: direction + feature values
  3. SHAP bar                                - global: magnitude ranking
  4. SHAP waterfall                          - local: one high-risk student
  5. SHAP scatter/dependence                 - top-3 feature effects
  6. LIME                                    - local cross-check of waterfall

SHAP explainer selection (model_output='probability' per ADMML5 guidance):
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
    """Set feature names on SHAP Explanation if absent."""
    if sv is None:
        return sv
    n = sv.values.shape[1] if sv.values.ndim >= 2 else sv.values.shape[0]
    if not getattr(sv, "feature_names", None):
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
            top_n = min(25, len(importances))
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
            top_n = min(25, len(idx))
            names = _safe_feature_names(feature_names, len(idx))
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(
                [names[i] for i in idx[:top_n]][::-1],
                perm.importances_mean[idx[:top_n]][::-1],
                xerr=perm.importances_std[idx[:top_n]][::-1],
                color="darkorange", edgecolor="black",
            )
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
    n_explain = min(500, n)
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
                return sv[:, :, 1]
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


def plot_shap_bar(sv_class, feature_names: list, model_name: str) -> None:
    """Global SHAP bar: mean |SHAP| magnitude ranking."""
    import shap
    if sv_class is None:
        return
    _attach_names(sv_class, feature_names)
    shap.plots.bar(sv_class, max_display=20, show=False)
    plt.title(f"SHAP Mean |SHAP| (Bar) - {model_name}", fontweight="bold")
    _savefig(f"shap_bar_{model_name}.png")


def plot_shap_waterfall(sv_class, feature_names: list, model_name: str) -> int:
    """
    SHAP waterfall: local explanation for the highest-risk student.

    From ADMML5 recommended workflow: 'SHAP waterfall for one prediction'.
    Answers: why did the model give THIS student a high dropout probability?
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


def plot_shap_dependence(sv_class, feature_names: list, model_name: str, top_n: int = 3) -> None:
    """SHAP scatter plot for top-N most impactful features."""
    import shap
    if sv_class is None:
        return
    _attach_names(sv_class, feature_names)

    mean_abs = np.abs(sv_class.values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:top_n]

    for rank, feat_idx in enumerate(top_indices):
        n = sv_class.values.shape[1]
        feat_name = (feature_names[feat_idx] if feat_idx < len(feature_names)
                     else f"feat_{feat_idx}")
        safe_name = feat_name.replace("/", "_").replace(" ", "_")[:40]
        try:
            shap.plots.scatter(sv_class[:, feat_idx], color=sv_class, show=False)
            plt.title(f"SHAP Scatter - {feat_name} | {model_name}", fontweight="bold")
            _savefig(f"shap_dependence_{model_name}_{rank+1}_{safe_name}.png")
        except Exception:
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                shap.dependence_plot(
                    feat_idx, sv_class.values, sv_class.data,
                    feature_names=_safe_feature_names(feature_names, n),
                    ax=ax, show=False, alpha=0.5,
                )
                ax.set_title(f"SHAP Dependence - {feat_name} | {model_name}", fontweight="bold")
                _savefig(f"shap_dependence_{model_name}_{rank+1}_{safe_name}.png")
            except Exception as exc2:
                logger.warning("SHAP dependence failed for feat %d: %s", feat_idx, exc2)


# ── 6. LIME (manual Ridge surrogate, ADMML5 lecture implementation) ───────────

def run_lime_local(
    clf,
    X_transformed: np.ndarray,
    feature_names: list,
    model_name: str,
    sample_idx: int = None,
) -> None:
    """
    LIME local explanation — manual implementation matching ADMML5 slides 23-25.

    Steps (identical to the course code):
    1. Pick highest-risk student (or given index)
    2. Generate 2500 Gaussian neighbours (scale=0.25 * feature_std)
    3. Clip to observed feature range
    4. Query black-box: clf.predict_proba -> P(Dropout)
    5. Weight by Gaussian kernel (kernel_width = 0.75 * sqrt(p))
    6. Fit weighted Ridge surrogate in standardised space
    7. Plot local coefficients as bar chart

    Using the same student as the SHAP waterfall allows direct comparison:
    'Does LIME agree with SHAP on which features drive this student's risk?'
    """
    if not hasattr(clf, "predict_proba"):
        logger.info("LIME skipped for %s (no predict_proba).", model_name)
        return

    logger.info("Running LIME for %s...", model_name)
    try:
        proba = clf.predict_proba(X_transformed)[:, 1]
        if sample_idx is None:
            sample_idx = int(np.argmax(proba))
        x0 = X_transformed[sample_idx]
        logger.info(
            "LIME: student #%d | P(Dropout)=%.3f", sample_idx, proba[sample_idx]
        )

        rng = np.random.default_rng(42)
        feature_std = X_transformed.std(axis=0).clip(min=1e-8)
        n_perturb = 2500
        Z = rng.normal(
            loc=x0, scale=0.25 * feature_std,
            size=(n_perturb, X_transformed.shape[1]),
        )
        Z = Z.clip(X_transformed.min(axis=0), X_transformed.max(axis=0))

        black_box_output = clf.predict_proba(Z)[:, 1]

        feature_mean = X_transformed.mean(axis=0)
        Z_scaled = (Z - feature_mean) / feature_std
        x0_scaled = (x0 - feature_mean) / feature_std
        distances = np.sqrt(((Z_scaled - x0_scaled) ** 2).sum(axis=1))
        kernel_width = 0.75 * np.sqrt(X_transformed.shape[1])
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2) / 2)

        local_model = Ridge(alpha=0.1)
        local_model.fit(Z_scaled, black_box_output, sample_weight=weights)
        r2 = r2_score(
            black_box_output,
            local_model.predict(Z_scaled),
            sample_weight=weights,
        )
        logger.info("LIME surrogate weighted-R2=%.3f", r2)

        n_coef = min(len(local_model.coef_), len(feature_names))
        lime_coefs = pd.Series(
            local_model.coef_[:n_coef], index=feature_names[:n_coef]
        )
        lime_top = lime_coefs.reindex(lime_coefs.abs().nlargest(15).index)
        colors = ["tomato" if c > 0 else "steelblue" for c in lime_top]

        fig, ax = plt.subplots(figsize=(10, 7))
        lime_top.plot(kind="barh", ax=ax, color=colors, edgecolor="black")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(
            f"Local coefficient (positive = raises P(Dropout) near student #{sample_idx})",
            fontsize=9,
        )
        ax.set_title(
            f"LIME Local Explanation - {model_name} | Student #{sample_idx}\n"
            f"P(Dropout)={proba[sample_idx]:.3f}  |  Surrogate weighted-R2={r2:.3f}",
            fontweight="bold",
        )
        fig.tight_layout()
        _savefig(f"lime_local_{model_name}.png")

        top5 = lime_coefs.abs().nlargest(5).index.tolist()
        logger.info("LIME top-5 factors for student #%d: %s", sample_idx, top5)

    except Exception as exc:
        logger.error("LIME failed for %s: %s", model_name, exc, exc_info=True)


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
            plot_shap_bar(sv_class, feature_names, model_name)
            high_risk_idx = plot_shap_waterfall(sv_class, feature_names, model_name)
            plot_shap_dependence(sv_class, feature_names, model_name, top_n=3)
    except Exception as exc:
        logger.error("SHAP failed for %s: %s", model_name, exc, exc_info=True)

    # Step 6: LIME — same student as waterfall for direct SHAP vs LIME comparison
    run_lime_local(clf, X_transformed, feature_names, model_name, sample_idx=high_risk_idx)

    logger.info("=== Explainability complete ===")
