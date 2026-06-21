import os

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Cross-validation setup (project plan §7) ───────────────────────────────────
OUTER_CV_FOLDS = 5
INNER_CV_FOLDS = 3
RANDOMIZED_SEARCH_ITER = 30

# ── Output directories ─────────────────────────────────────────────────────────
RESULTS_DIR     = "results"
EDA_DIR         = os.path.join(RESULTS_DIR, "eda")
EXPERIMENT_DIR  = os.path.join(RESULTS_DIR, "experiments")
XAI_DIR         = os.path.join(RESULTS_DIR, "xai")
LOGS_DIR        = os.path.join(RESULTS_DIR, "logs")

# ── Target (project plan §4.0) ─────────────────────────────────────────────────
TARGET_COL     = "Target"
POSITIVE_CLASS = "Dropout"

# ── 2nd-semester features to remove (project plan §4.1) ───────────────────────
SECOND_SEM_FEATURES = [
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
]

# ── Nominal categorical features → One-Hot Encoding (project plan §4.2) ───────
# These are integers in the raw data but are *codes* for unordered categories.
CATEGORICAL_FEATURES = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nationality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

# ── Binary 0/1 features – treated as numerical (fed to StandardScaler) ─────────
BINARY_FEATURES = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

# ── Continuous numerical features → StandardScaler (project plan §4.3) ────────
NUMERICAL_FEATURES = [
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

# All features that receive StandardScaler (numerical + binary)
SCALE_FEATURES = NUMERICAL_FEATURES + BINARY_FEATURES

# ── Hyperparameter search spaces (project plan §7, §6) ────────────────────────
# Keys must match the pipeline step name 'clf__<param>'
PARAM_GRIDS = {
    "LogisticRegression": {
        "clf__C":        [0.001, 0.01, 0.1, 1, 10, 100],
        "clf__solver":   ["lbfgs", "liblinear"],
        "clf__max_iter": [2000],
    },
    "KNN": {
        "clf__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "clf__weights":     ["uniform", "distance"],
        "clf__metric":      ["euclidean", "manhattan"],
    },
    "RandomForest": {
        "clf__n_estimators":    [100, 200, 300],
        "clf__max_depth":       [None, 5, 10, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf":  [1, 2, 4],
        "clf__max_features":    ["sqrt", "log2"],
    },
    "AdaBoost": {
        "clf__n_estimators": [50, 100, 200, 300],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
    },
    "NaiveBayes": {
        "clf__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
    },
}
