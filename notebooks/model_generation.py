from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.stats import randint, uniform

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier
import shap

CSV_PATH = "merged_may_june_july_2026_enriched.csv"  # Change this to your input CSV
OUTPUT_DIR = Path("fraud_model_outputs_605")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "is_fraud_event"
# Always include these 2 columns, and then add in any additional columns you want to be features/used for feature engineering
INCLUDED_COLUMNS = ["is_fraud_event", "confirmation_number", "is_member", "member_enrollment_source", "booking_timestamp", "check_in_date", "booking_channel", "stay_length", "member_tenure_days_fraud"]
RANDOM_STATE = 42

# Work on this more
RUN_HYPERPARAMETER_TUNING = False

# Potentially inaccurate here
INCLUDE_POST_BOOKING_FIELDS = True

DATE_COLUMNS = ["booking_timestamp", "check_in_date"]

# --- FIX: any text column with more distinct values than this is dropped
# (with a warning) instead of being one-hot encoded into hundreds of columns.
MAX_CATEGORIES = 50

# --- FIX: SHAP is computed on a sample, not the whole test set.
SHAP_SAMPLE_SIZE = 5000


# --- FIX: read only the columns we need instead of the whole CSV.
wanted = set(INCLUDED_COLUMNS)
df = pd.read_csv(
    CSV_PATH,
    usecols=lambda c: c.strip() in wanted,
    low_memory=False,
)

# Normalize whitespace in column names.
df.columns = df.columns.str.strip()

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found. "
        f"Available columns: {df.columns.tolist()}"
    )

if df.empty:
    raise ValueError("The input CSV contains no rows.")

# Remove rows with a missing target.
df[TARGET] = df[TARGET].astype("string").str.strip().str.lower()

target_mapping = {
    "1": 1,
    "true": 1,
    "yes": 1,
    "fraud": 1,
    "fraudulent": 1,
    "0": 0,
    "false": 0,
    "no": 0,
    "legitimate": 0,
    "non-fraud": 0,
    "non_fraud": 0,
}

df[TARGET] = df[TARGET].map(target_mapping)
df = df.dropna(subset=[TARGET]).copy()
df[TARGET] = df[TARGET].astype("int8")

if df[TARGET].nunique() != 2:
    raise ValueError(
        "The target must contain both fraud (1) and non-fraud (0) rows."
    )

class_counts = df[TARGET].value_counts().sort_index()

print("\nTarget distribution:")
print(f"Non-fraud (0): {class_counts.get(0, 0):,}")
print(f"Fraud     (1): {class_counts.get(1, 0):,}")
print(f"Fraud prevalence: {df[TARGET].mean():.4%}")

if class_counts.min() < 20:
    warnings.warn(
        "There are fewer than 20 examples in at least one target class. "
        "A reliable stratified three-way split and stable fraud metrics "
        "may not be possible."
    )

if class_counts.min() < 4:
    raise ValueError(
        "Too few examples in the minority class for a reliable "
        "stratified 70/15/15 split. Collect more labeled data or "
        "use a carefully designed alternative evaluation strategy."
    )

# Optionally exclude fields that may be known only after a booking
# Potentially inaccurate here
if not INCLUDE_POST_BOOKING_FIELDS:
    post_booking_candidates = [
        "cancellation_lead_time_days",
        "cancellation_date",
        "days_booking_to_cancellation",
        "has_cancellation_date",
        "member_tenure_days_fraud",
        "qualification_code",
    ]
    df = df.drop(
        columns=[c for c in post_booking_candidates if c in df.columns],
        errors="ignore",
    )

print(df.columns)


# ---------------------------------------------------------------------------
# FIX: turn the raw dates into a few numeric features (like the first script)
# ---------------------------------------------------------------------------
if "booking_timestamp" in df.columns:
    booking_ts = (
        pd.to_datetime(df["booking_timestamp"], errors="coerce", utc=True)
        .dt.tz_localize(None)
    )
    # df["booking_hour"] = booking_ts.dt.hour
    # df["booking_dayofweek"] = booking_ts.dt.dayofweek
else:
    booking_ts = None

if "check_in_date" in df.columns:
    check_in = pd.to_datetime(df["check_in_date"], errors="coerce")
    # df["checkin_dayofweek"] = check_in.dt.dayofweek
    if booking_ts is not None:
        df["lead_time_days"] = (check_in.dt.normalize() - booking_ts.dt.normalize()).dt.days

# Make sure a True/False column is numeric, not text.
if "is_member" in df.columns and df["is_member"].dtype == object:
    df["is_member"] = (
        df["is_member"].astype("string").str.strip().str.lower()
        .map({"true": 1, "1": 1, "yes": 1, "false": 0, "0": 0, "no": 0})
    )


# Preserve booking IDs separately for reporting.
if "confirmation_number" in df.columns:
    booking_ids = df["confirmation_number"].copy()
else:
    booking_ids = pd.Series(
        df.index, index=df.index, name="confirmation_number"
    )

# FIX: drop the raw date columns from the features.
X = df.drop(
    columns=[TARGET, "confirmation_number"] + DATE_COLUMNS,
    errors="ignore",
)
y = df[TARGET].astype(int)

for c in ["stay_length", "member_tenure_days_fraud"]:
    if c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
booking_ids = booking_ids.reset_index(drop=True)

# FIX: safety net against high-cardinality text columns.
text_columns = X.select_dtypes(exclude=["number", "bool"]).columns
too_many = [c for c in text_columns if X[c].nunique(dropna=True) > MAX_CATEGORIES]
if too_many:
    warnings.warn(
        f"Dropping text columns with more than {MAX_CATEGORIES} distinct "
        f"values (they would explode into many one-hot columns): {too_many}"
    )
    X = X.drop(columns=too_many)

# First reserve 30% for validation + test.
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Split the remaining 30% equally into validation and test.
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)

def show_split(name, labels):
    positives = int(labels.sum())
    total = len(labels)
    negatives = total - positives
    prevalence = positives / total if total else 0
    print(
        f"{name:10s}: {total:>8,} rows | "
        f"fraud={positives:>6,} | non-fraud={negatives:>7,} | "
        f"fraud rate={prevalence:.4%}"
    )

print("\nStratified split distributions:")
show_split("Train", y_train)
show_split("Validation", y_val)
show_split("Test", y_test)

for name, labels in [
    ("train", y_train),
    ("validation", y_val),
    ("test", y_test),
]:
    if labels.nunique() < 2:
        raise ValueError(
            f"The {name} split contains only one class. "
            "More labeled fraud examples are required."
        )

numeric_columns = X_train.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_columns = [
    col for col in X_train.columns
    if col not in numeric_columns
]

print("\nFeature preprocessing:")
print(f"Numeric columns ({len(numeric_columns)}): {numeric_columns}")
print(
    f"Categorical columns ({len(categorical_columns)}): "
    f"{categorical_columns}"
)

# FIX: no StandardScaler (trees don't need it). XGBoost also handles NaN
# natively, so numeric columns are passed through as-is.
numeric_pipeline = "passthrough"

# Impute missing categories and one-hot encode.
# FIX: dense output, since there are now only a handful of columns.
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=10,
                sparse_output=False,
                dtype=np.float32,
            ),
        ),
    ]
)

transformers = []
if numeric_columns:
    transformers.append(("numeric", numeric_pipeline, numeric_columns))
if categorical_columns:
    transformers.append(
        ("categorical", categorical_pipeline, categorical_columns)
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
    sparse_threshold=0.0,
)

# Fit preprocessing only on training data to prevent data leakage.
X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)
X_val_processed = preprocessor.transform(X_val).astype(np.float32)
X_test_processed = preprocessor.transform(X_test).astype(np.float32)

print("\nProcessed feature matrix shapes:")
print("Train:", X_train_processed.shape)
print("Validation:", X_val_processed.shape)
print("Test:", X_test_processed.shape)

# Give fraud cases more weight. This does not alter validation/test data.
n_negative = int((y_train == 0).sum())
n_positive = int((y_train == 1).sum())
scale_pos_weight = n_negative / max(n_positive, 1)

print(f"\nscale_pos_weight: {scale_pos_weight:.3f}")

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=220,
    learning_rate=0.04,
    max_depth=5,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=2.0,
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=75,
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

if RUN_HYPERPARAMETER_TUNING:
    print("\nRunning randomized hyperparameter search...")

    # Search uses only the training set. The validation set remains
    # available for early stopping and threshold selection.
    param_distributions = {
        "max_depth": randint(3, 8),
        "learning_rate": uniform(0.02, 0.10),
        "min_child_weight": randint(1, 10),
        "subsample": uniform(0.65, 0.35),
        "colsample_bytree": uniform(0.65, 0.35),
        "reg_lambda": uniform(0.5, 5.0),
    }

    search_model = clone(model)
    search_model.set_params(early_stopping_rounds=None)

    search = RandomizedSearchCV(
        estimator=search_model,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="average_precision",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
    )

    # The CV score is average precision (PR-oriented), not accuracy.
    search.fit(X_train_processed, y_train)
    print("Best parameters:", search.best_params_)

    model.set_params(**search.best_params_)

model.fit(
    X_train_processed,
    y_train,
    eval_set=[(X_val_processed, y_val)],
    verbose=True,
)

print("\nTraining complete.")
print("Best iteration:", getattr(model, "best_iteration", "not available"))

# Probabilities are more useful than a hard-coded 0.5 threshold
# when fraud is rare.
val_probabilities = model.predict_proba(X_val_processed)[:, 1]

precision_values, recall_values, thresholds = precision_recall_curve(
    y_val, val_probabilities
)

# precision_recall_curve returns one more precision/recall value
# than threshold values, so exclude the final point.
if len(thresholds) == 0:
    best_threshold = 0.5
else:
    p = precision_values[:-1]
    r = recall_values[:-1]
    f1_values = 2 * p * r / np.maximum(p + r, 1e-12)

    # Choose the threshold that maximizes validation F1.
    best_idx = int(np.nanargmax(f1_values))
    best_threshold = float(thresholds[best_idx])

print(f"\nValidation-selected threshold: {best_threshold:.6f}")

# Keep a standard 0.5-threshold evaluation as a reference.
# The optimized threshold is applied to the held-out test set.
test_probabilities = model.predict_proba(X_test_processed)[:, 1]

test_predictions_05 = (test_probabilities >= 0.5).astype(int)
test_predictions_tuned = (
    test_probabilities >= best_threshold
).astype(int)

def calculate_metrics(y_true, probabilities, predictions):
    """Return metrics for binary fraud classification."""
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "n_samples": int(len(y_true)),
        "fraud_count": int(np.sum(y_true)),
        "fraud_prevalence": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision_pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


results = {
    "threshold_0.5": calculate_metrics(
        y_test, test_probabilities, test_predictions_05
    ),
    "validation_tuned_threshold": calculate_metrics(
        y_test, test_probabilities, test_predictions_tuned
    ),
}

results["validation_tuned_threshold"]["decision_threshold"] = best_threshold

print("\n" + "=" * 70)
print("HELD-OUT TEST RESULTS")
print("=" * 70)

for label, metrics in results.items():
    print(f"\n--- {label} ---")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric:30s}: {value:.6f}")
        else:
            print(f"{metric:30s}: {value}")

print("\nClassification report at tuned threshold:")
print(
    classification_report(
        y_test,
        test_predictions_tuned,
        labels=[0, 1],
        target_names=["Non-fraud", "Fraud"],
        digits=4,
        zero_division=0,
    )
)

print("Confusion matrix at tuned threshold:")
print(
    confusion_matrix(
        y_test,
        test_predictions_tuned,
        labels=[0, 1],
    )
)

with open(OUTPUT_DIR / "test_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

# Retrieve the IDs corresponding to the held-out test rows.
test_booking_ids = booking_ids.iloc[X_test.index].to_numpy()

# Compare the tuned predictions against the actual labels.
predictions_df = pd.DataFrame({
    "confirmation_number": test_booking_ids,
    "actual_is_fraud_event": y_test.to_numpy(),
    "fraud_probability": test_probabilities,
    "prediction_threshold_0_5": test_predictions_05,
    "prediction_tuned_threshold": test_predictions_tuned,
})

# Add human-readable labels.
predictions_df["actual_label"] = predictions_df[
    "actual_is_fraud_event"
].map({0: "Non-fraud", 1: "Fraud"})

predictions_df["predicted_label"] = predictions_df[
    "prediction_tuned_threshold"
].map({0: "Non-fraud", 1: "Fraud"})

# Flag whether the tuned-threshold prediction was correct.
predictions_df["correct"] = (
    predictions_df["actual_is_fraud_event"]
    == predictions_df["prediction_tuned_threshold"]
)

predictions_df["result"] = np.where(
    predictions_df["correct"],
    "Correct",
    "Incorrect"
)

predictions_df.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False,
)

# Save a classification report in machine-readable form.
report_dict = classification_report(
    y_test,
    test_predictions_tuned,
    labels=[0, 1],
    target_names=["Non-fraud", "Fraud"],
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(report_dict).transpose().to_csv(
    OUTPUT_DIR / "classification_report.csv"
)

feature_names = preprocessor.get_feature_names_out()
importance_values = model.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance_values,
}).sort_values("importance", ascending=False)

importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False,
)

print("\nTop 25 features:")
print(importance_df.head(25).to_string(index=False))

top_n = min(25, len(importance_df))
plot_df = importance_df.head(top_n).sort_values("importance")

plt.figure(figsize=(10, max(6, top_n * 0.28)))
plt.barh(plot_df["feature"], plot_df["importance"])
plt.xlabel("XGBoost feature importance")
plt.ylabel("Feature")
plt.title("Top Fraud Prediction Features")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
plt.close()

# FIX: explain a random sample of test rows, not all of them.
rng = np.random.default_rng(RANDOM_STATE)
shap_sample_size = min(SHAP_SAMPLE_SIZE, X_test_processed.shape[0])
shap_indices = rng.choice(
    X_test_processed.shape[0],
    size=shap_sample_size,
    replace=False,
)

X_shap = X_test_processed[shap_indices]

if sparse.issparse(X_shap):
    X_shap = X_shap.toarray()

# Use the trained XGBoost model.
explainer = shap.TreeExplainer(model)

# Calculate SHAP values.
shap_values = explainer(X_shap)

print("SHAP calculation complete.")
print("Explained rows:", shap_sample_size)
print("Number of features:", len(feature_names))
print("SHAP values shape:", shap_values.values.shape)

# Beeswarm plot: importance, direction, and distribution.
plt.figure()
shap.summary_plot(
    shap_values.values,
    X_shap,
    feature_names=feature_names,
    max_display=25,
    show=False,
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shap_summary.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

# Bar plot: mean absolute SHAP value for each feature.
plt.figure()
shap.summary_plot(
    shap_values.values,
    X_shap,
    feature_names=feature_names,
    plot_type="bar",
    max_display=25,
    show=False,
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shap_feature_importance.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

print("Saved SHAP summary plots.")

shap_importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_absolute_shap": np.abs(
        shap_values.values
    ).mean(axis=0),
})

shap_importance_df = shap_importance_df.sort_values(
    "mean_absolute_shap",
    ascending=False,
)

shap_importance_df.to_csv(
    OUTPUT_DIR / "shap_feature_importance.csv",
    index=False,
)

print("\nTop 25 SHAP features:")
print(
    shap_importance_df.head(25).to_string(index=False)
)

shap_contributions_df = pd.DataFrame(
    shap_values.values,
    columns=feature_names,
)

shap_contributions_df.insert(
    0,
    "confirmation_number",
    booking_ids.iloc[
        X_test.index.to_numpy()[shap_indices]
    ].to_numpy(),
)

shap_contributions_df.to_csv(
    OUTPUT_DIR / "shap_values.csv",
    index=False,
)

print("Saved: shap_values.csv")


fpr, tpr, _ = roc_curve(y_test, test_probabilities)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_test, test_probabilities):.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Test Set")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150)
plt.close()

pr_precision, pr_recall, _ = precision_recall_curve(
    y_test, test_probabilities
)
plt.figure(figsize=(7, 5))
plt.plot(
    pr_recall,
    pr_precision,
    label=(
        f"Average precision = "
        f"{average_precision_score(y_test, test_probabilities):.4f}"
    ),
)
plt.axhline(
    y=float(y_test.mean()),
    linestyle="--",
    color="gray",
    label="Fraud prevalence baseline",
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — Test Set")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "precision_recall_curve.png", dpi=150)
plt.close()

joblib.dump(
    preprocessor,
    OUTPUT_DIR / "fraud_preprocessor.joblib",
)

joblib.dump(
    model,
    OUTPUT_DIR / "fraud_xgboost_model.joblib",
)

with open(OUTPUT_DIR / "model_metadata.json", "w") as f:
    json.dump(
        {
            "target": TARGET,
            "included_columns": INCLUDED_COLUMNS,
            "dropped_raw_date_columns": DATE_COLUMNS,
            "random_state": RANDOM_STATE,
            "split": {
                "train": 0.70,
                "validation": 0.15,
                "test": 0.15,
            },
            "best_iteration": int(
                getattr(model, "best_iteration", -1)
            ),
            "decision_threshold": best_threshold,
            "scale_pos_weight": float(scale_pos_weight),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "feature_names": [str(n) for n in feature_names],
        },
        f,
        indent=4,
    )

print(f"\nAll output files saved to: {OUTPUT_DIR.resolve()}")
print("Fraud detection pipeline finished successfully.")