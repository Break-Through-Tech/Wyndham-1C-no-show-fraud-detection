from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
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
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")

FOLDER = "other/content/"
TRAIN_FILE = FOLDER + "bookings_train_may_june.parquet"
TEST_FILE = FOLDER + "bookings_test_july.parquet"

OUTPUT_DIR = Path("fraud_model_outputs_xg")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "is_fraud"
RANDOM_STATE = 42

COLS = [
    "confirmation_number", "member_number", "booking_timestamp",
    "check_in_date", "cancellation_date", "rate_code",
    "room_revenue", "points_earned", "booking_channel",
    "qualification_code", "member_enrollment_date",
    "member_enrollment_source", "first_name", "last_name",
    "email", "phone_number", "data_month", "is_fraud", "check_out_date"
]

# Can comment out any features not desired, or add in ones you do want
# Note that some might not be possible to use for the stretch goal

FEATURES = [
    "lead_time_days",
    "account_age_days",
    "stay_length_days",
    "is_digital",
    # "is_sdc4",
    # "is_srb",
    # "is_gmail_or_outlook",
    # "room_revenue",
    # "points_earned",
    # "booking_channel",
    # "member_enrollment_source",
    "account_age_at_first_booking",
    "enrolled_within_30d",
    "enrolled_online",
    # "is_late_cancel",
    # "cancel_same_day_as_booking",
    # "email_initial_lastname_format",
    # "members_sharing_phone_prefix",
    # "m_mtd_noshows"
]

all_rows = pd.concat([
    pd.read_parquet(TRAIN_FILE, columns=COLS),
    pd.read_parquet(TEST_FILE, columns=COLS)
], ignore_index=True)

all_rows[TARGET] = pd.to_numeric(all_rows[TARGET], errors="coerce")
all_rows = all_rows.dropna(subset=[TARGET]).copy()
all_rows[TARGET] = all_rows[TARGET].astype("int8")

if all_rows[TARGET].nunique() != 2:
    raise ValueError("Target must contain both fraud (1) and normal (0) rows.")

print("Loaded data:", all_rows.shape)
print("\nFraud distribution:")
print(all_rows.groupby("data_month")[TARGET].agg(["count", "sum"]))


# Various calculations for features occur here

members = all_rows[all_rows["member_number"].notna()].copy()

booking_timestamps = pd.to_datetime(
    members["booking_timestamp"], errors="coerce", utc=True
).dt.tz_localize(None)

first_booking = (
    booking_timestamps
    .groupby(members["member_number"])
    .min()
    .dt.normalize()
)

prefix = (
    members["phone_number"]
    .astype("string")
    .fillna("")
    .str.replace(r"\D", "", regex=True)
    .str[:9]
)

# For the stretch goal we can filter to only members
# Can't use the no-show events to filter since those get classiifed later

pairs = pd.DataFrame({
    "member_number": members["member_number"],
    "prefix": prefix
}).drop_duplicates()

pairs["n_members"] = pairs.groupby("prefix")["member_number"].transform("nunique")

phone_sharing = pairs.groupby("member_number")["n_members"].max()

del members, pairs

df = all_rows.copy()
# df = all_rows[all_rows["member_number"].notna()].copy()
# del all_rows


# Various calculations for features occur here

booking_date = (
    pd.to_datetime(df["booking_timestamp"], errors="coerce", utc=True)
    .dt.tz_localize(None)
    .dt.normalize()
)

check_in = pd.to_datetime(df["check_in_date"], errors="coerce")
enrolled = pd.to_datetime(df["member_enrollment_date"], errors="coerce")
cancellation = pd.to_datetime(df["cancellation_date"], errors="coerce")

# Original six features
df["lead_time_days"] = (check_in - booking_date).dt.days
df["account_age_days"] = (booking_date - enrolled).dt.days

df["is_digital"] = df["booking_channel"].isin(
    ["Website", "Mobile App"]
).astype("int8")

df["stay_length_days"] = (
    pd.to_datetime(df["check_out_date"], errors="coerce")
    - pd.to_datetime(df["check_in_date"], errors="coerce")
).dt.days

df["is_sdc4"] = (df["rate_code"] == "SDC4").astype("int8")
df["is_srb"] = (df["rate_code"] == "SRB").astype("int8")

email = df["email"].astype("string").str.lower()

df["is_gmail_or_outlook"] = (
    email.str.split("@").str[1].isin(["gmail.com", "outlook.com"])
).astype("int8")

# Additional account-level features
df["account_age_at_first_booking"] = (
    first_booking.reindex(df["member_number"]).reset_index(drop=True)
    - enrolled.reset_index(drop=True)
).dt.days

df["enrolled_within_30d"] = (
    df["account_age_at_first_booking"] <= 30
).astype("int8")

df["enrolled_online"] = df["member_enrollment_source"].isin(
    ["ONLINE_WEB", "ONLINE_APP"]
).astype("int8")

df["is_late_cancel"] = (
    df["qualification_code"] == "QXY"
).astype("int8")

df["cancel_same_day_as_booking"] = (
    cancellation.dt.normalize().reset_index(drop=True)
    == booking_date.reset_index(drop=True)
).astype("int8")

handle = email.str.split("@").str[0].fillna("")

first = (
    df["first_name"].astype("string").str.lower()
    .str.replace(r"[^a-z]", "", regex=True).fillna("")
)

last = (
    df["last_name"].astype("string").str.lower()
    .str.replace(r"[^a-z]", "", regex=True).fillna("")
)

df["email_initial_lastname_format"] = [
    int(h.startswith(f[:1] + l) and not h.startswith(f + l))
    if f and l else 0
    for h, f, l in zip(handle, first, last)
]

# This are currently iffy

df["members_sharing_phone_prefix"] = (
    phone_sharing.reindex(df["member_number"]).to_numpy()
)

# Sorting will take a long time here

# df = df.sort_values(
#     ["member_number", "check_in_date"]
# ).copy()

# This are currently iffy
df["m_mtd_noshows"] = (
    df.groupby(["member_number", "data_month"]).cumcount()
)

missing_features = [c for c in FEATURES if c not in df.columns]

if missing_features:
    raise ValueError(f"Missing engineered features: {missing_features}")

df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)

booking_ids = df["confirmation_number"].reset_index(drop=True)
member_ids = df["member_number"].reset_index(drop=True)

X = df[FEATURES].reset_index(drop=True)
y = df[TARGET].astype("int8").reset_index(drop=True)

print("\nFeature columns:")
print(FEATURES)
print("\nFeature matrix:", X.shape)


# Identify unique member accounts and whether each account
# has at least one fraudulent booking.
account_labels = (
    df.groupby("member_number")[TARGET]
    .max()
    .reset_index()
)

account_ids = account_labels["member_number"].to_numpy()
account_strata = account_labels[TARGET].to_numpy()

# First split: 70% training accounts, 30% held-out accounts.
train_accounts, heldout_accounts = train_test_split(
    account_ids,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=account_strata
)

# Identify fraud status for the held-out accounts.
heldout_labels = (
    account_labels.set_index("member_number")
    .loc[heldout_accounts, TARGET]
    .to_numpy()
)

# Second split: divide held-out accounts equally:
# 15% validation and 15% testing.
val_accounts, test_accounts = train_test_split(
    heldout_accounts,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=heldout_labels
)

train_accounts = set(train_accounts)
val_accounts = set(val_accounts)
test_accounts = set(test_accounts)

# Training: May and June bookings from training accounts.
train_df = df[
    df["data_month"].isin(["may", "june"])
    & df["member_number"].isin(train_accounts)
].copy()

# Validation: July bookings from validation accounts.
val_df = df[
    (df["data_month"] == "july")
    & df["member_number"].isin(val_accounts)
].copy()

# Testing: July bookings from test accounts.
test_df = df[
    (df["data_month"] == "july")
    & df["member_number"].isin(test_accounts)
].copy()

# Keep original row ordering.
train_df = train_df.sort_index()
val_df = val_df.sort_index()
test_df = test_df.sort_index()

# Separate features and labels.
X_train = train_df[FEATURES].copy()
y_train = train_df[TARGET].astype(int).copy()

X_val = val_df[FEATURES].copy()
y_val = val_df[TARGET].astype(int).copy()

X_test = test_df[FEATURES].copy()
y_test = test_df[TARGET].astype(int).copy()

# Preserve booking and member identifiers.
train_booking_ids = train_df["confirmation_number"].copy()
val_booking_ids = val_df["confirmation_number"].copy()
test_booking_ids = test_df["confirmation_number"].copy()

train_member_ids = train_df["member_number"].copy()
val_member_ids = val_df["member_number"].copy()
test_member_ids = test_df["member_number"].copy()

print("\nAccount-level split:")
print("Training accounts:", len(train_accounts))
print("Validation accounts:", len(val_accounts))
print("Testing accounts:", len(test_accounts))

print("\nBooking-level split:")
print("Training bookings:", len(train_df))
print("Training fraud:", int(y_train.sum()))

print("Validation bookings:", len(val_df))
print("Validation fraud:", int(y_val.sum()))

print("Testing bookings:", len(test_df))
print("Testing fraud:", int(y_test.sum()))

# Verify account separation.
assert not (train_accounts & val_accounts)
assert not (train_accounts & test_accounts)
assert not (val_accounts & test_accounts)

# Verify datasets are populated and contain both classes.
for name, dataset, labels in [
    ("Training", train_df, y_train),
    ("Validation", val_df, y_val),
    ("Testing", test_df, y_test),
]:
    if dataset.empty:
        raise ValueError(f"{name} dataset is empty.")

    if labels.nunique() != 2:
        raise ValueError(
            f"{name} dataset must contain both fraud and non-fraud."
        )

print("\nAccount separation verified successfully.")

# Explicitly identify categorical features.
categorical_features = [
    # "booking_channel",
    # "member_enrollment_source"
]

# All remaining features are numerical.
numeric_features = [
    feature for feature in FEATURES
    if feature not in categorical_features
]

# Check feature types.
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

X_train_numeric = X_train[numeric_features].values
X_val_numeric   = X_val[numeric_features].values
X_test_numeric  = X_test[numeric_features].values

# Encode categorical features.
categorical_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
    dtype=np.float32
)

X_train_categorical = categorical_encoder.fit_transform(
    X_train[categorical_features].fillna("Missing")
)

X_val_categorical = categorical_encoder.transform(
    X_val[categorical_features].fillna("Missing")
)

X_test_categorical = categorical_encoder.transform(
    X_test[categorical_features].fillna("Missing")
)

categorical_feature_names = (
    categorical_encoder.get_feature_names_out(
        categorical_features
    )
)

feature_names = (
    numeric_features +
    categorical_feature_names.tolist()
)

# Combine numerical and categorical features.
X_train_processed = np.column_stack([
    X_train_numeric,
    X_train_categorical
]).astype(np.float32)

X_val_processed = np.column_stack([
    X_val_numeric,
    X_val_categorical
]).astype(np.float32)

X_test_processed = np.column_stack([
    X_test_numeric,
    X_test_categorical
]).astype(np.float32)

print("\nProcessed feature shapes:")
print("Train:", X_train_processed.shape)
print("Validation:", X_val_processed.shape)
print("Test:", X_test_processed.shape)
print("Total features:", len(feature_names))


n_negative = int((y_train == 0).sum())
n_positive = int((y_train == 1).sum())

scale_pos_weight = n_negative / max(n_positive, 1)

print("\nscale_pos_weight:", scale_pos_weight)

# These hyperparameters might not yield the best results, I think they could improve with tuning

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

    # Early stopping configuration.
    early_stopping_rounds=75,

    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Train using May/June accounts.
# Validation data is used for early stopping only.
model.fit(
    X_train_processed,
    y_train,

    eval_set=[
        (X_val_processed, y_val)
    ],

    verbose=True
)

print("\nTraining complete.")

print(
    "Best iteration:",
    getattr(model, "best_iteration", "not available")
)

print(
    "Best validation score:",
    getattr(model, "best_score", "not available")
)

# Generate probabilities for validation bookings.
val_probabilities = model.predict_proba(
    X_val_processed
)[:, 1]

# Calculate precision and recall across thresholds.
precision_values, recall_values, thresholds = precision_recall_curve(
    y_val,
    val_probabilities
)

if len(thresholds) == 0:
    best_threshold = 0.5

else:
    p = precision_values[:-1]
    r = recall_values[:-1]

    f1_values = (
        2 * p * r / np.maximum(p + r, 1e-12)
    )

    # Select threshold that maximizes validation F1.
    best_idx = int(np.nanargmax(f1_values))

    best_threshold = float(thresholds[best_idx])

print("\nValidation-selected threshold:", best_threshold)

# Only now generate predictions for the July test accounts.
test_probabilities = model.predict_proba(
    X_test_processed
)[:, 1]

# Standard threshold.
test_predictions_05 = (
    test_probabilities >= 0.5
).astype(int)

# Validation-optimized threshold.
test_predictions_tuned = (
    test_probabilities >= best_threshold
).astype(int)

print("\nTest predictions generated.")
print("Threshold 0.5:", 0.5)
print("Optimized threshold:", best_threshold)

def calculate_metrics(y_true, probabilities, predictions):
    cm = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp) if (tn + fp) else 0.0
    )

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
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "average_precision_pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "log_loss": float(
            log_loss(y_true, probabilities, labels=[0, 1])
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }

results = {
    "threshold_0.5": calculate_metrics(
        y_test,
        test_probabilities,
        test_predictions_05
    ),
    "validation_tuned_threshold": calculate_metrics(
        y_test,
        test_probabilities,
        test_predictions_tuned
    )
}

results["validation_tuned_threshold"]["decision_threshold"] = (
    best_threshold
)

print("\n" + "=" * 70)
print("HELD-OUT JULY TEST RESULTS")
print("=" * 70)

for label, metrics in results.items():
    print(f"\n--- {label} ---")

    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric:30s}: {value:.6f}")
        else:
            print(f"{metric:30s}: {value}")

print("\nClassification report:")
print(classification_report(
    y_test,
    test_predictions_tuned,
    labels=[0, 1],
    target_names=["Non-fraud", "Fraud"],
    digits=4,
    zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(
    y_test,
    test_predictions_tuned,
    labels=[0, 1]
))

with open(OUTPUT_DIR / "test_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

predictions_df = pd.DataFrame({
    "confirmation_number": test_booking_ids.to_numpy(),
    "actual_is_fraud": y_test.to_numpy(),
    "fraud_probability": test_probabilities,
    "prediction_threshold_0_5": test_predictions_05,
    "prediction_tuned_threshold": test_predictions_tuned
})

predictions_df["actual_label"] = predictions_df[
    "actual_is_fraud"
].map({0: "Non-fraud", 1: "Fraud"})

predictions_df["predicted_label"] = predictions_df[
    "prediction_tuned_threshold"
].map({0: "Non-fraud", 1: "Fraud"})

predictions_df["correct"] = (
    predictions_df["actual_is_fraud"]
    == predictions_df["prediction_tuned_threshold"]
)

predictions_df.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)

report_dict = classification_report(
    y_test,
    test_predictions_tuned,
    labels=[0, 1],
    target_names=["Non-fraud", "Fraud"],
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report_dict).transpose().to_csv(
    OUTPUT_DIR / "classification_report.csv"
)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

print("\nTop 25 XGBoost features:")
print(importance_df.head(25).to_string(index=False))

top_n = min(25, len(importance_df))

plot_df = importance_df.head(top_n).sort_values(
    "importance"
)

plt.figure(figsize=(10, max(6, top_n * 0.28)))
plt.barh(plot_df["feature"], plot_df["importance"])
plt.xlabel("XGBoost feature importance")
plt.ylabel("Feature")
plt.title("Top Fraud Prediction Features")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "feature_importance.png",
    dpi=150
)
plt.close()

rng = np.random.default_rng(RANDOM_STATE)

shap_sample_size = min(5000, X_test_processed.shape[0])

shap_indices = rng.choice(
    X_test_processed.shape[0],
    size=shap_sample_size,
    replace=False
)

X_shap = X_test_processed[shap_indices]

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_shap)

print("\nSHAP calculation complete.")
print("Explained rows:", shap_sample_size)
print("Number of features:", len(feature_names))

# SHAP beeswarm plot
plt.figure()
shap.summary_plot(
    shap_values.values,
    X_shap,
    feature_names=feature_names,
    max_display=25,
    show=False
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shap_summary.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# SHAP bar plot
plt.figure()
shap.summary_plot(
    shap_values.values,
    X_shap,
    feature_names=feature_names,
    plot_type="bar",
    max_display=25,
    show=False
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shap_feature_importance.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# SHAP importance CSV
shap_importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_absolute_shap": np.abs(
        shap_values.values
    ).mean(axis=0)
}).sort_values(
    "mean_absolute_shap",
    ascending=False
)

shap_importance_df.to_csv(
    OUTPUT_DIR / "shap_feature_importance.csv",
    index=False
)

print("\nTop 25 SHAP features:")
print(shap_importance_df.head(25).to_string(index=False))

# Individual SHAP contributions
shap_contributions_df = pd.DataFrame(
    shap_values.values,
    columns=feature_names
)

shap_contributions_df.insert(
    0,
    "confirmation_number",
    test_booking_ids.iloc[shap_indices].to_numpy()
)

shap_contributions_df.to_csv(
    OUTPUT_DIR / "shap_values.csv",
    index=False
)

fpr, tpr, _ = roc_curve(
    y_test,
    test_probabilities
)

plt.figure(figsize=(7, 5))
plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc_score(y_test, test_probabilities):.4f}"
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — July Test Set")
plt.legend()
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "roc_curve.png",
    dpi=150
)
plt.close()

pr_precision, pr_recall, _ = precision_recall_curve(
    y_test,
    test_probabilities
)

plt.figure(figsize=(7, 5))
plt.plot(
    pr_recall,
    pr_precision,
    label=(
        f"Average precision = "
        f"{average_precision_score(y_test, test_probabilities):.4f}"
    )
)
plt.axhline(
    y=float(y_test.mean()),
    linestyle="--",
    color="gray",
    label="Fraud prevalence baseline"
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — July Test Set")
plt.legend()
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "precision_recall_curve.png",
    dpi=150
)
plt.close()

joblib.dump(
    categorical_encoder,
    OUTPUT_DIR / "fraud_categorical_encoder.joblib"
)

joblib.dump(
    model,
    OUTPUT_DIR / "fraud_xgboost_model.joblib"
)

with open(OUTPUT_DIR / "model_metadata.json", "w") as f:
    json.dump(
            {
                "target": TARGET,
                "features": FEATURES,
                "numeric_features": numeric_features,
                "encoded_booking_channel_features": categorical_feature_names.tolist(),

                "random_state": RANDOM_STATE,

                "training_period": ["may", "june"],
                "validation_period": ["july"],
                "testing_period": ["july"],

                "account_split": {
                    "training": 0.70,
                    "validation": 0.15,
                    "testing": 0.15
                },

                "training_accounts": len(train_accounts),
                "validation_accounts": len(val_accounts),
                "testing_accounts": len(test_accounts),

                "best_iteration": int(
                    getattr(model, "best_iteration", -1)
                ),

                "best_validation_score": float(
                    getattr(model, "best_score", -1)
                ),

                "decision_threshold": best_threshold,

                "threshold_selection": "validation data - maximum F1",

                "early_stopping_rounds": 75,

                "scale_pos_weight": float(scale_pos_weight)
            }
    , f, indent=4)

print(f"\nAll output files saved to: {OUTPUT_DIR.resolve()}")
print("Fraud detection pipeline finished successfully.")
