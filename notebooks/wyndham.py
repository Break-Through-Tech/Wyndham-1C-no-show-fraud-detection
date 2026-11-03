"""
Wyndham fraud detection: baseline model on the team's parquet files, ALL features.

What is different from baseline_parquet.py:
  The first version silently skipped date columns, so lead_time_days and
  account_age_days (the two strongest signals) never reached the model.
  This version builds them, plus stay_length_days, from the date columns.
  If the parquet has no date columns, it pulls them from the raw
  reservations_*.csv files by confirmation_number.

Setup:
    pip install duckdb pandas scikit-learn matplotlib

Trains on May+June, tests on July. Saves one dashboard image with:
  A) top features   B) precision-recall curve
  C) confusion matrix at 90% recall   D) analyst workload

Post-stay columns (qualification code, points earned, revenue, cancellation) stay
excluded on purpose. They only exist after the stay is processed and restate the label.
Set MODEL to "logistic" or "tree".
"""

import os
import re
import time

import duckdb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = r"C:\Users\Maisha\Desktop\wyndham"   # folder with the parquet files (and raw CSVs)
TRAIN_FILE = "bookings_train_may_june.parquet"
TEST_FILE = "bookings_test_july.parquet"
TRAIN_MONTHS = ["may", "june"]
TEST_MONTHS = ["july"]
RAW_CSV = {                                      # only used if the parquet has no date columns
    "may": "reservations_may_2026.csv",
    "june": "reservations_june_2026.csv",
    "july": "reservations_july_2026.csv",
}

LABEL_COL = None        # None = auto-detect. Or set it, e.g. "is_fraud_event"
MODEL = "logistic"      # "logistic" or "tree"
DROP_LEAKY = True       # drop columns that only exist after the stay is processed
MAX_ONEHOT = 15         # text columns with this many values or fewer get one-hot encoded
SHOW_PLOTS = False
OPEN_WHEN_DONE = True

LABEL_CANDIDATES = ["is_fraud_event", "is_fraud", "fraud_label", "label", "target",
                    "proxy_label", "is_fraud_account", "y"]
ID_COLS = {"confirmation_number", "member_number", "first_name", "last_name", "email",
           "phone_number", "address_line_1", "city", "state", "zip", "site_name"}
LEAKY_KEYWORDS = ["qualif", "points_earned", "revenue", "cancel"]
REVIEW_KEYWORDS = ["no_show", "noshow", "qns", "qxy", "qxn", "redeem", "points"]
STRONG = ["lead_time_days", "account_age_days"]

if not SHOW_PLOTS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LEGIT_C = "#4C72B0"
FRAUD_C = "#C44E52"
OUT_PATH = os.path.join(DATA_DIR, f"baseline_{MODEL}_parquet_all_features.png")

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})

con = duckdb.connect()
try:
    con.execute("SET TimeZone='UTC'")
except Exception:
    pass

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def step(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def path(fname):
    return f"{DATA_DIR}/{fname}"


def q(name):
    return '"' + name.replace('"', '""') + '"'


# ---------------------------------------------------------------------------
# STEP 1: look at the files
# ---------------------------------------------------------------------------
step("STEP 1: schema of the parquet files")


def describe(fname):
    d = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path(fname)}')").df()
    return d[["column_name", "column_type"]]


train_schema = describe(TRAIN_FILE)
test_schema = describe(TEST_FILE)
n_train_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{path(TRAIN_FILE)}')").fetchone()[0]
n_test_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{path(TEST_FILE)}')").fetchone()[0]
print(f"{TRAIN_FILE}: {n_train_rows:,} rows, {len(train_schema)} columns")
print(f"{TEST_FILE}: {n_test_rows:,} rows, {len(test_schema)} columns")

types = dict(zip(train_schema.column_name, train_schema.column_type))
test_cols = set(test_schema.column_name)
common = [c for c in train_schema.column_name if c in test_cols]
only_one = set(train_schema.column_name) ^ test_cols
if only_one:
    print(f"\nColumns in only one file (ignored): {sorted(only_one)}")
print("\nShared columns:")
for c in common:
    print(f"  {c:42s} {types[c]}")

# ---------------------------------------------------------------------------
# STEP 2: timing features (lead time, account age, stay length)
# ---------------------------------------------------------------------------
step("STEP 2: building the timing features from the date columns")
DATE_TYPE = re.compile(r"^(DATE|TIMESTAMP.*)$", re.I)


def find_date(exact, key):
    for c in common:
        if c.lower() == exact and DATE_TYPE.match(types[c]):
            return c
    for c in common:
        if key in c.lower() and DATE_TYPE.match(types[c]):
            return c
    return None


roles = {
    "book": find_date("booking_timestamp", "booking"),
    "checkin": find_date("check_in_date", "check_in"),
    "checkout": find_date("check_out_date", "check_out"),
    "enroll": find_date("member_enrollment_date", "enroll"),
}
print("Date columns found in the parquet:")
for r, c in roles.items():
    print(f"  {r:9s} {c}")

need = {}
if "lead_time_days" not in common:
    need["lead_time_days"] = ("book", "checkin")
if "account_age_days" not in common:
    need["account_age_days"] = ("book", "enroll")
if "stay_length_days" not in common:
    need["stay_length_days"] = ("checkin", "checkout")
already = [f for f in ["lead_time_days", "account_age_days", "stay_length_days"] if f in common]
if already:
    print(f"\nAlready in the parquet, used as-is: {already}")

missing_roles = {r for pair in need.values() for r in pair if roles[r] is None}
use_raw = False
if missing_roles:
    print(f"\nThe parquet has no usable date column for: {sorted(missing_roles)}")
    raw_paths = {m: path(RAW_CSV[m]) for m in set(TRAIN_MONTHS + TEST_MONTHS)}
    absent = sorted(m for m, p in raw_paths.items() if not os.path.exists(p))
    if "confirmation_number" not in common:
        print("No confirmation_number column, so the raw CSVs can't be joined.")
    elif absent:
        print("Raw CSVs not found for: " + ", ".join(f"{m} ({RAW_CSV[m]})" for m in absent))
        print(f"Put those files in {DATA_DIR} and rerun, or ask your teammate to add the "
              "date columns to the parquet.")
    else:
        use_raw = True
        print("Joining to the raw reservations CSVs on confirmation_number to get the dates.")
        alias = {"book": "_bt", "checkin": "_ci", "checkout": "_co", "enroll": "_en"}
        for r in missing_roles:
            roles[r] = alias[r]


def relation(fname, months):
    base = f"read_parquet('{path(fname)}')"
    if not use_raw:
        return base
    files = "[" + ", ".join(f"'{path(RAW_CSV[m])}'" for m in months) + "]"
    raw = ("(SELECT confirmation_number AS _cn, "
           "TRY_CAST(TRY_CAST(booking_timestamp AS TIMESTAMPTZ) AS DATE) AS _bt, "
           "TRY_CAST(check_in_date AS DATE) AS _ci, "
           "TRY_CAST(check_out_date AS DATE) AS _co, "
           "TRY_CAST(member_enrollment_date AS DATE) AS _en "
           f"FROM read_csv({files}, header=true, all_varchar=true))")
    return (f"(SELECT p.*, r._bt, r._ci, r._co, r._en FROM {base} p "
            f"LEFT JOIN {raw} r ON p.confirmation_number = r._cn)")


def D(role):
    return f"TRY_CAST({q(roles[role])} AS DATE)"


derived = []
for name, (a, b) in need.items():
    if roles[a] is None or roles[b] is None:
        print(f"Cannot build {name} (missing a date column).")
        continue
    derived.append((name, f"DATE_DIFF('day', {D(a)}, {D(b)})"))
if derived:
    print("\nBuilt from dates:")
    for name, _ in derived:
        print(f"  {name}")

# ---------------------------------------------------------------------------
# STEP 3: find the label and choose features
# ---------------------------------------------------------------------------
step("STEP 3: label and feature selection")

label = LABEL_COL or next((c for c in LABEL_CANDIDATES if c in common), None)
if label is None:
    raise SystemExit(
        "\nCould not find a label column. Look at the shared columns above, then set\n"
        "LABEL_COL = \"<the fraud label column>\" near the top of this file and rerun."
    )
other_labels = [c for c in LABEL_CANDIDATES if c in common and c != label]
print(f"Label column: {label}")
if other_labels:
    print(f"Other label-like columns (excluded from features): {other_labels}")

NUM_TYPE = re.compile(
    r"^(BOOLEAN|TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT|UTINYINT|USMALLINT|UINTEGER|UBIGINT|"
    r"FLOAT|DOUBLE|REAL|DECIMAL.*)$", re.I)
TEXT_TYPE = re.compile(r"^(VARCHAR|ENUM.*|STRING)$", re.I)

if not NUM_TYPE.match(types[label]):
    raise SystemExit(f"\nLabel column {label} has type {types[label]}. It must be 0/1 or boolean.")


def drop_reason(col):
    low = col.lower()
    if col == label:
        return "is the label"
    if "fraud" in low:
        return "name contains 'fraud' (label leakage guard)"
    if col in ID_COLS:
        return "identifier or personal field"
    if DROP_LEAKY and any(k in low for k in LEAKY_KEYWORDS):
        return "only exists after the stay is processed"
    return None


dropped, num_cols, text_cols, skipped_other = [], [], [], []
for c in common:
    why = drop_reason(c)
    if why:
        dropped.append((c, why))
    elif NUM_TYPE.match(types[c]):
        num_cols.append(c)
    elif TEXT_TYPE.match(types[c]):
        text_cols.append(c)
    else:
        skipped_other.append((c, types[c]))

onehot = {}
skipped_text = []
for c in text_cols:
    n = con.execute(f"SELECT COUNT(DISTINCT {q(c)}) FROM read_parquet('{path(TRAIN_FILE)}')").fetchone()[0]
    if n <= MAX_ONEHOT:
        cats = con.execute(
            f"SELECT DISTINCT {q(c)} FROM read_parquet('{path(TRAIN_FILE)}') "
            f"WHERE {q(c)} IS NOT NULL ORDER BY 1").df().iloc[:, 0].tolist()
        onehot[c] = cats
    else:
        skipped_text.append((c, n))

print("\nDropped:")
for c, why in dropped:
    print(f"  {c:42s} {why}")
if skipped_text:
    print("\nSkipped text columns (too many distinct values):")
    for c, n in skipped_text:
        print(f"  {c:42s} {n:,} values")
if skipped_other:
    print("\nRaw date and other columns not used directly (dates feed the timing features):")
    for c, t in skipped_other:
        print(f"  {c:42s} {t}")

review = [c for c in num_cols + list(onehot) if any(k in c.lower() for k in REVIEW_KEYWORDS)]
if review:
    print("\nREVIEW these kept columns. They leak if computed from the booking's own outcome,")
    print("and are fine if computed only from the member's earlier bookings:")
    for c in review:
        print(f"  {c}")

# ---------------------------------------------------------------------------
# STEP 4: load only the columns we need
# ---------------------------------------------------------------------------
step("STEP 4: loading data")
has_conf = "confirmation_number" in common
select_cols = [f"CAST({q(label)} AS INTEGER) AS {q(label)}"]
if has_conf:
    select_cols.append(q("confirmation_number"))
select_cols += [f"CAST({q(c)} AS FLOAT) AS {q(c)}" for c in num_cols]
select_cols += [f"CAST({expr} AS FLOAT) AS {q(name)}" for name, expr in derived]
select_cols += [q(c) for c in onehot]
num_all = num_cols + [name for name, _ in derived]


def load(fname, months):
    return con.execute(
        f"SELECT {', '.join(select_cols)} FROM {relation(fname, months)} "
        f"WHERE {q(label)} IS NOT NULL").df()


train_raw = load(TRAIN_FILE, TRAIN_MONTHS)
test_raw = load(TEST_FILE, TEST_MONTHS)


def build_X(df):
    parts = {c: df[c].astype("float32") for c in num_all}
    for c, cats in onehot.items():
        for cat in cats:
            parts[f"{c}={cat}"] = (df[c] == cat).astype("float32")
    return pd.DataFrame(parts)


X_train, y_train = build_X(train_raw), train_raw[label]
X_test, y_test = build_X(test_raw), test_raw[label]
feat_names = list(X_train.columns)
y_arr = y_test.values

print(f"Features used: {len(feat_names)}")
for f in feat_names:
    print(f"  {f}")

print("\nTiming feature coverage (share of rows with no value):")
for name, _ in derived:
    print(f"  {name:20s} train {train_raw[name].isna().mean():.2%}   test {test_raw[name].isna().mean():.2%}")
if "account_age_days" in feat_names:
    print("  (account_age_days is empty for nonmember rows, which have no enrollment date. "
          "They get the median.)")
if "lead_time_days" in feat_names and max(train_raw["lead_time_days"].isna().mean(),
                                         test_raw["lead_time_days"].isna().mean()) > 0.01:
    print("  WARNING: lead_time_days is missing for more than 1% of rows. The join to the raw "
          "CSVs likely did not match. Check that each month's CSV is the right file.")

strong_present = [f for f in STRONG if f in feat_names]
strong_missing = [f for f in STRONG if f not in feat_names]
if strong_missing:
    print(f"\nWARNING: strong features NOT in the model: {strong_missing}")
    timing_note = "timing features MISSING: " + ", ".join(strong_missing)
else:
    print(f"\nStrong features included: {strong_present}")
    timing_note = "timing features included (lead time, account age)"

# ---------------------------------------------------------------------------
# STEP 5: label and integrity check
# ---------------------------------------------------------------------------
step("STEP 5: label and integrity check")
pos_tr, pos_te = int(y_train.sum()), int(y_test.sum())
print(f"Train: {len(y_train):,} rows, {pos_tr:,} positives ({y_train.mean():.4%})")
print(f"Test:  {len(y_test):,} rows, {pos_te:,} positives ({y_test.mean():.4%})")
if pos_te == 1600:
    label_note = "test positives match the 1,600 ground-truth fraud events per month"
elif 1900 <= pos_te <= 2300:
    label_note = "test positives are close to 2,100, which looks like the proxy label, not the ground truth"
else:
    label_note = "test positives match neither 1,600 (ground truth) nor 2,100 (proxy), check how the label was built"
print(f"Label check: {label_note}")

if has_conf:
    overlap = int(test_raw["confirmation_number"].isin(train_raw["confirmation_number"]).sum())
    integ_note = f"{overlap:,} confirmation numbers shared between train and test (must be 0)"
else:
    overlap = 0
    integ_note = "no confirmation_number column, overlap not checked"
print(f"Overlap check: {integ_note}")

# ---------------------------------------------------------------------------
# STEP 6: train
# ---------------------------------------------------------------------------
step(f"STEP 6: training {MODEL} (can take a few minutes on millions of rows)")
t0 = time.time()
if MODEL == "logistic":
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
             ("scale", StandardScaler()),
             ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))]
else:
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
             ("clf", DecisionTreeClassifier(max_depth=8, min_samples_leaf=50,
                                            class_weight="balanced", random_state=0))]
model = Pipeline(steps)
model.fit(X_train, y_train)
print(f"Done in {time.time() - t0:.0f} seconds.")

# ---------------------------------------------------------------------------
# STEP 7: evaluate
# ---------------------------------------------------------------------------
step("STEP 7: test performance")
probs = model.predict_proba(X_test)[:, 1]
pr_auc = average_precision_score(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)
print(f"PR-AUC:  {pr_auc:.4f}   (a random guesser would score {y_test.mean():.4f})")
print(f"ROC-AUC: {roc_auc:.4f}")

precision, recall, thresholds = precision_recall_curve(y_test, probs)


def workload_at_recall(target):
    ok = np.where(recall[:-1] >= target)[0]
    if len(ok) == 0:
        return None
    i = ok[-1]
    thr = float(thresholds[i])
    flagged = int((probs >= thr).sum())
    caught = int(round(precision[i] * flagged))
    return thr, flagged, caught, flagged - caught, float(precision[i])


rows = []
for target in [0.50, 0.70, 0.90, 0.95, 0.99]:
    w = workload_at_recall(target)
    if w:
        thr, flagged, caught, fa, prec = w
        rows.append({"target_recall": f"{target:.0%}", "threshold": round(thr, 4),
                     "rows_flagged": flagged, "fraud_caught": caught,
                     "false_alarms": fa, "precision": f"{prec:.2%}"})
print("\nRecall vs analyst workload:")
print(pd.DataFrame(rows).to_string(index=False))

clf = model.named_steps["clf"]
if MODEL == "logistic":
    imp = pd.DataFrame({"feature": feat_names, "value": clf.coef_[0]})
else:
    imp = pd.DataFrame({"feature": feat_names, "value": clf.feature_importances_})
imp = imp.reindex(imp["value"].abs().sort_values(ascending=False).index).head(15)
print("\nTop features:")
print(imp.to_string(index=False))

# ---------------------------------------------------------------------------
# STEP 8: dashboard
# ---------------------------------------------------------------------------
step("STEP 8: building the dashboard")
w90 = workload_at_recall(0.90)
T = dict(fontsize=11, fontweight="bold", loc="left")

fig, axes = plt.subplots(2, 2, figsize=(17, 12.5))
ax_feat, ax_pr, ax_cm, ax_work = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

# A) features
d = imp.iloc[::-1]
if MODEL == "logistic":
    ax_feat.barh(d["feature"], d["value"],
                 color=[FRAUD_C if v > 0 else LEGIT_C for v in d["value"]])
    ax_feat.set_xscale("symlog", linthresh=1)
    ax_feat.axvline(0, color="black", linewidth=0.8)
    for y, v in enumerate(d["value"]):
        ax_feat.text(v, y, f" {v:.1f} ", va="center", ha="left" if v > 0 else "right", fontsize=8)
    ax_feat.set_xlabel("Standardized coefficient (symlog)")
    ax_feat.set_title("A. Top features: red pushes toward the fraud label, blue away\n"
                      "trust sign and rank, not exact size", **T)
else:
    ax_feat.barh(d["feature"], d["value"], color=FRAUD_C)
    ax_feat.set_xlabel("Importance (share of the tree's splitting power)")
    ax_feat.set_title("A. Top features by importance\nhigher means the tree leans on it more", **T)
ax_feat.tick_params(axis="y", labelsize=8)

# B) precision-recall
idx = np.linspace(0, len(recall) - 1, min(5000, len(recall))).astype(int)
ax_pr.plot(recall[idx], precision[idx], color=FRAUD_C, linewidth=2, label=f"Model (PR-AUC {pr_auc:.3f})")
ax_pr.axhline(y_test.mean(), color="gray", linestyle="--", label=f"Random guesser ({y_test.mean():.4f})")
for target in [0.50, 0.70, 0.90, 0.95, 0.99]:
    w = workload_at_recall(target)
    if w:
        ax_pr.scatter([target], [w[4]], color="black", zorder=5)
        ax_pr.annotate(f"{target:.0%} recall\n{w[4]:.0%} prec.", (target, w[4]),
                       textcoords="offset points", xytext=(0, -30), fontsize=8, ha="center")
ax_pr.set_xlabel("Recall (share of fraud caught)")
ax_pr.set_ylabel("Precision (share of flags that are fraud)")
ax_pr.set_ylim(0, 1.05)
ax_pr.set_xlim(0, 1.03)
ax_pr.set_title("B. Precision-recall curve\ncatching more fraud costs more false alarms", **T)
ax_pr.legend(fontsize=9, loc="lower left")

# C) confusion matrix at 90% recall
if w90:
    cm = confusion_matrix(y_test, (probs >= w90[0]).astype(int))
    row_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    ax_cm.imshow(row_pct, cmap="Blues", vmin=0, vmax=100)
    ax_cm.set_xticks([0, 1])
    ax_cm.set_xticklabels(["Predicted legit", "Predicted fraud"])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_yticklabels(["Actually legit", "Actually fraud"])
    ax_cm.grid(False)
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, f"{cm[i, j]:,}\n({row_pct[i, j]:.2f}% of row)", ha="center",
                       va="center", fontsize=11, color="white" if row_pct[i, j] > 50 else "black")
    ax_cm.set_title(f"C. Confusion matrix at 90% recall (threshold {w90[0]:.3f})\n"
                    "bottom-left = missed fraud, top-right = false alarms", **T)
else:
    ax_cm.axis("off")

# D) analyst workload
sweep = []
for t in np.linspace(0.05, 0.99, 40):
    w = workload_at_recall(t)
    if w:
        sweep.append((t, w[1], w[3]))
sweep = np.array(sweep)
if len(sweep):
    ax_work.plot(sweep[:, 0] * 100, sweep[:, 1], color="black", linewidth=2, label="Rows flagged for review")
    ax_work.plot(sweep[:, 0] * 100, sweep[:, 2], color=FRAUD_C, linewidth=2, label="False alarms (wasted review)")
    ax_work.fill_between(sweep[:, 0] * 100, sweep[:, 2], color=FRAUD_C, alpha=0.15)
    if w90:
        ax_work.scatter([90], [w90[1]], color="black", zorder=5)
        ax_work.annotate(f"90% recall:\n{w90[1]:,} flagged,\n{w90[3]:,} false alarms", (90, w90[1]),
                         textcoords="offset points", xytext=(-120, 10), fontsize=9,
                         arrowprops=dict(arrowstyle="->"))
ax_work.set_xlabel("Target recall (% of fraud caught)")
ax_work.set_ylabel("Rows an analyst must review")
ax_work.set_title(f"D. Analyst workload (of {len(y_arr):,} test rows)\n"
                  "the last points of recall cost the most review time", **T)
ax_work.legend(fontsize=9, loc="upper left")

head = (f"Wyndham fraud baseline ({MODEL}): train {'+'.join(TRAIN_MONTHS)} ({len(y_train):,} rows, {pos_tr:,} positives), "
        f"test {'+'.join(TEST_MONTHS)} ({len(y_test):,} rows, {pos_te:,} positives)\n"
        f"PR-AUC {pr_auc:.3f}   ROC-AUC {roc_auc:.4f}   {len(feat_names)} features, {timing_note}")
if w90:
    head += f"\nAt 90% recall: {w90[1]:,} flagged, {w90[3]:,} false alarms, {w90[4]:.0%} precision"
head += f"\nLabel check: {label_note}.   Overlap check: {integ_note}."
fig.suptitle(head, fontsize=12, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.91])
fig.savefig(OUT_PATH, dpi=110, bbox_inches="tight")
print(f"Saved dashboard: {OUT_PATH}")
if SHOW_PLOTS:
    plt.show()
plt.close(fig)

if OPEN_WHEN_DONE:
    try:
        os.startfile(OUT_PATH)
    except Exception:
        pass