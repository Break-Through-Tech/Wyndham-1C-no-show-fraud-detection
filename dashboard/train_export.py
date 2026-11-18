"""
train_and_export.py: rebuild the team's EBM (same features and settings as
notebooks/Wyndham_1CD_EBM_Model_Test.ipynb), train it, score the test month, and write the files
the dashboard needs.

    python train_and_export.py --data-dir /path/to/folder/with/parquets

Reads   bookings_train_may_june.parquet, bookings_test_july.parquet   (in --data-dir)
Writes  dashboard_data.json          alerts for the dashboard (scores + per-signal reasons)
        models/ebm_model.joblib      the trained model, its feature list and reference scores
        models/run_summary.json      metrics for this run

Options
    --unseen     score only held-out accounts (the notebook's stricter "Test B")
    --no-price   drop room_revenue, points_earned and account_age_days (the notebook's robustness model)
    --min-score  lowest score that becomes an alert (default 30)
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from export_scores import export_alerts, to_score

HERE = os.path.dirname(os.path.abspath(__file__))
REQUIRED = ["confirmation_number", "member_number", "booking_timestamp", "check_in_date", "check_out_date",
            "cancellation_date", "rate_code", "room_revenue", "points_earned", "booking_channel",
            "qualification_code", "member_enrollment_date", "member_enrollment_source", "first_name",
            "last_name", "email", "phone_number", "data_month", "is_fraud"]
OPTIONAL = ["site_id", "site_name", "site_country", "site_region"]
MURTAZA = ["lead_time_days", "account_age_days", "is_digital", "is_sdc4", "is_srb", "is_gmail_or_outlook",
           "room_revenue", "points_earned", "booking_channel"]
EXTRA = ["account_age_at_first_booking", "enrolled_within_30d", "enrolled_online", "is_late_cancel",
         "cancel_same_day_as_booking", "email_initial_lastname_format", "members_sharing_phone_prefix", "m_mtd_noshows"]
SETTINGS = dict(interactions=5, outer_bags=4, max_rounds=2000, random_state=42)


def dtn(s):
    """Parse to timezone-naive datetimes (the raw files mix formats)."""
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None)


def read(path, cols, filters=None):
    have = set(pq.read_schema(path).names)
    missing = [c for c in REQUIRED if c in cols and c not in have]
    if missing:
        raise SystemExit(f"{os.path.basename(path)} is missing columns: {missing}")
    return pd.read_parquet(path, columns=[c for c in cols if c in have], filters=filters)


def build_features(df, first_booking, phone_sharing):
    book, check_in, enrolled = dtn(df["booking_timestamp"]).dt.normalize(), dtn(df["check_in_date"]), dtn(df["member_enrollment_date"])
    df["lead_time_days"] = (check_in - book).dt.days
    df["account_age_days"] = (book - enrolled).dt.days
    df["is_digital"] = df["booking_channel"].isin(["Website", "Mobile App"]).astype(int)
    df["is_sdc4"] = (df["rate_code"] == "SDC4").astype(int)
    df["is_srb"] = (df["rate_code"] == "SRB").astype(int)
    email = df["email"].str.lower()
    df["is_gmail_or_outlook"] = email.str.split("@").str[1].isin(["gmail.com", "outlook.com"]).astype(int)
    # whole-history facts (computed before the QNS/QXY filter, as in the notebook)
    df["account_age_at_first_booking"] = (df["member_number"].map(first_booking) - enrolled).dt.days
    df["enrolled_within_30d"] = (df["account_age_at_first_booking"] <= 30).astype(int)
    df["enrolled_online"] = df["member_enrollment_source"].isin(["ONLINE_WEB", "ONLINE_APP"]).astype(int)
    df["is_late_cancel"] = (df["qualification_code"] == "QXY").astype(int)
    df["cancel_same_day_as_booking"] = (dtn(df["cancellation_date"]) == book).astype(int)
    handle = email.str.split("@").str[0].fillna("")
    first = df["first_name"].str.lower().str.replace(r"[^a-z]", "", regex=True).fillna("")
    last = df["last_name"].str.lower().str.replace(r"[^a-z]", "", regex=True).fillna("")
    df["email_initial_lastname_format"] = [int(h.startswith(f[:1] + l) and not h.startswith(f + l)) if f and l else 0
                                           for h, f, l in zip(handle, first, last)]
    df["members_sharing_phone_prefix"] = df["member_number"].map(phone_sharing)
    df = df.sort_values(["member_number", "check_in_date"])
    df["m_mtd_noshows"] = df.groupby(["member_number", "data_month"]).cumcount()   # earlier no-shows this month only
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--train", default="bookings_train_may_june.parquet")
    ap.add_argument("--test", default="bookings_test_july.parquet")
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--unseen", action="store_true")
    ap.add_argument("--no-price", action="store_true")
    ap.add_argument("--min-score", type=int, default=30)
    a = ap.parse_args()
    paths = [os.path.join(a.data_dir, a.train), os.path.join(a.data_dir, a.test)]
    os.makedirs(os.path.join(a.out_dir, "models"), exist_ok=True)

    print("1/4 whole-history facts (first booking date, shared phone prefixes) ...")
    hist = pd.concat([read(p, ["member_number", "booking_timestamp", "phone_number"]) for p in paths])
    hist = hist[hist["member_number"].notna()]
    first_booking = dtn(hist["booking_timestamp"]).groupby(hist["member_number"]).min().dt.normalize()
    prefix = hist["phone_number"].astype(str).str.replace(r"\D", "", regex=True).str[:9]
    pairs = pd.DataFrame({"member_number": hist["member_number"], "prefix": prefix}).drop_duplicates()
    pairs["n"] = pairs.groupby("prefix")["member_number"].transform("count")
    phone_sharing = pairs.groupby("member_number")["n"].max()
    del hist, pairs

    print("2/4 member no-shows and late cancels, with features ...")
    cols = REQUIRED + OPTIONAL
    rows = pd.concat([read(p, cols, [("qualification_code", "in", ["QNS", "QXY"])]) for p in paths], ignore_index=True)
    df = build_features(rows[rows["member_number"].notna()].copy(), first_booking, phone_sharing)
    feats = MURTAZA + EXTRA
    if a.no_price:
        feats = [f for f in feats if f not in ("room_revenue", "points_earned", "account_age_days")]
    train, test = df[df["data_month"].isin(["may", "june"])], df[df["data_month"] == "july"]
    if a.unseen:   # hold out 30% of accounts entirely, as in the notebook's Test B
        rng = np.random.RandomState(42)
        fraud = np.sort(df.loc[df["is_fraud"] == 1, "member_number"].unique())
        normal = np.setdiff1d(df["member_number"].unique(), fraud)
        held = set(rng.choice(fraud, int(.3 * len(fraud)), replace=False)) | set(rng.choice(normal, int(.3 * len(normal)), replace=False))
        train, test = train[~train["member_number"].isin(held)], test[test["member_number"].isin(held)]
    print(f"    train {len(train):,} rows ({int(train['is_fraud'].sum()):,} fraud) | test {len(test):,} rows ({int(test['is_fraud'].sum()):,} fraud)")

    print("3/4 training the EBM ...")
    t0 = time.time()
    model = ExplainableBoostingClassifier(**SETTINGS).fit(train[feats], train["is_fraud"])
    ref_logits = model.decision_function(train[feats])
    print(f"    done in {time.time() - t0:.0f}s")

    print("4/4 scoring the test month and writing files ...")
    out_json = os.path.join(a.out_dir, "dashboard_data.json")
    export_alerts(model, test, feats, ref_logits, out=out_json, min_score=a.min_score)
    joblib.dump({"model": model, "features": feats, "ref_logits": ref_logits}, os.path.join(a.out_dir, "models", "ebm_model.joblib"))

    logit = model.decision_function(test[feats])
    flagged, y = to_score(logit, ref_logits) >= a.min_score, test["is_fraud"].to_numpy()
    summary = {"features": "no_price" if a.no_price else "extended", "unseen_accounts": a.unseen, "min_score": a.min_score,
               "roc_auc": round(float(roc_auc_score(y, logit)), 4), "pr_auc": round(float(average_precision_score(y, logit)), 4),
               "alerts": int(flagged.sum()), "fraud_caught": int(y[flagged].sum()), "fraud_total": int(y.sum()),
               "recall": round(float(y[flagged].sum() / max(y.sum(), 1)), 4),
               "precision": round(float(y[flagged].mean()) if flagged.any() else 0.0, 4)}
    json.dump(summary, open(os.path.join(a.out_dir, "models", "run_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()