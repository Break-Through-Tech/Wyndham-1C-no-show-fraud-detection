"""
export_scores.py: turn a trained EBM into the JSON the dashboard reads (dashboard_data.json).
Normally called by train_and_export.py, but you can also call it from a notebook:

    from export_scores import export_alerts
    export_alerts(model, test_df, feats, ref_logits=model.decision_function(train_df[feats]))
"""
import json
import numpy as np
import pandas as pd

# Plain-English text per model term. Keys must match model.term_names_.
# Terms not listed (including pairwise interactions such as "a & b") get a generic label.
TEXT = {
    "room_revenue": ("High room rate", lambda r: f"Rate of ${r.room_revenue:,.0f} is {r.room_revenue / 181:.1f}x the typical no-show stay"),
    "points_earned": ("High points earned", lambda r: f"{int(r.points_earned):,} points posted on this stay"),
    "lead_time_days": ("Last-minute booking", lambda r: f"Booked {int(r.lead_time_days)} days before check-in (typical: 42)"),
    "account_age_days": ("New account", lambda r: f"Account was {int(r.account_age_days)} days old at booking"),
    "account_age_at_first_booking": ("New account", lambda r: f"Account was {int(r.account_age_at_first_booking)} days old at first booking"),
    "enrolled_within_30d": ("Recently enrolled", lambda r: "Enrolled within 30 days of the first booking"),
    "m_mtd_noshows": ("Repeat no-shows", lambda r: f"{int(r.m_mtd_noshows)} earlier no-shows by this member this month"),
    "members_sharing_phone_prefix": ("Shared phone number", lambda r: f"{int(r.members_sharing_phone_prefix)} member accounts share this phone prefix"),
    "email_initial_lastname_format": ("Email pattern", lambda r: "Email follows the first-initial + last-name pattern seen in confirmed fraud"),
    "cancel_same_day_as_booking": ("Same-day cancel", lambda r: "Cancelled the same day it was booked"),
    "is_late_cancel": ("Late cancellation", lambda r: "Late cancel (earns points like a no-show)"),
    "is_digital": ("Digital booking", lambda r: "Booked on the website or app"),
    "enrolled_online": ("Online enrollment", lambda r: "Loyalty account was created online"),
    "is_gmail_or_outlook": ("Gmail or Outlook email", lambda r: "Email is on a Gmail or Outlook address"),
    "is_srb": ("Free-night redemption", lambda r: "Booked with a points redemption rate"),
    "is_sdc4": ("Rate code SDC4", lambda r: "Booked on rate code SDC4 (earns double points)"),
    "booking_channel": ("Booking channel", lambda r: f"Booked via {r.booking_channel}"),
}


def to_score(logit, ref_logits, floor_pct=90):
    """0-100 risk score from the model's log-odds.
    Step 1: percentile of the booking against the TRAINING score distribution (stable across days;
            raw probabilities sit at ~0 or ~1, so they cannot be compared).
    Step 2: rescale so the riskiest 10% of historical bookings span 0-100 (floor_pct=90).
            0 = not in the riskiest 10%, 100 = riskiest seen in training."""
    ref = np.sort(np.asarray(ref_logits))
    pct = 100 * np.searchsorted(ref, logit, side="right") / len(ref)
    return np.clip(np.round((pct - floor_pct) / (100 - floor_pct) * 100), 0, 100).astype(int)


def _get(r, col, default=""):
    return r[col] if col in r.index and pd.notna(r[col]) else default


def export_alerts(model, df, feats, ref_logits, out="dashboard_data.json", min_score=30, top_k=6,
                  date_col="check_in_date", floor_pct=90):
    """date_col is the day the alert is filed under in the dashboard (default: the check-in / no-show day)."""
    df = df.reset_index(drop=True)
    contrib = model.eval_terms(df[feats])              # (rows, terms) additive log-odds pieces
    names = list(model.term_names_)
    logit = contrib.sum(axis=1) + model.intercept_     # identical to model.decision_function
    score = to_score(logit, ref_logits, floor_pct)

    pos = np.clip(contrib, 0, None)                    # only terms pushing toward fraud become reasons
    points = pos / np.maximum(pos.sum(axis=1, keepdims=True), 1e-9) * score[:, None]  # reasons sum to the score

    day = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_localize(None)
    los = (pd.to_datetime(df["check_out_date"]) - pd.to_datetime(df["check_in_date"])).dt.days
    alerts = []
    for i in np.where(score >= min_score)[0]:
        r = df.iloc[i]
        agg = {}                                       # merge terms that share a label (e.g. two "New account" terms)
        for j in np.where(points[i] >= 0.5)[0]:
            label, fn = TEXT.get(names[j], (names[j].replace("_", " ").title(), None))
            text = fn(r) if fn and names[j] in df.columns else f"{label} contributed to this score"
            agg.setdefault(label, {"n": label, "t": text, "c": 0.0})["c"] += points[i, j]
        top = sorted(agg.values(), key=lambda x: -x["c"])[:top_k]
        reasons = [{**x, "c": int(round(x["c"]))} for x in top]
        rest = int(score[i]) - sum(x["c"] for x in reasons)   # keep the bars summing to the score
        if rest >= 1:
            reasons.append({"n": "Other signals", "t": "Smaller contributions from other features", "c": rest})
        elif rest < 0 and reasons:                      # rounding pushed the total over the score
            reasons[0]["c"] += rest
        country = str(_get(r, "site_country"))
        region = str(_get(r, "site_region")) or ("APAC" if country == "CN" else "NAMER" if country else "Unknown")
        alerts.append({
            "id": r["confirmation_number"], "date": day.iloc[i].strftime("%Y-%m-%d"),
            "m": {"id": r["member_number"], "age": int(_get(r, "account_age_at_first_booking", 0)),
                  "share": int(_get(r, "members_sharing_phone_prefix", 1)), "src": str(_get(r, "member_enrollment_source"))},
            "site": [str(_get(r, "site_id")), str(_get(r, "site_name")), country], "reg": region,
            "los": int(los.iloc[i]), "lead": int(_get(r, "lead_time_days", 0)), "rev": float(r["room_revenue"]),
            "pts": int(r["points_earned"]), "mtd": int(_get(r, "m_mtd_noshows", 0)),
            "rs": reasons, "score": int(score[i]), "st": "new"})
    with open(out, "w") as f:
        json.dump(alerts, f)
    print(f"wrote {len(alerts):,} alerts (score >= {min_score}) to {out}")
    return alerts