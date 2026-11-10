"""
Fraud Analysis Dashboard: Streamlit analyst dashboard.

    pip install -r requirements.txt
    streamlit run app.py

Data: reads dashboard_data.json (written by export_scores.py). If that file is missing it shows
synthetic demo data and says so. Point to another file with the DASHBOARD_DATA environment variable.
Analyst decisions are saved to analyst_decisions.csv next to this file.
"""
import datetime as dt
import json
import os

import altair as alt
import pandas as pd
import streamlit as st

from demo_data import make_demo

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.environ.get("DASHBOARD_DATA", os.path.join(HERE, "dashboard_data.json"))
DECISIONS = os.path.join(HERE, "analyst_decisions.csv")
TIERS = {"Critical": "#b3261e", "High": "#d98200", "Medium": "#2b84a8"}
STATUS = {"new": "Needs review", "confirmed": "Confirmed fraud", "escalated": "Escalated", "dismissed": "Dismissed"}

st.set_page_config(page_title="Fraud Monitor", page_icon="🛡️", layout="wide")


def tier(score):
    return "Critical" if score >= 75 else "High" if score >= 50 else "Medium"


@st.cache_data
def load(path):
    demo = not os.path.exists(path)
    alerts = make_demo() if demo else json.load(open(path))
    df = pd.DataFrame([{
        "id": a["id"], "date": a["date"], "reg": a["reg"], "los": a["los"], "pts": a["pts"],
        "score": a["score"], "st": a["st"], "member": a["m"]["id"], "age": a["m"]["age"],
        "site_id": a["site"][0], "geo": ", ".join(x for x in a["site"][1:] if x), "rs": a["rs"]} for a in alerts])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["tier"] = df["score"].map(tier)
    df["why"] = df["rs"].map(lambda r: " · ".join(x["n"] for x in r[:2]))
    return df, demo


def read_decisions():
    if not os.path.exists(DECISIONS):
        return {}
    d = pd.read_csv(DECISIONS).drop_duplicates("id", keep="last")
    return dict(zip(d["id"], d["status"]))


def decide(alert_id, status):
    row = pd.DataFrame([{"id": alert_id, "status": status, "time": dt.datetime.now().isoformat(timespec="seconds")}])
    row.to_csv(DECISIONS, mode="a", header=not os.path.exists(DECISIONS), index=False)
    st.toast(f"{alert_id}: {STATUS[status].lower()}")


def shift_day(step):
    days = sorted(DF["date"].unique())
    i = days.index(st.session_state["date"]) if st.session_state["date"] in days else len(days) - 1
    st.session_state["date"] = days[max(0, min(len(days) - 1, i + step))]


DF, IS_DEMO = load(DATA_PATH)
DF["status"] = DF["id"].map(read_decisions()).fillna(DF["st"])
DAYS = sorted(DF["date"].unique())
LABEL2DATE = {f"{d.month}/{d.day}": d for d in DAYS}

# A click on the chart sets the date. This runs before the date widget is created, so it can set its state.
try:
    picked = st.session_state["daychart"]["selection"]["pick"][0]["label"]
except (KeyError, IndexError, TypeError):
    picked = None
if picked and picked != st.session_state.get("_last_pick"):
    st.session_state["_last_pick"] = picked
    st.session_state["date"] = LABEL2DATE.get(picked, DAYS[-1])
st.session_state.setdefault("date", DAYS[-1])

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Filters")
    c1, c2 = st.columns(2)
    c1.button("‹ Previous day", on_click=shift_day, args=(-1,), width="stretch")
    c2.button("Next day ›", on_click=shift_day, args=(1,), width="stretch")
    sel_date = st.date_input("Date", key="date", min_value=DAYS[0], max_value=DAYS[-1])
    region = st.selectbox("Region", ["All"] + sorted(DF["reg"].unique()))
    min_score = st.slider("Minimum risk score", 0, 90, 30,
                          help="Raise it to shrink the analyst queue. Lower it to catch more.")
    tiers = st.multiselect("Risk tiers shown in the table", list(TIERS), default=list(TIERS))

st.title("Fraud Monitor Dashboard")
if IS_DEMO:
    st.warning("Showing synthetic demo data. Run export_scores.py to create dashboard_data.json "
               "and this page will use your model’s scores.")

base = DF[(DF["score"] >= min_score) & ((region == "All") | (DF["reg"] == region))]
day = base[base["date"] == sel_date]
per_day = base.groupby("date").size().reindex(DAYS, fill_value=0)

# ---------------------------------------------------------------- 30-day chart (click a bar to pick a day)
st.subheader("Past 30 days: flagged reservations per day")
daily = base.groupby(["date", "tier"]).size().reset_index(name="n")
daily["label"] = daily["date"].map(lambda d: f"{d.month}/{d.day}")
daily["picked"] = daily["date"] == sel_date
daily["rank"] = daily["tier"].map({t: i for i, t in enumerate(TIERS)})
pick = alt.selection_point(name="pick", fields=["label"])
chart = (alt.Chart(daily).mark_bar().encode(
    x=alt.X("label:N", sort=list(LABEL2DATE), title=None, axis=alt.Axis(labelAngle=0, labelOverlap="parity")),
    y=alt.Y("n:Q", title="Flagged reservations"),
    color=alt.Color("tier:N", scale=alt.Scale(domain=list(TIERS), range=list(TIERS.values())),
                    legend=alt.Legend(title=None, orient="top")),
    opacity=alt.condition("datum.picked", alt.value(1), alt.value(.5)),
    order=alt.Order("rank:Q", sort="ascending"),
    tooltip=["label", "tier", "n"]).add_params(pick).properties(height=240))
st.altair_chart(chart, width="stretch", on_select="rerun", key="daychart")
st.caption("Click a bar to inspect that day.")

# ---------------------------------------------------------------- counts
st.subheader(f"{'Today’s' if sel_date == DAYS[-1] else sel_date.strftime('%b %d')} counts")
prior = per_day[per_day.index < sel_date].tail(7)
delta = (len(day) / prior.mean() - 1) * 100 if len(prior) and prior.mean() else 0
reviewed = int((day["status"] != "new").sum())
k1, k2, k3, k4 = st.columns(4)
k1.metric("Potential fraudulent reservations", len(day), f"{delta:+.0f}% vs prior 7-day average", delta_color="inverse")
k2.metric("Potential fraudulent members", day["member"].nunique())
k3.metric("Loyalty points at risk", f"{int(day['pts'].sum()):,}")
k4.metric("Analyst progress", f"{reviewed} of {len(day)}")
k4.progress(reviewed / len(day) if len(day) else 0.0)

# ---------------------------------------------------------------- tier hit-rates (false-positive tradeoff)
done = base[base["status"] != "new"]
cols = st.columns(len(TIERS))
for col, t in zip(cols, TIERS):
    r = done[done["tier"] == t]
    rate = f"{(r['status'] != 'dismissed').mean():.0%} confirmed" if len(r) else "no reviews yet"
    col.caption(f"**{t}**: {int((day['tier'] == t).sum())} today · {rate} across all reviewed alerts")

# ---------------------------------------------------------------- table
st.subheader(f"Transactions for {sel_date.strftime('%a %b %d, %Y')}")
t1, t2 = st.columns([2, 3])
view = t1.radio("Group by", ["Reservation", "Member"], horizontal=True)
todo = t2.toggle("Only items that still need review")
rows = day[day["tier"].isin(tiers)]
if todo:
    rows = rows[rows["status"] == "new"]

if view == "Reservation":
    tbl = rows.sort_values("score", ascending=False).assign(
        **{"Confirmation #": lambda d: d["id"], "Site ID": lambda d: d["site_id"], "Site geo": lambda d: d["geo"],
           "LOS (nights)": lambda d: d["los"], "Member #": lambda d: d["member"], "Risk score": lambda d: d["score"],
           "Score explanation": lambda d: d["why"], "Status": lambda d: d["status"].map(STATUS)})
    shown = ["Confirmation #", "Site ID", "Site geo", "LOS (nights)", "Member #", "Risk score", "Score explanation", "Status"]
else:
    top = rows.sort_values("score", ascending=False)
    g = top.groupby("member", sort=False)
    tbl = pd.DataFrame({"id": g["id"].first(), "Member #": g["member"].first(), "Flagged bookings": g.size(),
                        "Risk score": g["score"].max(), "Points at risk": g["pts"].sum(),
                        "Score explanation": g["why"].first(),
                        "Status": g["status"].apply(lambda s: STATUS["new"] if (s == "new").any() else STATUS[s.iloc[0]])})
    tbl = tbl.sort_values("Risk score", ascending=False).reset_index(drop=True)
    shown = ["Member #", "Flagged bookings", "Risk score", "Points at risk", "Score explanation", "Status"]
tbl = tbl.reset_index(drop=True)

if tbl.empty:
    st.info("Nothing matches these filters. Lower the minimum risk score or add a tier back.")
    selected = []
else:
    ev = st.dataframe(tbl[shown], hide_index=True, width="stretch", on_select="rerun",
                      selection_mode="single-row", key="tbl",
                      column_config={"Risk score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d")})
    selected = ev.selection.rows
    st.caption(f"At a minimum risk of {min_score}, the queue averages {per_day.mean():.0f} alerts a day. "
               "Select a row to see why it was flagged.")
    st.download_button("Download this queue (CSV)", tbl[shown].to_csv(index=False), f"queue_{sel_date}.csv", "text/csv")

# ---------------------------------------------------------------- drill-down
if selected:
    a = DF[DF["id"] == tbl.iloc[selected[0]]["id"]].iloc[0]
    with st.container(border=True):
        h1, h2, h3 = st.columns([1, 1, 2])
        h1.metric("Risk score", int(a["score"]))
        h2.metric("Tier", a["tier"])
        h3.markdown(f"**{a['id']}**  \n{a['geo']} · site {a['site_id']} · {a['los']} night(s) · member `{a['member']}`  \n"
                    f"Status: **{STATUS[a['status']]}**")
        st.markdown("##### Why it was flagged")
        st.caption("Each bar is the risk points one signal added. Together they make the score.")
        biggest = max(r["c"] for r in a["rs"])
        for r in a["rs"]:
            st.markdown(f"**+{r['c']}**  {r['t']}")
            st.progress(r["c"] / biggest)
        hist = DF[DF["member"] == a["member"]].sort_values("date", ascending=False)
        st.markdown(f"##### Member `{a['member']}`")
        m1, m2, m3 = st.columns(3)
        m1.metric("Age at first booking", f"{int(a['age'])} days")
        m2.metric("Flagged bookings", len(hist))
        m3.metric("Points at risk", f"{int(hist['pts'].sum()):,}")
        st.dataframe(hist.assign(Date=hist["date"], Booking=hist["id"], Score=hist["score"],
                                 Status=hist["status"].map(STATUS))[["Date", "Booking", "Score", "Status"]].head(10),
                     hide_index=True, width="stretch")
        b1, b2, b3 = st.columns(3)
        b1.button("Confirm fraud", on_click=decide, args=(a["id"], "confirmed"), type="primary", width="stretch")
        b2.button("Escalate", on_click=decide, args=(a["id"], "escalated"), width="stretch")
        b3.button("Dismiss", on_click=decide, args=(a["id"], "dismissed"), width="stretch")