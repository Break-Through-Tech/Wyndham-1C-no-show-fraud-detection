"""Synthetic alerts with the same schema export_scores.py writes. Used only when no real data file exists."""
from datetime import date, timedelta
import numpy as np

SITES = [("00003", "Beijing", "CN"), ("00117", "Shanghai", "CN"), ("00241", "Shenzhen", "CN"),
         ("00388", "Chengdu", "CN"), ("01022", "Orlando", "US"), ("01045", "Las Vegas", "US"),
         ("01290", "Miami", "US"), ("01377", "Honolulu", "US"), ("01411", "Atlanta", "US"),
         ("01508", "Dallas", "US"), ("01632", "Toronto", "CA"), ("01775", "Phoenix", "US")]


def _reasons(a, m):
    s = "" if a["lead"] == 1 else "s"
    return [("Last-minute booking", f"Booked {a['lead']} day{s} before check-in (typical: 42)"),
            ("New account", f"Account was {m['age']} days old at first booking (typical: 2,600)"),
            ("High room rate", f"Rate of ${a['rev']:,} is {a['rev'] / 181:.1f}x the typical no-show stay"),
            ("Shared phone number", f"{m['share']} member accounts share this phone prefix"),
            ("Repeat no-shows", f"{a['mtd']} no-shows this month (typical: 0.1)"),
            ("Email pattern", "Email follows the first-initial + last-name pattern seen in confirmed fraud"),
            ("Same-day cancel", "Cancelled the same day it was booked"),
            ("Digital booking", "Booked on the website or app, with no card validation")]


def make_demo(seed=11, days=31, end=date(2026, 8, 19)):
    rng = np.random.default_rng(seed)
    chars = list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")
    members = [{"id": "".join(rng.choice(chars, 11)), "age": int(rng.integers(1, 20)),
                "share": int(rng.integers(2, 7)), "src": str(rng.choice(["Website", "Mobile app"]))}
               for _ in range(70)]
    out = []
    for d in range(days):
        day = end - timedelta(days=days - 1 - d)
        n = int(rng.integers(16, 41)) + (30 if d == 8 else 0) + (8 if d < 12 else 0)
        for _ in range(n):
            m = members[int(rng.integers(len(members)))]
            u = rng.random()
            target = int(rng.integers(75, 99) if u < .22 else rng.integers(50, 75) if u < .6 else rng.integers(30, 50))
            crit = target >= 75
            site = SITES[int(rng.integers(len(SITES)))]
            a = {"id": f"HTL26{int(rng.integers(0, 10_000_000)):07d}", "date": day.isoformat(), "m": m,
                 "site": list(site), "reg": "APAC" if site[2] == "CN" else "NAMER",
                 "los": int(rng.choice([1, 1, 2, 2, 3])),
                 "lead": int(rng.integers(0, 2) if crit else rng.integers(0, 10)),
                 "rev": int(rng.integers(520, 1401) if crit else rng.integers(160, 761)),
                 "mtd": int(rng.integers(1, 7)) + (2 if crit else 0)}
            a["pts"] = a["rev"] * 10
            cand = _reasons(a, m)
            idx = rng.permutation(len(cand))[:int(rng.integers(3, 7))]
            w = .3 + rng.random(len(idx))
            c = np.maximum(2, np.round(w / w.sum() * target)).astype(int)
            a["rs"] = sorted([{"n": cand[i][0], "t": cand[i][1], "c": int(ci)} for i, ci in zip(idx, c)],
                             key=lambda r: -r["c"])
            a["score"] = sum(r["c"] for r in a["rs"])
            if day < end - timedelta(days=1):
                p = .85 if crit else .55 if target >= 50 else .25
                a["st"] = str(rng.choice(["confirmed", "confirmed", "escalated"])) if rng.random() < p else "dismissed"
            else:
                a["st"] = "new"
            out.append(a)
    return out