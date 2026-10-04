"""
Shared code for the notebooks: loading data, running the SQL, building
recommendations, the Consumer Duty guardrails and the evaluation metrics.
"""

import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from surprise import Dataset, Reader

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SQL = ROOT / "sql"
TOP_N = 5
RELEVANT_SCORE = 4          # engagement 4-5 (activated / used repeatedly) counts as a "good" recommendation


# ------------------------------------------------------------------ data + SQL
def load_tables():
    customers = pd.read_csv(DATA / "customers.csv")
    catalogue = pd.read_csv(DATA / "catalogue.csv")
    engagement = pd.read_csv(DATA / "engagement.csv")
    return customers, catalogue, engagement


def make_database(customers, catalogue, engagement):
    """Load the three tables into an in-memory SQLite database and create the SQL views."""
    con = sqlite3.connect(":memory:")
    customers.to_sql("customers", con, index=False)
    catalogue.to_sql("catalogue", con, index=False)
    engagement.to_sql("engagement", con, index=False)
    con.executescript((SQL / "01_vulnerability_flags.sql").read_text())
    return con


def run_sql_file(con, name):
    return pd.read_sql((SQL / name).read_text(), con)


def get_flags(con):
    return pd.read_sql("SELECT * FROM customer_flags", con)


# ------------------------------------------------------------------ Surprise helpers
def to_surprise(engagement):
    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(engagement[["customer_id", "item_id", "engagement_score"]], reader)


def rank_unseen(algo, trainset, all_items, users=None):
    """For each customer, predict a score for every item they have NOT seen, best first."""
    users = users if users is not None else [trainset.to_raw_uid(u) for u in trainset.all_users()]
    ranked = {}
    for uid in users:
        try:
            seen = {trainset.to_raw_iid(i) for i, _ in trainset.ur[trainset.to_inner_uid(uid)]}
        except ValueError:
            seen = set()
        preds = [(iid, algo.predict(uid, iid).est) for iid in all_items if iid not in seen]
        ranked[uid] = sorted(preds, key=lambda x: x[1], reverse=True)
    return ranked


def top_n(ranked, n=TOP_N):
    return {u: [iid for iid, _ in items[:n]] for u, items in ranked.items()}


# ------------------------------------------------------------------ Consumer Duty guardrails
def passes_affordability(flag_row):
    """Very simple stand-in for a real affordability / eligibility check."""
    return (flag_row["financial_difficulty_flag"] == 0
            and flag_row["income_band"] in ("mid", "high")
            and flag_row["avg_month_end_balance_gbp"] >= 500)


def apply_guardrails(ranked, catalogue, flags, n=TOP_N):
    """
    Re-rank each customer's list using four rules:

    R1 Avoid foreseeable harm: customers flagged as in financial difficulty are never shown
       restricted items (credit, gambling offers, crypto, stocks & shares).
    R2 Products & services: credit and risky investments are never pushed just because
       "similar customers liked them". They only appear if a basic affordability check passes,
       and at most ONE such item per list.
    R3 Consumer support: flagged customers always get at least one supportive feature
       (budgeting, savings, debt advice, gambling block...).
    R4 Consumer understanding: every recommendation carries a plain-English reason.
    """
    cat = catalogue.set_index("item_id")
    flag_lookup = flags.set_index("customer_id")
    out = {}
    for uid, items in ranked.items():
        f = flag_lookup.loc[uid]
        flagged = f["financial_difficulty_flag"] == 1
        affordable = passes_affordability(f)
        chosen, risky_used = [], 0
        for iid, score in items:
            row = cat.loc[iid]
            risky = row["item_type"] in ("credit", "investment") and row["risk_level"] != "none"
            if flagged and row["restricted_if_vulnerable"] == 1:          # R1
                continue
            if risky and (not affordable or risky_used >= 1):              # R2
                continue
            risky_used += risky
            chosen.append((iid, score))
            if len(chosen) == n:
                break
        if flagged and not any(cat.loc[i, "supportive"] == 1 for i, _ in chosen):   # R3
            support = [(i, s) for i, s in items if cat.loc[i, "supportive"] == 1]
            if support:
                chosen = chosen[: n - 1] + [support[0]]
        out[uid] = [(iid, score, reason(iid, cat, flagged, affordable)) for iid, score in chosen]
    return out


def reason(iid, cat, flagged, affordable):                                 # R4
    row = cat.loc[iid]
    if row["item_type"] == "offer":
        return "Cashback where customers with similar spending habits shop."
    if row["item_type"] == "investment" and row["risk_level"] == "none":
        return "A simple way to build savings, protected up to the FSCS limit."
    if row["supportive"] == 1 and flagged:
        return "A free tool that can help you stay on top of your money."
    if row["item_type"] in ("credit", "investment") and row["risk_level"] != "none":
        return ("You passed our basic eligibility check. Read the full costs and risks before "
                "applying - this is not personal advice.")
    return "Customers who use the app like you do find this useful."


# ------------------------------------------------------------------ evaluation
def precision_recall_at_k(recs, testset, k=TOP_N, threshold=RELEVANT_SCORE):
    """recs: {uid: [iid,...]}. Relevant = items in the test set the customer scored >= threshold."""
    relevant = defaultdict(set)
    for uid, iid, r in testset:
        if r >= threshold:
            relevant[uid].add(iid)
    precisions, recalls = [], []
    for uid, rel in relevant.items():
        rec = recs.get(uid, [])[:k]
        hits = len(set(rec) & rel)
        precisions.append(hits / k)
        recalls.append(hits / len(rel))
    return float(np.mean(precisions)), float(np.mean(recalls))


def harm_report(recs, catalogue, flags):
    """How often do recommendations put risky products in front of people in difficulty?"""
    cat = catalogue.set_index("item_id")
    f = flags.set_index("customer_id")
    rows = []
    for uid, items in recs.items():
        for iid in items:
            rows.append({"customer_id": uid, "item_id": iid,
                         "flagged": f.loc[uid, "financial_difficulty_flag"],
                         "affordable": passes_affordability(f.loc[uid]),
                         "restricted": cat.loc[iid, "restricted_if_vulnerable"],
                         "risky_product": int(cat.loc[iid, "item_type"] in ("credit", "investment")
                                              and cat.loc[iid, "risk_level"] != "none"),
                         "supportive": cat.loc[iid, "supportive"]})
    r = pd.DataFrame(rows)
    flagged = r[r["flagged"] == 1]
    per_cust = flagged.groupby("customer_id")
    return {
        "% of flagged customers shown a restricted item": 100 * (per_cust["restricted"].max().mean()),
        "% of flagged customers' slots that are restricted": 100 * flagged["restricted"].mean(),
        "% of flagged customers shown a supportive feature": 100 * (per_cust["supportive"].max().mean()),
        "% of all customers shown credit/risky investment without passing affordability":
            100 * r[(r["risky_product"] == 1) & (~r["affordable"])]["customer_id"].nunique() / r["customer_id"].nunique(),
    }
