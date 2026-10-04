"""
Generate synthetic data for a UK fintech app's recommendation system.

Three tables are created:
    customers.csv     - one row per customer, incl. financial-resilience signals
    catalogue.csv     - everything the app could recommend (offers, features, credit, investments)
    engagement.csv    - how each customer responded to items they were shown (score 1-5)

Engagement score:
    1 = dismissed   2 = viewed   3 = clicked   4 = activated / applied   5 = used repeatedly

The data is made up. It is designed to contain one uncomfortable, realistic pattern:
customers in financial difficulty ENGAGE MORE with high-cost credit, gambling offers
and crypto - and then have worse outcomes. A recommender that only chases engagement
will therefore push harmful products to the people most at risk.

Run:  python data/generate_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
N_CUSTOMERS = 6_000
OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(SEED)

# --------------------------------------------------------------------------- catalogue
# item_id, name, item_type, risk_level, restricted_if_vulnerable, supportive, taste profile
# taste profile = how much each hidden "taste" likes the item:
#   [saver, everyday_spender, traveller, gamer_entertainment]
CATALOGUE = [
    ("O01", "Cashback: supermarket",          "offer",      "none",   0, 0, [0.4, 1.0, 0.0, 0.0]),
    ("O02", "Cashback: coffee shops",         "offer",      "none",   0, 0, [0.0, 1.0, 0.2, 0.1]),
    ("O03", "Cashback: restaurants",          "offer",      "none",   0, 0, [-.2, 0.9, 0.5, 0.2]),
    ("O04", "Cashback: fuel",                 "offer",      "none",   0, 0, [0.2, 0.7, 0.3, 0.0]),
    ("O05", "Cashback: flights & hotels",     "offer",      "none",   0, 0, [0.0, 0.1, 1.2, 0.0]),
    ("O06", "Cashback: fashion",              "offer",      "none",   0, 0, [-.3, 0.8, 0.3, 0.2]),
    ("O07", "Cashback: electronics",          "offer",      "none",   0, 0, [-.2, 0.3, 0.1, 0.9]),
    ("O08", "Cashback: streaming",            "offer",      "none",   0, 0, [0.0, 0.3, 0.0, 1.0]),
    ("O09", "Cashback: cinema",               "offer",      "none",   0, 0, [0.0, 0.4, 0.1, 0.8]),
    ("O10", "Cashback: gym",                  "offer",      "none",   0, 0, [0.4, 0.5, 0.1, 0.0]),
    ("O11", "Cashback: video games",          "offer",      "none",   0, 0, [-.2, 0.1, 0.0, 1.2]),
    ("O12", "Free bet: sports betting",       "offer",      "high",   1, 0, [-.5, 0.0, 0.0, 0.9]),
    ("O13", "Bonus spins: online casino",     "offer",      "high",   1, 0, [-.6, 0.0, 0.0, 0.8]),
    ("F01", "Savings pots",                   "feature",    "none",   0, 1, [1.2, 0.2, 0.3, 0.0]),
    ("F02", "Round-ups",                      "feature",    "none",   0, 1, [1.0, 0.4, 0.0, 0.0]),
    ("F03", "Budgeting tool",                 "feature",    "none",   0, 1, [0.9, 0.3, 0.0, 0.0]),
    ("F04", "Bill splitting",                 "feature",    "none",   0, 0, [0.0, 0.8, 0.5, 0.3]),
    ("F05", "Salary sorter",                  "feature",    "none",   0, 1, [1.0, 0.2, 0.0, 0.0]),
    ("F06", "Spending insights",              "feature",    "none",   0, 1, [0.8, 0.5, 0.0, 0.0]),
    ("F07", "Credit score checker",           "feature",    "none",   0, 1, [0.5, 0.3, 0.0, 0.2]),
    ("F08", "Cancel unused subscriptions",    "feature",    "none",   0, 1, [0.7, 0.2, 0.0, 0.4]),
    ("F09", "Free debt advice (MoneyHelper)", "feature",    "none",   0, 1, [0.3, 0.0, 0.0, 0.0]),
    ("F10", "Travel insurance add-on",        "feature",    "none",   0, 0, [0.1, 0.0, 1.1, 0.0]),
    ("F11", "Gambling block",                 "feature",    "none",   0, 1, [0.4, 0.0, 0.0, 0.2]),
    ("C01", "Credit card",                    "credit",     "medium", 1, 0, [-.2, 0.6, 0.5, 0.2]),
    ("C02", "Personal loan",                  "credit",     "medium", 1, 0, [-.2, 0.3, 0.2, 0.2]),
    ("C03", "Overdraft limit increase",       "credit",     "high",   1, 0, [-.5, 0.5, 0.0, 0.3]),
    ("C04", "Buy now, pay later",             "credit",     "high",   1, 0, [-.4, 0.7, 0.1, 0.5]),
    ("I01", "Cash ISA",                       "investment", "none",   0, 1, [1.1, 0.0, 0.0, 0.0]),
    ("I02", "Stocks & shares ISA",            "investment", "medium", 1, 0, [0.9, 0.0, 0.2, 0.1]),
    ("I03", "Crypto trading",                 "investment", "high",   1, 0, [-.3, 0.0, 0.1, 1.0]),
]

# how much extra a customer in financial difficulty is drawn to each item (the "trap")
DIFFICULTY_PULL = {"C03": 1.6, "C04": 1.4, "C02": 1.0, "C01": 0.8, "O12": 1.3, "O13": 1.2,
                   "I03": 0.9, "F01": -0.6, "F02": -0.4, "I01": -0.8, "I02": -0.6, "O05": -0.6}


def make_catalogue():
    cat = pd.DataFrame(CATALOGUE, columns=["item_id", "item_name", "item_type", "risk_level",
                                           "restricted_if_vulnerable", "supportive", "taste"])
    return cat


def make_customers():
    n = N_CUSTOMERS
    in_difficulty = rng.random(n) < 0.16          # hidden truth, NOT given to the model
    income_band = np.where(in_difficulty,
                           rng.choice(["low", "mid", "high"], n, p=[0.6, 0.33, 0.07]),
                           rng.choice(["low", "mid", "high"], n, p=[0.25, 0.5, 0.25]))
    taste = rng.dirichlet([0.7, 0.7, 0.7, 0.7], size=n)   # each customer's mix of tastes

    df = pd.DataFrame({
        "customer_id": [f"U{i:05d}" for i in range(n)],
        "age_band": rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], n, p=[.2, .32, .22, .14, .12]),
        "income_band": income_band,
        "region": rng.choice(["London", "South East", "North West", "West Midlands", "Scotland",
                              "Yorkshire", "East", "South West", "Wales", "North East",
                              "East Midlands", "Northern Ireland"], n),
        "account_age_months": np.clip(rng.gamma(2, 12, n).round(), 1, 120).astype(int),
        # financial-resilience signals a bank can see from its own data
        "overdraft_days_90d": np.where(in_difficulty, rng.integers(10, 90, n),
                                       rng.choice([0, 2, 5, 12, 35], n, p=[0.62, 0.14, 0.12, 0.08, 0.04])),
        "missed_payments_6m": np.where(in_difficulty, rng.poisson(1.3, n), rng.poisson(0.05, n)),
        "gambling_spend_share": np.round(np.where(in_difficulty,
                                                  rng.beta(1.2, 6, n), rng.beta(0.3, 40, n)), 3),
        "high_cost_credit_payments_90d": np.where(in_difficulty, rng.poisson(2.0, n), rng.poisson(0.1, n)),
        "avg_month_end_balance_gbp": np.round(np.where(in_difficulty, rng.normal(80, 150, n),
                                                       rng.lognormal(6.6, 1.0, n)), 2),
    })
    return df, taste, in_difficulty


def make_engagement(customers, taste, in_difficulty, catalogue):
    item_taste = np.vstack(catalogue["taste"].values)                    # items x 4
    pull = catalogue["item_id"].map(DIFFICULTY_PULL).fillna(0).values    # items
    popularity = rng.normal(0, 0.3, len(catalogue))
    rows = []
    for u in range(len(customers)):
        # affinity of this customer for every item
        aff = taste[u] @ item_taste.T * 2.2 + popularity + in_difficulty[u] * pull
        # which items were they shown? (apps show you things you're more likely to like)
        n_seen = rng.integers(6, 15)
        p = np.exp(aff * 0.8); p /= p.sum()
        seen = rng.choice(len(catalogue), size=n_seen, replace=False, p=p)
        for i in seen:
            latent = aff[i] + rng.normal(0, 0.55)
            score = int(np.clip(np.round(1.8 + latent * 1.3), 1, 5))
            iid = catalogue.at[i, "item_id"]
            # what happened afterwards - only meaningful for credit products the customer activated
            later_missed = np.nan
            if catalogue.at[i, "item_type"] == "credit" and score >= 4:
                later_missed = int(rng.random() < (0.38 if in_difficulty[u] else 0.04))
            rows.append((customers.at[u, "customer_id"], iid, score, later_missed))
    eng = pd.DataFrame(rows, columns=["customer_id", "item_id", "engagement_score",
                                      "missed_payment_within_6m"])
    eng["missed_payment_within_6m"] = eng["missed_payment_within_6m"].astype("Int64")
    return eng


def main():
    catalogue = make_catalogue()
    customers, taste, in_difficulty = make_customers()
    engagement = make_engagement(customers, taste, in_difficulty, catalogue)
    catalogue.drop(columns="taste").to_csv(OUT / "catalogue.csv", index=False)
    customers.to_csv(OUT / "customers.csv", index=False)
    engagement.to_csv(OUT / "engagement.csv", index=False)
    print(f"customers: {len(customers):,} | items: {len(catalogue)} | engagement rows: {len(engagement):,}")
    print(engagement["engagement_score"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
