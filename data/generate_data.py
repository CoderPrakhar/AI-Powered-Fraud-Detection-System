"""
generate_data.py
-----------------
Generates a realistic SYNTHETIC credit-card transaction dataset for the
fraud detection project. Real fraud datasets (e.g. from banks) are private,
so we simulate one with the same statistical patterns real datasets have:

- Fraud transactions tend to have unusually high amounts
- Fraud transactions tend to happen far from the cardholder's home
- Fraud transactions tend to happen far from the location of the last transaction
- Fraud is more common in online orders, and less common when a chip or PIN was used
- Fraud is rare overall (highly imbalanced, like in real life ~1-3%)

Run:  python generate_data.py
Output: transactions.csv (in the same folder)
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_SAMPLES = 20000
FRAUD_RATE = 0.03  # ~3% of transactions are fraudulent (realistic imbalance)

n_fraud = int(N_SAMPLES * FRAUD_RATE)
n_legit = N_SAMPLES - n_fraud


def generate_legit(n):
    return pd.DataFrame({
        "amount": np.round(np.random.gamma(shape=2.0, scale=40, size=n), 2),
        "distance_from_home": np.round(np.abs(np.random.normal(5, 8, n)), 2),
        "distance_from_last_transaction": np.round(np.abs(np.random.normal(2, 4, n)), 2),
        "ratio_to_median_purchase": np.round(np.abs(np.random.normal(1, 0.5, n)), 2),
        "repeat_retailer": np.random.choice([1, 0], n, p=[0.85, 0.15]),
        "used_chip": np.random.choice([1, 0], n, p=[0.7, 0.3]),
        "used_pin": np.random.choice([1, 0], n, p=[0.6, 0.4]),
        "online_order": np.random.choice([1, 0], n, p=[0.3, 0.7]),
        "hour_of_day": np.random.normal(14, 5, n).clip(0, 23).astype(int),
        "fraud": 0,
    })


def generate_fraud(n):
    return pd.DataFrame({
        "amount": np.round(np.random.gamma(shape=3.0, scale=150, size=n), 2),
        "distance_from_home": np.round(np.abs(np.random.normal(80, 60, n)), 2),
        "distance_from_last_transaction": np.round(np.abs(np.random.normal(60, 50, n)), 2),
        "ratio_to_median_purchase": np.round(np.abs(np.random.normal(5, 3, n)), 2),
        "repeat_retailer": np.random.choice([1, 0], n, p=[0.2, 0.8]),
        "used_chip": np.random.choice([1, 0], n, p=[0.15, 0.85]),
        "used_pin": np.random.choice([1, 0], n, p=[0.1, 0.9]),
        "online_order": np.random.choice([1, 0], n, p=[0.85, 0.15]),
        "hour_of_day": np.random.choice(
            range(24), n, p=_night_heavy_probs()
        ),
        "fraud": 1,
    })


def _night_heavy_probs():
    # Fraud skews toward late-night / early-morning hours
    weights = np.array([3 if (h <= 5 or h >= 22) else 1 for h in range(24)], dtype=float)
    return weights / weights.sum()


df = pd.concat([generate_legit(n_legit), generate_fraud(n_fraud)], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
df.insert(0, "transaction_id", [f"TXN{100000+i}" for i in range(len(df))])

out_path = "transactions.csv"
df.to_csv(out_path, index=False)

print(f"Generated {len(df)} transactions -> {out_path}")
print(f"Fraud cases: {df['fraud'].sum()} ({df['fraud'].mean()*100:.2f}%)")
print(df.head())
