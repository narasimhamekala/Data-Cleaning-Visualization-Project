"""Generates a realistic, intentionally messy sales dataset (raw_sales_data.csv).
Replace data/raw_sales_data.csv with your own dataset if your mentor provided one."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 1200
cats = {"Electronics": (150, 900), "Clothing": (15, 120), "Home & Kitchen": (20, 250),
        "Books": (5, 40), "Sports": (20, 300)}
regions = ["North", "South", "East", "West"]
payments = ["Credit Card", "UPI", "Cash", "Net Banking"]
cat = rng.choice(list(cats), n, p=[.2, .3, .2, .15, .15])
price = np.array([rng.uniform(*cats[c]) for c in cat]).round(2)
df = pd.DataFrame({
    "order_id": np.arange(1001, 1001 + n),
    "order_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
    "customer_age": rng.normal(35, 10, n).round().astype(int),
    "region": rng.choice(regions, n),
    "category": cat,
    "unit_price": price,
    "quantity": rng.integers(1, 6, n),
    "discount_pct": rng.choice([0, 5, 10, 15, 20], n),
    "payment_method": rng.choice(payments, n),
    "rating": rng.choice([1, 2, 3, 4, 5], n, p=[.05, .1, .2, .35, .3]).astype(float),
})
# ---- inject mess ----
df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")
idx = rng.choice(n, 150, replace=False); df.loc[idx, "order_date"] = pd.to_datetime(df.loc[idx, "order_date"]).dt.strftime("%d/%m/%Y")
for col, k in [("customer_age", 70), ("rating", 110), ("region", 50), ("unit_price", 40), ("payment_method", 45)]:
    df.loc[rng.choice(n, k, replace=False), col] = np.nan
df.loc[rng.choice(n, 10, replace=False), "customer_age"] = rng.choice([150, 200, -5, 999], 10)   # impossible ages
df.loc[rng.choice(n, 12, replace=False), "unit_price"] = rng.uniform(5000, 20000, 12).round(2)   # price outliers
df.loc[rng.choice(n, 8, replace=False), "quantity"] = rng.choice([-1, -3, 50, 100], 8)           # bad quantities
df["region"] = df["region"].apply(lambda x: x if pd.isna(x) else rng.choice([x, x.lower(), x.upper(), " " + x + " "], p=[.7, .1, .1, .1]))
df["category"] = df["category"].apply(lambda x: rng.choice([x, x.lower(), x.upper(), x + " "], p=[.7, .1, .1, .1]))
df["unit_price"] = df["unit_price"].apply(lambda x: f"${x}" if (not pd.isna(x) and rng.random() < .1) else x)
df = pd.concat([df, df.sample(60, random_state=1)], ignore_index=True)                           # duplicates
df = df.sample(frac=1, random_state=3).reset_index(drop=True)
df.to_csv("data/raw_sales_data.csv", index=False)
print("Raw data saved:", df.shape)
