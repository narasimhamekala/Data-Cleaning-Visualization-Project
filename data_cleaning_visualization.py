"""
Task 1: Data Cleaning & Visualization Project
Pipeline: load -> inspect -> clean (duplicates, formats, missing, outliers) -> EDA -> visualize -> dashboard
Run:  python data_cleaning_visualization.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="viridis")
RAW, CLEAN, CHARTS = "data/raw_sales_data.csv", "data/cleaned_sales_data.csv", "charts"
os.makedirs(CHARTS, exist_ok=True)
log = []  # cleaning log for the report

def note(msg):
    print(msg); log.append(msg)

# ------------------------------------------------------------------ 1. LOAD & INSPECT
df = pd.read_csv(RAW)
raw_shape = df.shape
note(f"Raw dataset shape: {raw_shape}")
print(df.head(), "\n"); print(df.dtypes, "\n")
missing_before = df.isna().sum()
print("Missing values BEFORE cleaning:\n", missing_before[missing_before > 0], "\n")

# ------------------------------------------------------------------ 2. REMOVE DUPLICATES
dups = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
note(f"Duplicates removed: {dups}")

# ------------------------------------------------------------------ 3. FIX FORMATS / INCONSISTENCIES
# text columns: strip spaces + consistent Title Case
for c in ["region", "category", "payment_method"]:
    df[c] = df[c].astype("string").str.strip().str.title()
df["payment_method"] = df["payment_method"].replace({"Upi": "UPI"})  # fix acronym broken by title-casing
# price: remove "$" and convert to number
df["unit_price"] = pd.to_numeric(df["unit_price"].astype(str).str.replace("$", "", regex=False), errors="coerce")
# dates: two mixed formats (YYYY-MM-DD and DD/MM/YYYY)
d1 = pd.to_datetime(df["order_date"], format="%Y-%m-%d", errors="coerce")
d2 = pd.to_datetime(df["order_date"], format="%d/%m/%Y", errors="coerce")
df["order_date"] = d1.fillna(d2)
note(f"Unparseable dates after fixing: {df['order_date'].isna().sum()}")

# ------------------------------------------------------------------ 4. INVALID VALUES -> NaN (so they are handled with missing values)
bad_age = ~df["customer_age"].between(10, 100) & df["customer_age"].notna()
bad_qty = ~df["quantity"].between(1, 10)
note(f"Impossible ages set to NaN: {bad_age.sum()} | invalid quantities set to NaN: {bad_qty.sum()}")
df.loc[bad_age, "customer_age"] = np.nan
df.loc[bad_qty, "quantity"] = np.nan

# ------------------------------------------------------------------ 5. OUTLIERS (IQR method, per category for price)
def iqr_bounds(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return q1 - 1.5 * i, q3 + 1.5 * i

# boxplot BEFORE outlier treatment
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(data=df, x="category", y="unit_price", ax=ax[0]); ax[0].set_title("Unit price by category - BEFORE outlier treatment")
ax[0].tick_params(axis="x", rotation=30)

out_total = 0
for cat, g in df.groupby("category"):
    lo, hi = iqr_bounds(g["unit_price"].dropna())
    mask = (df["category"] == cat) & ((df["unit_price"] < lo) | (df["unit_price"] > hi))
    out_total += mask.sum()
    df.loc[mask, "unit_price"] = df.loc[mask, "unit_price"].clip(lo, hi)  # winsorize (cap)
note(f"Price outliers capped (IQR, per category): {out_total}")
sns.boxplot(data=df, x="category", y="unit_price", ax=ax[1]); ax[1].set_title("AFTER outlier treatment")
ax[1].tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig(f"{CHARTS}/01_outliers_before_after.png", dpi=150); plt.close()

# ------------------------------------------------------------------ 6. MISSING VALUES
# categorical -> mode ; numeric -> median (grouped by category for price)
for c in ["region", "payment_method"]:
    df[c] = df[c].fillna(df[c].mode()[0])
df["customer_age"] = df["customer_age"].fillna(df["customer_age"].median())
df["quantity"] = df["quantity"].fillna(df["quantity"].median())
df["rating"] = df["rating"].fillna(df["rating"].median())
df["unit_price"] = df["unit_price"].fillna(df.groupby("category")["unit_price"].transform("median"))
df["customer_age"] = df["customer_age"].astype(int); df["quantity"] = df["quantity"].astype(int)
assert df.isna().sum().sum() == 0
note(f"Missing values filled: {int(missing_before.sum())} originally missing + invalid values -> remaining NaN = 0")

# ------------------------------------------------------------------ 7. FEATURE ENGINEERING
df["revenue"] = (df["unit_price"] * df["quantity"] * (1 - df["discount_pct"] / 100)).round(2)
df["month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
df["age_group"] = pd.cut(df["customer_age"], [0, 24, 34, 44, 54, 100], labels=["<25", "25-34", "35-44", "45-54", "55+"])
df.to_csv(CLEAN, index=False)
note(f"Cleaned dataset shape: {df.shape} -> saved to {CLEAN}")

# ------------------------------------------------------------------ 8. VISUALIZATIONS
# 8.1 missing values before
fig, ax = plt.subplots(figsize=(7, 4))
mb = missing_before[missing_before > 0].sort_values()
ax.barh(mb.index, mb.values, color="#e76f51"); ax.set_title("Missing values per column (raw data)"); ax.set_xlabel("Count")
plt.tight_layout(); plt.savefig(f"{CHARTS}/02_missing_values_raw.png", dpi=150); plt.close()

# 8.2 monthly revenue trend
monthly = df.groupby("month")["revenue"].sum()
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(monthly.index, monthly.values, marker="o", color="#2a9d8f"); ax.set_title("Monthly revenue trend"); ax.set_ylabel("Revenue")
plt.xticks(rotation=30); plt.tight_layout(); plt.savefig(f"{CHARTS}/03_monthly_revenue.png", dpi=150); plt.close()

# 8.3 revenue by category
cat_rev = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=cat_rev.index, y=cat_rev.values, ax=ax); ax.set_title("Total revenue by category"); ax.set_ylabel("Revenue")
plt.tight_layout(); plt.savefig(f"{CHARTS}/04_revenue_by_category.png", dpi=150); plt.close()

# 8.4 region x category heatmap
pivot = df.pivot_table(index="region", columns="category", values="revenue", aggfunc="sum")
fig, ax = plt.subplots(figsize=(9, 4))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax); ax.set_title("Revenue: Region vs Category")
plt.tight_layout(); plt.savefig(f"{CHARTS}/05_region_category_heatmap.png", dpi=150); plt.close()

# 8.5 customer age distribution + rating distribution
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["customer_age"], bins=20, kde=True, ax=ax[0]); ax[0].set_title("Customer age distribution")
sns.countplot(x="rating", data=df, ax=ax[1]); ax[1].set_title("Rating distribution")
plt.tight_layout(); plt.savefig(f"{CHARTS}/06_age_rating_distribution.png", dpi=150); plt.close()

# 8.6 payment method share
fig, ax = plt.subplots(figsize=(5, 5))
df["payment_method"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax, startangle=90); ax.set_ylabel("")
ax.set_title("Payment method share"); plt.tight_layout(); plt.savefig(f"{CHARTS}/07_payment_share.png", dpi=150); plt.close()

# 8.7 correlation heatmap
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df[["customer_age", "unit_price", "quantity", "discount_pct", "rating", "revenue"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlation matrix"); plt.tight_layout(); plt.savefig(f"{CHARTS}/08_correlation.png", dpi=150); plt.close()

# ------------------------------------------------------------------ 9. DASHBOARD
fig = plt.figure(figsize=(16, 10)); gs = fig.add_gridspec(3, 4)
fig.suptitle("Sales Dashboard - Cleaned Data", fontsize=20, fontweight="bold")
kpis = [("Total Revenue", f"{df['revenue'].sum():,.0f}"), ("Orders", f"{len(df):,}"),
        ("Avg Order Value", f"{df['revenue'].mean():,.0f}"), ("Avg Rating", f"{df['rating'].mean():.2f}")]
for i, (k, v) in enumerate(kpis):
    a = fig.add_subplot(gs[0, i]); a.axis("off")
    a.text(.5, .62, v, ha="center", fontsize=26, fontweight="bold", color="#264653")
    a.text(.5, .22, k, ha="center", fontsize=13, color="gray")
a = fig.add_subplot(gs[1, :2]); a.plot(monthly.index, monthly.values, marker="o", color="#2a9d8f"); a.set_title("Monthly revenue"); a.tick_params(axis="x", rotation=30)
a = fig.add_subplot(gs[1, 2:]); sns.barplot(x=cat_rev.index, y=cat_rev.values, ax=a); a.set_title("Revenue by category"); a.tick_params(axis="x", rotation=20)
a = fig.add_subplot(gs[2, :2]); sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", ax=a, cbar=False); a.set_title("Region vs Category revenue"); a.tick_params(axis="y", rotation=0)
a = fig.add_subplot(gs[2, 2]); sns.histplot(df["customer_age"], bins=15, kde=True, ax=a); a.set_title("Customer age")
a = fig.add_subplot(gs[2, 3]); df["payment_method"].value_counts().plot.pie(autopct="%1.0f%%", ax=a, textprops={"fontsize": 8}); a.set_ylabel(""); a.set_title("Payment method")
plt.tight_layout(rect=[0, 0, 1, .95]); plt.savefig(f"{CHARTS}/00_dashboard.png", dpi=150); plt.close()

# ------------------------------------------------------------------ 10. KEY INSIGHTS (auto-generated from the data)
top_cat = cat_rev.index[0]; top_reg = df.groupby("region")["revenue"].sum().idxmax()
best_m = monthly.idxmax().strftime("%B %Y"); top_pay = df["payment_method"].value_counts().idxmax()
insights = [
    f"{top_cat} is the top revenue category ({cat_rev.iloc[0]:,.0f}, {cat_rev.iloc[0]/cat_rev.sum():.0%} of total).",
    f"{top_reg} is the best-performing region by revenue.",
    f"Peak sales month: {best_m}.",
    f"Most used payment method: {top_pay}.",
    f"Average customer age is {df['customer_age'].mean():.0f}; average rating is {df['rating'].mean():.2f}/5.",
]
print("\nKEY INSIGHTS"); [print(" -", i) for i in insights]
with open("data/cleaning_report.txt", "w") as f:
    f.write("CLEANING LOG\n" + "\n".join(f"- {l}" for l in log) + "\n\nKEY INSIGHTS\n" + "\n".join(f"- {i}" for i in insights))
