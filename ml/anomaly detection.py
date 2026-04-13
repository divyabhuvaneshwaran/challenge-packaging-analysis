"""
Challenge Packaging Industries
03 — Anomaly Detection
=======================
Detects unusual transactions in both Sales and Purchases
using customer/supplier-aware IQR method.

Why NOT Isolation Forest (old approach)?
  Isolation Forest flagged large Sundram transactions as anomalies
  simply because they were big — but those ARE Sundram's normal
  order size. Treating all customers as one pool is wrong.

Why customer-aware IQR?
  Each customer has their own normal transaction range.
  A Rs.1.5L transaction is normal for KMD Precision but
  would be a genuine anomaly for J.C Packaging.
  IQR per customer gives meaningful, explainable flags.

Input  : day_book_final.csv
Output : outputs/anomalies.csv
         outputs/anomaly_detection.png

Run    : python 03_anomaly_detection.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE    = r"C:\Users\divya\Documents\challenge-packaging-analysis\data\day_book_final.csv"
OUTPUT_DIR    = r"C:\Users\divya\Documents\challenge-packaging-analysis\outputs"
MIN_TXN_COUNT = 5      # minimum transactions needed for reliable IQR baseline
IQR_FACTOR    = 1.5    # standard IQR multiplier (1.5 = mild, 3.0 = extreme only)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── NAME CONSOLIDATION ────────────────────────────────────────────────────────
NAME_MAP = {
    "Sundram Fasteners Limited ( Madurai )": "SUNDRAM FASTENERS LIMITED",
    "M/S ASUS TECHNOLOGY PVT LTD"          : "ASUS Technology Pvt Ltd",
    "ASUS Technology Pvt. Ltd."            : "ASUS Technology Pvt Ltd",
    "Red Star Polymers Private Limited ( Mangadu)"       : "Red Star Polymers Pvt Ltd",
    "RED STAR POLYMERS PRIVATE LIMITED( Valasaravakkam)": "Red Star Polymers Pvt Ltd",
    "RED STAR POLYMERS PRIVATE LIMITED"                  : "Red Star Polymers Pvt Ltd",
    "Red Star Plastick Pvt Ltd"                          : "Red Star Plastick Pvt Ltd",
    "Redstar Plastick Private Ltd ( Warehouse )"         : "Red Star Plastick Pvt Ltd",
    "REDSTAR PLASTIC LIMITED (Valasaravakkam )"          : "Red Star Plastick Pvt Ltd",
    "Regenix Drugs Ltd ( Bharti Life )"                  : "Regenix Drugs Group",
    "Regenix Drugs Ltd (Queen )"                         : "Regenix Drugs Group",
    "Regenix Drugs Ltd"                                  : "Regenix Drugs Group",
    "REGENIX DRUGS MUR & MUR DIVISION"                   : "Regenix Drugs Group",
    "Regenix Super Speciality Laboratories Pvt Ltd"      : "Regenix Drugs Group",
    "M/S Regenix Drugs Ltd (B)"                          : "Regenix Drugs Group",
    "M/S Regenix Drugs ( Q)"                             : "Regenix Drugs Group",
}

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
df["Particulars"] = df["Particulars"].replace(NAME_MAP)

sales     = df[df["Txn_Category"] == "Sales"].copy()
purchases = df[df["Txn_Category"] == "Purchase"].copy()

# Remove zero-value transactions — they are credit adjustments not real txns
sales     = sales[sales["Debit_Amount"]   > 0]
purchases = purchases[purchases["Credit_Amount"] > 0]

print(f"  Sales transactions    : {len(sales)}")
print(f"  Purchase transactions : {len(purchases)}")


# ── IQR ANOMALY DETECTOR ─────────────────────────────────────────────────────
def detect_anomalies(df_in, entity_col, amount_col, entity_type="Customer"):
    """
    For each entity (customer/supplier) with >= MIN_TXN_COUNT transactions:
    - Calculate Q1, Q3, IQR
    - Flag transactions outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    - Calculate deviation % from entity mean
    - Classify as HIGH (above range) or LOW (below range)
    """
    anomalies  = []
    skipped    = []
    baselines  = []

    for entity, grp in df_in.groupby(entity_col):
        grp = grp.copy()

        if len(grp) < MIN_TXN_COUNT:
            skipped.append(entity)
            continue

        q1   = grp[amount_col].quantile(0.25)
        q3   = grp[amount_col].quantile(0.75)
        iqr  = q3 - q1
        mean = grp[amount_col].mean()
        low  = max(0, q1 - IQR_FACTOR * iqr)
        high = q3 + IQR_FACTOR * iqr

        baselines.append({
            entity_type : entity,
            "Txn_Count" : len(grp),
            "Mean"      : round(mean, 0),
            "Q1"        : round(q1, 0),
            "Q3"        : round(q3, 0),
            "IQR"       : round(iqr, 0),
            "Low_Bound" : round(low, 0),
            "High_Bound": round(high, 0),
        })

        flags = grp[(grp[amount_col] < low) | (grp[amount_col] > high)].copy()

        if len(flags) == 0:
            continue

        flags[entity_type]      = entity
        flags["Amount"]         = flags[amount_col].round(2)
        flags["Entity_Mean"]    = round(mean, 0)
        flags["Expected_Low"]   = round(low, 0)
        flags["Expected_High"]  = round(high, 0)
        flags["Deviation_%"]    = ((flags[amount_col] - mean) / mean * 100).round(1)
        flags["Flag_Type"]      = flags[amount_col].apply(
            lambda x: "HIGH — unusually large" if x > high else "LOW — unusually small"
        )
        anomalies.append(flags)

    return anomalies, skipped, pd.DataFrame(baselines)


# ── DETECT SALES ANOMALIES ────────────────────────────────────────────────────
print("\n── Sales anomaly detection ──────────────────────────────────")
s_anomalies, s_skipped, s_baselines = detect_anomalies(
    sales, "Particulars", "Debit_Amount", "Customer"
)

s_anom_df = pd.concat(s_anomalies)[
    ["Date", "Customer", "Vch_No", "Amount",
     "Entity_Mean", "Expected_Low", "Expected_High",
     "Deviation_%", "Flag_Type", "FY"]
].sort_values("Deviation_%", ascending=False).reset_index(drop=True)

print(f"  Customers analysed    : {len(s_baselines)}")
print(f"  Customers skipped     : {len(s_skipped)} (< {MIN_TXN_COUNT} transactions)")
print(f"  Sales anomalies found : {len(s_anom_df)}")


# ── DETECT PURCHASE ANOMALIES ─────────────────────────────────────────────────
print("\n── Purchase anomaly detection ───────────────────────────────")
p_anomalies, p_skipped, p_baselines = detect_anomalies(
    purchases, "Particulars", "Credit_Amount", "Supplier"
)

p_anom_df = pd.concat(p_anomalies)[
    ["Date", "Supplier", "Vch_No", "Amount",
     "Entity_Mean", "Expected_Low", "Expected_High",
     "Deviation_%", "Flag_Type", "FY"]
].sort_values("Deviation_%", ascending=False).reset_index(drop=True)

print(f"  Suppliers analysed    : {len(p_baselines)}")
print(f"  Suppliers skipped     : {len(p_skipped)} (< {MIN_TXN_COUNT} transactions)")
print(f"  Purchase anomalies    : {len(p_anom_df)}")


# ── PRINT SUMMARY TABLES ──────────────────────────────────────────────────────
print("\n── Top 10 Sales Anomalies (by deviation) ────────────────────")
print(s_anom_df[["Date","Customer","Amount","Entity_Mean",
                  "Deviation_%","Flag_Type"]].head(10).to_string(index=False))

print("\n── Top 10 Purchase Anomalies (by deviation) ─────────────────")
print(p_anom_df[["Date","Supplier","Amount","Entity_Mean",
                  "Deviation_%","Flag_Type"]].head(10).to_string(index=False))


# ── SAVE CSV ──────────────────────────────────────────────────────────────────
# Tag and combine both
s_anom_df["Txn_Type"] = "Sale"
p_anom_df["Txn_Type"] = "Purchase"
p_anom_df = p_anom_df.rename(columns={"Supplier": "Customer"})

all_anomalies = pd.concat([s_anom_df, p_anom_df], ignore_index=True)
all_anomalies = all_anomalies.sort_values(
    ["Txn_Type", "Deviation_%"], ascending=[True, False]
).reset_index(drop=True)

csv_path = os.path.join(OUTPUT_DIR, "anomalies.csv")
all_anomalies.to_csv(csv_path, index=False)
print(f"\n  anomalies.csv saved → {csv_path}")


# ── CHART ─────────────────────────────────────────────────────────────────────
print("\nGenerating chart...")

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :])   # Sales scatter — full width
ax2 = fig.add_subplot(gs[1, 0])   # Top anomalous customers bar
ax3 = fig.add_subplot(gs[1, 1])   # Top anomalous suppliers bar

# ── Chart 1: Sales scatter — normal vs anomaly ────────────────────────────────
normal_sales = sales[~sales.index.isin(s_anom_df.index)]

ax1.scatter(
    sales["Date"], sales["Debit_Amount"],
    color="#B5D4F4", alpha=0.4, s=15, label="Normal", zorder=2
)
# Plot anomalies coloured by type
high_flags = s_anom_df[s_anom_df["Flag_Type"].str.startswith("HIGH")]
low_flags  = s_anom_df[s_anom_df["Flag_Type"].str.startswith("LOW")]

# Re-merge date back for plotting
s_with_date = sales.copy()
s_with_date["Amount"] = s_with_date["Debit_Amount"]

high_plot = s_with_date[s_with_date["Vch_No"].isin(high_flags["Vch_No"])]
low_plot  = s_with_date[s_with_date["Vch_No"].isin(low_flags["Vch_No"])]

ax1.scatter(
    high_plot["Date"], high_plot["Debit_Amount"],
    color="#D85A30", alpha=0.85, s=60, marker="^",
    label=f"HIGH anomaly ({len(high_flags)})", zorder=4
)
ax1.scatter(
    low_plot["Date"], low_plot["Debit_Amount"],
    color="#BA7517", alpha=0.85, s=60, marker="v",
    label=f"LOW anomaly ({len(low_flags)})", zorder=4
)

# Average line
ax1.axhline(
    y=sales["Debit_Amount"].mean(),
    color="#888780", linestyle="--", linewidth=1, alpha=0.7,
    label=f"Overall avg ₹{sales['Debit_Amount'].mean()/1000:.0f}K"
)

ax1.set_title(
    f"Sales Transaction Anomalies — {len(s_anom_df)} flagged out of {len(sales)}",
    fontsize=11, fontweight="bold"
)
ax1.set_xlabel("Date", fontsize=9)
ax1.set_ylabel("Transaction Amount (₹)", fontsize=9)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"₹{v/1000:.0f}K"))
ax1.legend(fontsize=8)
ax1.grid(axis="y", alpha=0.25)

# ── Chart 2: Customers with most anomalies ────────────────────────────────────
cust_counts = s_anom_df["Customer"].value_counts().head(8)
colors2 = ["#D85A30" if c > 3 else "#EF9F27" for c in cust_counts.values]
bars2 = ax2.barh(
    [c[:25] for c in cust_counts.index[::-1]],
    cust_counts.values[::-1],
    color=colors2[::-1], edgecolor="white", linewidth=0.5
)
ax2.set_title("Customers with most flagged transactions", fontsize=10, fontweight="bold")
ax2.set_xlabel("Number of anomalies")
for bar, val in zip(bars2, cust_counts.values[::-1]):
    ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
             str(val), va="center", fontsize=9)
ax2.grid(axis="x", alpha=0.25)

# ── Chart 3: Suppliers with most anomalies ────────────────────────────────────
supp_counts = p_anom_df["Customer"].value_counts().head(8)
colors3 = ["#D85A30" if c > 3 else "#EF9F27" for c in supp_counts.values]
bars3 = ax3.barh(
    [s[:25] for s in supp_counts.index[::-1]],
    supp_counts.values[::-1],
    color=colors3[::-1], edgecolor="white", linewidth=0.5
)
ax3.set_title("Suppliers with most flagged purchases", fontsize=10, fontweight="bold")
ax3.set_xlabel("Number of anomalies")
for bar, val in zip(bars3, supp_counts.values[::-1]):
    ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
             str(val), va="center", fontsize=9)
ax3.grid(axis="x", alpha=0.25)

fig.suptitle(
    "Challenge Packaging Industries — Transaction Anomaly Detection",
    fontsize=13, fontweight="bold", y=1.01
)

# Note
fig.text(
    0.99, 0.01,
    f"Method: IQR per customer/supplier  |  "
    f"Threshold: Q1 − 1.5×IQR  to  Q3 + 1.5×IQR  |  "
    f"Min transactions for analysis: {MIN_TXN_COUNT}",
    ha="right", va="bottom", fontsize=7, color="#888780"
)

chart_path = os.path.join(OUTPUT_DIR, "anomaly_detection.png")
plt.tight_layout()
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  anomaly_detection.png saved → {chart_path}")


# ── BUSINESS SUMMARY ──────────────────────────────────────────────────────────
print("\n── Business Action Items ────────────────────────────────────")
print(f"\n  SALES ({len(s_anom_df)} anomalies):")

high_s = s_anom_df[s_anom_df["Flag_Type"].str.startswith("HIGH")]
low_s  = s_anom_df[s_anom_df["Flag_Type"].str.startswith("LOW")]
print(f"  Unusually large  : {len(high_s)} transactions — verify these were genuine orders")
print(f"  Unusually small  : {len(low_s)} transactions — may be partial/credit adjustments")

top_s = s_anom_df.groupby("Customer")["Amount"].sum().sort_values(ascending=False).head(3)
print(f"\n  Top customers to review:")
for cust, val in top_s.items():
    print(f"    {cust:<45} ₹{val/100000:.1f}L in flagged transactions")

print(f"\n  PURCHASES ({len(p_anom_df)} anomalies):")
high_p = p_anom_df[p_anom_df["Flag_Type"].str.startswith("HIGH")]
low_p  = p_anom_df[p_anom_df["Flag_Type"].str.startswith("LOW")]
print(f"  Unusually large  : {len(high_p)} — bulk buys or price spikes?")
print(f"  Unusually small  : {len(low_p)} — partial deliveries or returns?")

top_p = p_anom_df.groupby("Customer")["Amount"].sum().sort_values(ascending=False).head(3)
print(f"\n  Top suppliers to review:")
for supp, val in top_p.items():
    print(f"    {supp:<45} ₹{val/100000:.1f}L in flagged transactions")

print(f"\n  Note on J.C Packaging:")
print(f"  Many J.C flags are a gradual price increase over 2 years")
print(f"  (Rs.3-8K → Rs.11-17K). This is normal business growth,")
print(f"  not a genuine anomaly. Bhuvanesh to confirm if prices rose.")

print("\n✅ Anomaly detection complete!")
print(f"   CSV → {csv_path}")
print(f"   PNG → {chart_path}")