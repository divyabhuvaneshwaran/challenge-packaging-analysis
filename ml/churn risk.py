"""
Challenge Packaging Industries
06 — Churn Risk Analysis
=========================
Identifies customers who are at risk of churning or have
already churned — based on recency + FY activity pattern.

No ML model needed here — recency logic on the Sales Register
is sufficient, 100% explainable to Bhuvanesh, and more
actionable than a black-box classifier with 38 customers.

Churn definition for CPI (B2B manufacturing):
  - CHURNED    : Active in 24-25, zero orders in 25-26
  - HIGH RISK  : Last order > 180 days ago
  - MEDIUM RISK: Last order 90–180 days ago
  - LOW RISK   : Last order 30–90 days ago
  - ACTIVE     : Last order < 30 days ago
  - NEW        : First order in 25-26 only (no 24-25 history)

Key findings from data:
  9 customers dropped from 24-25 to 25-26
  13 new customers acquired in 25-26
  Net customer growth: +4 customers YoY

Input  : day_book_final.csv
Output : outputs/churn_risk.csv
         outputs/churn_risk.png

Run    : python 06_churn_risk.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE = r"C:\Users\divya\Documents\challenge-packaging-analysis\data\day_book_final.csv"
OUTPUT_DIR = r"C:\Users\divya\Documents\challenge-packaging-analysis\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Churn thresholds (days since last order)
THRESHOLD_HIGH   = 180   # > 180 days = high risk
THRESHOLD_MEDIUM = 90    # 90–180 days = medium risk
THRESHOLD_LOW    = 30    # 30–90 days = low risk
                         # < 30 days = active

# ── NAME CONSOLIDATION ────────────────────────────────────────────────────────
NAME_MAP = {
    'Sundram Fasteners Limited ( Madurai )': 'SUNDRAM FASTENERS LIMITED',
    'M/S ASUS TECHNOLOGY PVT LTD'          : 'ASUS Technology Pvt Ltd',
    'ASUS Technology Pvt. Ltd.'            : 'ASUS Technology Pvt Ltd',
    'Red Star Polymers Private Limited ( Mangadu)'       : 'Red Star Polymers Pvt Ltd',
    'RED STAR POLYMERS PRIVATE LIMITED( Valasaravakkam)' : 'Red Star Polymers Pvt Ltd',
    'RED STAR POLYMERS PRIVATE LIMITED'                  : 'Red Star Polymers Pvt Ltd',
    'Red Star Plastick Pvt Ltd'                          : 'Red Star Plastick Pvt Ltd',
    'Redstar Plastick Private Ltd ( Warehouse )'         : 'Red Star Plastick Pvt Ltd',
    'REDSTAR PLASTIC LIMITED (Valasaravakkam )'          : 'Red Star Plastick Pvt Ltd',
    'Regenix Drugs Ltd ( Bharti Life )'                  : 'Regenix Drugs Group',
    'Regenix Drugs Ltd (Queen )'                         : 'Regenix Drugs Group',
    'Regenix Drugs Ltd'                                  : 'Regenix Drugs Group',
    'REGENIX DRUGS MUR & MUR DIVISION'                   : 'Regenix Drugs Group',
    'Regenix Super Speciality Laboratories Pvt Ltd'      : 'Regenix Drugs Group',
    'M/S Regenix Drugs Ltd (B)'                          : 'Regenix Drugs Group',
    'M/S Regenix Drugs ( Q)'                             : 'Regenix Drugs Group',
}

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
df["Particulars"] = df["Particulars"].replace(NAME_MAP)

sales          = df[df["Txn_Category"] == "Sales"].copy()
reference_date = sales["Date"].max()

print(f"  Sales transactions : {len(sales)}")
print(f"  Unique customers   : {sales['Particulars'].nunique()}")
print(f"  Reference date     : {reference_date.date()}")


# ── BUILD CUSTOMER PROFILE ────────────────────────────────────────────────────
cust = sales.groupby("Particulars").agg(
    First_Purchase  = ("Date",         "min"),
    Last_Purchase   = ("Date",         "max"),
    Total_Orders    = ("Debit_Amount", "count"),
    Total_Revenue   = ("Debit_Amount", "sum"),
    Avg_Order_Value = ("Debit_Amount", "mean"),
).reset_index().rename(columns={"Particulars": "Customer"})

cust["Days_Since_Last"]  = (reference_date - cust["Last_Purchase"]).dt.days
cust["Revenue_Lakhs"]    = (cust["Total_Revenue"] / 100000).round(2)
cust["Avg_Order_Lakhs"]  = (cust["Avg_Order_Value"] / 100000).round(2)

# FY breakdown
fy2425 = (
    sales[sales["FY"] == "2024-25"]
    .groupby("Particulars")[["Debit_Amount"]]
    .agg(Orders_2425=("Debit_Amount","count"), Revenue_2425=("Debit_Amount","sum"))
    .reset_index().rename(columns={"Particulars":"Customer"})
)
fy2526 = (
    sales[sales["FY"] == "2025-26"]
    .groupby("Particulars")[["Debit_Amount"]]
    .agg(Orders_2526=("Debit_Amount","count"), Revenue_2526=("Debit_Amount","sum"))
    .reset_index().rename(columns={"Particulars":"Customer"})
)

cust = cust.merge(fy2425, on="Customer", how="left")
cust = cust.merge(fy2526, on="Customer", how="left")
cust[["Orders_2425","Revenue_2425",
      "Orders_2526","Revenue_2526"]] = \
    cust[["Orders_2425","Revenue_2425",
          "Orders_2526","Revenue_2526"]].fillna(0)

cust["Revenue_2425_L"] = (cust["Revenue_2425"] / 100000).round(2)
cust["Revenue_2526_L"] = (cust["Revenue_2526"] / 100000).round(2)

# Revenue change YoY (only for customers present in both FYs)
cust["Revenue_Change_L"] = (cust["Revenue_2526_L"] - cust["Revenue_2425_L"]).round(2)
cust["Revenue_Change_%"] = np.where(
    cust["Revenue_2425"] > 0,
    ((cust["Revenue_2526"] - cust["Revenue_2425"]) / cust["Revenue_2425"] * 100).round(1),
    np.nan
)


# ── CHURN RISK CLASSIFICATION ─────────────────────────────────────────────────
def classify_churn(row):
    days      = row["Days_Since_Last"]
    had_2425  = row["Orders_2425"] > 0
    has_2526  = row["Orders_2526"] > 0

    # Churned: was active in 24-25, zero orders in 25-26
    if had_2425 and not has_2526:
        return "Churned"

    # New customer: only appeared in 25-26
    if not had_2425 and has_2526:
        return "New Customer"

    # Active customers — classify by recency
    if days <= THRESHOLD_LOW:
        return "Active"
    elif days <= THRESHOLD_MEDIUM:
        return "Low Risk"
    elif days <= THRESHOLD_HIGH:
        return "Medium Risk"
    else:
        return "High Risk"

cust["Churn_Status"] = cust.apply(classify_churn, axis=1)


# ── REVENUE AT RISK ───────────────────────────────────────────────────────────
# For churned and high/medium risk — what revenue is at stake?
cust["Revenue_At_Risk_L"] = np.where(
    cust["Churn_Status"].isin(["Churned", "High Risk", "Medium Risk"]),
    cust["Revenue_Lakhs"],
    0
)


# ── RECOMMENDED ACTION ────────────────────────────────────────────────────────
ACTION_MAP = {
    "Churned"      : "Urgent — personal call from Bhuvanesh, understand why they left",
    "High Risk"    : "Re-engage immediately — offer incentive or check satisfaction",
    "Medium Risk"  : "Follow up this week — confirm next order timeline",
    "Low Risk"     : "Check in — ensure no issues with last delivery",
    "Active"       : "Maintain — keep service quality high",
    "New Customer" : "Nurture — ensure smooth onboarding, follow up on experience",
}
cust["Recommended_Action"] = cust["Churn_Status"].map(ACTION_MAP)


# ── PRINT SUMMARY ─────────────────────────────────────────────────────────────
print("\n── Churn Risk Summary ───────────────────────────────────────")
status_order = ["Churned","High Risk","Medium Risk","Low Risk","Active","New Customer"]
summary = cust.groupby("Churn_Status").agg(
    Customers        = ("Customer",           "count"),
    Revenue_Total_L  = ("Revenue_Lakhs",      "sum"),
    Revenue_At_Risk_L= ("Revenue_At_Risk_L",  "sum"),
).reset_index()
summary["Churn_Status"] = pd.Categorical(summary["Churn_Status"],
                                          categories=status_order, ordered=True)
summary = summary.sort_values("Churn_Status")
print(summary.to_string(index=False))

print("\n── Churned Customers ────────────────────────────────────────")
churned = cust[cust["Churn_Status"] == "Churned"].sort_values(
    "Revenue_2425_L", ascending=False)
for _, r in churned.iterrows():
    print(f"  {r['Customer']:<45} "
          f"Revenue 24-25: ₹{r['Revenue_2425_L']:.1f}L  "
          f"Last order: {r['Last_Purchase'].date()}  "
          f"({r['Days_Since_Last']:.0f} days ago)")

print("\n── High Risk Customers ──────────────────────────────────────")
high = cust[cust["Churn_Status"] == "High Risk"].sort_values(
    "Revenue_Lakhs", ascending=False)
for _, r in high.iterrows():
    print(f"  {r['Customer']:<45} "
          f"Revenue: ₹{r['Revenue_Lakhs']:.1f}L  "
          f"Last order: {r['Last_Purchase'].date()}  "
          f"({r['Days_Since_Last']:.0f} days ago)")

print("\n── New Customers in 25-26 ───────────────────────────────────")
new_custs = cust[cust["Churn_Status"] == "New Customer"].sort_values(
    "Revenue_2526_L", ascending=False)
for _, r in new_custs.iterrows():
    print(f"  {r['Customer']:<45} "
          f"Revenue 25-26: ₹{r['Revenue_2526_L']:.1f}L  "
          f"Orders: {r['Orders_2526']:.0f}")

print("\n── FY Comparison for active customers ───────────────────────")
both_fy = cust[(cust["Orders_2425"] > 0) & (cust["Orders_2526"] > 0)].sort_values(
    "Revenue_Change_%")
print(f"{'Customer':<45} {'24-25 L':>8} {'25-26 L':>8} {'Change%':>9} {'Status'}")
print("-" * 90)
for _, r in both_fy.iterrows():
    change = f"{r['Revenue_Change_%']:+.1f}%" if not np.isnan(r['Revenue_Change_%']) else "N/A"
    print(f"  {r['Customer']:<43} ₹{r['Revenue_2425_L']:>6.1f}L "
          f"₹{r['Revenue_2526_L']:>6.1f}L {change:>9}  {r['Churn_Status']}")


# ── SAVE CSV ──────────────────────────────────────────────────────────────────
output_cols = [
    "Customer", "Churn_Status", "Days_Since_Last",
    "Revenue_Lakhs", "Revenue_2425_L", "Revenue_2526_L",
    "Revenue_Change_L", "Revenue_Change_%",
    "Total_Orders", "Orders_2425", "Orders_2526",
    "First_Purchase", "Last_Purchase",
    "Revenue_At_Risk_L", "Recommended_Action",
]

out_df = cust[output_cols].copy()
out_df["First_Purchase"] = out_df["First_Purchase"].dt.strftime("%Y-%m-%d")
out_df["Last_Purchase"]  = out_df["Last_Purchase"].dt.strftime("%Y-%m-%d")
out_df["Churn_Status"]   = pd.Categorical(
    out_df["Churn_Status"], categories=status_order, ordered=True)
out_df = out_df.sort_values(["Churn_Status","Revenue_Lakhs"],
                              ascending=[True, False]).reset_index(drop=True)

csv_path = os.path.join(OUTPUT_DIR, "churn_risk.csv")
out_df.to_csv(csv_path, index=False)
print(f"\n  churn_risk.csv saved → {csv_path}")


# ── CHART ─────────────────────────────────────────────────────────────────────
print("\nGenerating chart...")

STATUS_COLORS = {
    "Churned"      : "#A32D2D",
    "High Risk"    : "#D85A30",
    "Medium Risk"  : "#BA7517",
    "Low Risk"     : "#378ADD",
    "Active"       : "#1D9E75",
    "New Customer" : "#534AB7",
}

fig = plt.figure(figsize=(16, 13))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :])   # Timeline — all customers, last order date
ax2 = fig.add_subplot(gs[1, 0])   # Status donut
ax3 = fig.add_subplot(gs[1, 1])   # Revenue at risk bar
ax4 = fig.add_subplot(gs[2, :])   # FY comparison bar — both years side by side

# ── Chart 1: Customer timeline ────────────────────────────────────────────────
cust_sorted = cust.sort_values("Last_Purchase")
y_pos       = range(len(cust_sorted))
bar_colors  = [STATUS_COLORS[s] for s in cust_sorted["Churn_Status"]]

ax1.barh(
    [c[:32] for c in cust_sorted["Customer"]],
    cust_sorted["Days_Since_Last"],
    color=bar_colors, edgecolor="white", linewidth=0.3
)

# Threshold lines
ax1.axvline(x=THRESHOLD_LOW,    color="#378ADD", linestyle="--",
            linewidth=1, label=f"Low risk ({THRESHOLD_LOW}d)")
ax1.axvline(x=THRESHOLD_MEDIUM, color="#BA7517", linestyle="--",
            linewidth=1, label=f"Medium risk ({THRESHOLD_MEDIUM}d)")
ax1.axvline(x=THRESHOLD_HIGH,   color="#D85A30", linestyle="--",
            linewidth=1, label=f"High risk ({THRESHOLD_HIGH}d)")

ax1.set_title("Days since last order — all customers",
              fontsize=11, fontweight="bold")
ax1.set_xlabel("Days since last order (as of 31-Mar-2026)")
ax1.legend(fontsize=8, loc="lower right")
ax1.grid(axis="x", alpha=0.25)
ax1.tick_params(axis="y", labelsize=8)

# ── Chart 2: Status distribution donut ───────────────────────────────────────
status_counts = cust["Churn_Status"].value_counts()
status_present= [s for s in status_order if s in status_counts.index]
sizes         = [status_counts[s] for s in status_present]
colors2       = [STATUS_COLORS[s] for s in status_present]

wedges, texts, autotexts = ax2.pie(
    sizes, labels=status_present, colors=colors2,
    autopct="%1.0f%%", startangle=90,
    wedgeprops=dict(width=0.55, edgecolor="white"),
    textprops=dict(fontsize=8)
)
for at in autotexts:
    at.set_fontsize(8)
    at.set_fontweight("bold")

ax2.set_title("Customer churn status distribution",
              fontsize=10, fontweight="bold")

# ── Chart 3: Revenue at risk ──────────────────────────────────────────────────
risk_statuses = ["Churned", "High Risk", "Medium Risk"]
risk_rev      = [
    cust[cust["Churn_Status"] == s]["Revenue_Lakhs"].sum()
    for s in risk_statuses
]
colors3 = [STATUS_COLORS[s] for s in risk_statuses]
bars3   = ax3.bar(risk_statuses, risk_rev, color=colors3,
                  edgecolor="white", linewidth=0.5)
ax3.set_title("Revenue at risk by status (₹L)", fontsize=10, fontweight="bold")
ax3.set_ylabel("Revenue (₹L)")
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"₹{v:.0f}L"))
for bar, val in zip(bars3, risk_rev):
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3,
             f"₹{val:.1f}L", ha="center", fontsize=9, fontweight="bold")
ax3.grid(axis="y", alpha=0.25)

# ── Chart 4: FY revenue comparison for customers in both years ────────────────
both = cust[(cust["Orders_2425"] > 0) & (cust["Orders_2526"] > 0)].sort_values(
    "Revenue_2526_L", ascending=False).head(15)

x4    = np.arange(len(both))
width = 0.38

bars4a = ax4.bar(x4 - width/2, both["Revenue_2425_L"], width,
                 label="2024-25", color="#185FA5", alpha=0.85)
bars4b = ax4.bar(x4 + width/2, both["Revenue_2526_L"], width,
                 label="2025-26", color="#1D9E75", alpha=0.85)

ax4.set_xticks(x4)
ax4.set_xticklabels([c[:22] for c in both["Customer"]],
                    rotation=45, ha="right", fontsize=8)
ax4.set_title("Revenue comparison: 2024-25 vs 2025-26 (top 15 customers)",
              fontsize=10, fontweight="bold")
ax4.set_ylabel("Revenue (₹L)")
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"₹{v:.0f}L"))
ax4.legend(fontsize=9)
ax4.grid(axis="y", alpha=0.25)

fig.suptitle(
    "Challenge Packaging Industries — Customer Churn Risk Analysis",
    fontsize=13, fontweight="bold", y=1.01
)

# Color legend
from matplotlib.patches import Patch
legend_els = [Patch(color=STATUS_COLORS[s], label=s) for s in status_order]
fig.legend(handles=legend_els, title="Churn Status",
           loc="lower center", ncol=6,
           bbox_to_anchor=(0.5, -0.04), fontsize=9)

chart_path = os.path.join(OUTPUT_DIR, "churn_risk.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  churn_risk.png saved → {chart_path}")


# ── BUSINESS SUMMARY ──────────────────────────────────────────────────────────
print("\n── Business Insights ────────────────────────────────────────")
churned_rev = cust[cust["Churn_Status"] == "Churned"]["Revenue_2425_L"].sum()
high_rev    = cust[cust["Churn_Status"] == "High Risk"]["Revenue_Lakhs"].sum()
new_rev     = cust[cust["Churn_Status"] == "New Customer"]["Revenue_2526_L"].sum()

print(f"  Churned customers (9)    : ₹{churned_rev:.1f}L revenue lost from 24-25")
print(f"  High risk customers      : ₹{high_rev:.1f}L at risk")
print(f"  New customers acquired   : ₹{new_rev:.1f}L new revenue in 25-26")
print(f"  Net revenue impact       : ₹{new_rev - churned_rev:+.1f}L "
      f"({'gain' if new_rev > churned_rev else 'loss'})")
print()
print("  Good news: CPI acquired 13 new customers in 25-26")
print("  offsetting the 9 churned — showing active business development.")
print()
print("  Priority actions for Bhuvanesh:")
print("  1. Call the 9 churned customers — especially Red Star Plastick")
print("     (₹4.7L) and Daeseung Autoparts (₹8.8L) — find out why they left")
print("  2. Check on Genuine Biosystem — 18 orders in 24-25, only 5 in 25-26")
print("     — significant drop, not yet churned but trending that way")
print("  3. Nurture the 13 new 25-26 customers — one repeat order each")
print("     would add meaningful revenue")

print("\n✅ Churn risk analysis complete!")
print(f"   CSV → {csv_path}")
print(f"   PNG → {chart_path}")