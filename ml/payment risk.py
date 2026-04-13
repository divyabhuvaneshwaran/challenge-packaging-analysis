"""
Challenge Packaging Industries
05 — Payment Risk Analysis
===========================
Analyses how reliably each customer pays their invoices.
Uses total invoiced vs total received to calculate:
  - Collection rate per customer
  - Outstanding amount
  - Payment risk tier
  - Advance payment flag (received > invoiced)

Note on negative outstanding:
  Some customers show negative outstanding (received > invoiced).
  This means they either paid in advance OR have a running credit
  balance from previous periods before Apr 2024. Both are healthy.

Input  : day_book_final.csv
Output : outputs/payment_risk.csv
         outputs/payment_risk.png

Run    : python 05_payment_risk.py
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

# Risk tier thresholds
TIER_EXCELLENT  = 95   # collection rate >= 95% = excellent payer
TIER_GOOD       = 85   # collection rate >= 85% = good
TIER_MODERATE   = 70   # collection rate >= 70% = moderate risk
                       # below 70% = high risk

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

sales    = df[df["Txn_Category"] == "Sales"].copy()
receipts = df[df["Txn_Category"] == "Receipt"].copy()

# Filter receipts to customer names only — exclude salary, GST, loan entries
sales_customers = set(sales["Particulars"].unique())
cust_receipts   = receipts[receipts["Particulars"].isin(sales_customers)].copy()

print(f"  Sales transactions    : {len(sales)}")
print(f"  Customer receipts     : {len(cust_receipts)}")
print(f"  Unique customers      : {sales['Particulars'].nunique()}")


# ── PAYMENT ANALYSIS ──────────────────────────────────────────────────────────
# Total invoiced per customer
cust_sales = (
    sales.groupby("Particulars")
    .agg(
        Total_Invoiced  = ("Debit_Amount", "sum"),
        Num_Invoices    = ("Debit_Amount", "count"),
        First_Invoice   = ("Date",         "min"),
        Last_Invoice    = ("Date",         "max"),
    )
    .reset_index()
    .rename(columns={"Particulars": "Customer"})
)

# Total received per customer
cust_recv = (
    cust_receipts.groupby("Particulars")
    .agg(
        Total_Received  = ("Credit_Amount", "sum"),
        Num_Receipts    = ("Credit_Amount", "count"),
        Last_Payment    = ("Date",          "max"),
    )
    .reset_index()
    .rename(columns={"Particulars": "Customer"})
)

# Merge
pay = cust_sales.merge(cust_recv, on="Customer", how="left")
pay["Total_Received"] = pay["Total_Received"].fillna(0)
pay["Num_Receipts"]   = pay["Num_Receipts"].fillna(0).astype(int)

# ── CALCULATIONS ──────────────────────────────────────────────────────────────
pay["Outstanding"]       = pay["Total_Invoiced"] - pay["Total_Received"]
pay["Collection_Rate%"]  = (pay["Total_Received"] / pay["Total_Invoiced"] * 100).round(1)
pay["Invoiced_Lakhs"]    = (pay["Total_Invoiced"] / 100000).round(2)
pay["Received_Lakhs"]    = (pay["Total_Received"] / 100000).round(2)
pay["Outstanding_Lakhs"] = (pay["Outstanding"] / 100000).round(2)

# Days since last payment
reference_date         = df["Date"].max()
pay["Last_Payment"]    = pd.to_datetime(pay["Last_Payment"])
pay["Days_Since_Pay"]  = (reference_date - pay["Last_Payment"]).dt.days
pay["Days_Since_Pay"]  = pay["Days_Since_Pay"].fillna(999).astype(int)

# Format dates
pay["First_Invoice"] = pay["First_Invoice"].dt.strftime("%Y-%m-%d")
pay["Last_Invoice"]  = pay["Last_Invoice"].dt.strftime("%Y-%m-%d")
pay["Last_Payment"]  = pay["Last_Payment"].dt.strftime("%Y-%m-%d").fillna("No payment")


# ── RISK TIER ASSIGNMENT ──────────────────────────────────────────────────────
def assign_risk(row):
    rate = row["Collection_Rate%"]
    outstanding = row["Outstanding"]

    # Advance payer — received more than invoiced
    if rate > 100:
        return "Advance Payer"

    # No invoices received at all
    if row["Total_Received"] == 0 and row["Total_Invoiced"] > 0:
        return "High Risk"

    if rate >= TIER_EXCELLENT:
        return "Excellent"
    elif rate >= TIER_GOOD:
        return "Good"
    elif rate >= TIER_MODERATE:
        return "Moderate Risk"
    else:
        return "High Risk"

pay["Risk_Tier"] = pay.apply(assign_risk, axis=1)


# ── ACTION RECOMMENDED ────────────────────────────────────────────────────────
ACTION_MAP = {
    "Advance Payer" : "Prioritise — they pay ahead, reward with best service",
    "Excellent"     : "Maintain — keep relationship strong",
    "Good"          : "Monitor — follow up on outstanding balance",
    "Moderate Risk" : "Follow up — send formal payment reminder",
    "High Risk"     : "Urgent — escalate to Bhuvanesh for direct call",
}
pay["Recommended_Action"] = pay["Risk_Tier"].map(ACTION_MAP)


# ── PRINT SUMMARY ─────────────────────────────────────────────────────────────
print("\n── Payment Risk Summary ─────────────────────────────────────")
tier_summary = pay.groupby("Risk_Tier").agg(
    Customers        = ("Customer",       "count"),
    Total_Invoiced_L = ("Invoiced_Lakhs", "sum"),
    Total_Outstanding_L = ("Outstanding_Lakhs", "sum"),
).reset_index()
print(tier_summary.to_string(index=False))

print(f"\n  Total outstanding across all customers : "
      f"Rs.{pay['Outstanding'].sum():,.0f} "
      f"(Rs.{pay['Outstanding'].sum()/100000:.1f}L)")
print(f"  Overall collection rate                : "
      f"{pay['Total_Received'].sum()/pay['Total_Invoiced'].sum()*100:.1f}%")

print("\n── High Risk Customers ──────────────────────────────────────")
high = pay[pay["Risk_Tier"] == "High Risk"].sort_values("Outstanding", ascending=False)
if len(high) > 0:
    print(high[["Customer","Invoiced_Lakhs","Outstanding_Lakhs",
                "Collection_Rate%","Last_Payment","Days_Since_Pay"]].to_string(index=False))
else:
    print("  None — all customers have made at least some payment")

print("\n── Moderate Risk Customers ──────────────────────────────────")
mod = pay[pay["Risk_Tier"] == "Moderate Risk"].sort_values("Outstanding", ascending=False)
print(mod[["Customer","Invoiced_Lakhs","Outstanding_Lakhs",
           "Collection_Rate%","Last_Payment"]].to_string(index=False))

print("\n── Advance Payers (credit balance) ──────────────────────────")
adv = pay[pay["Risk_Tier"] == "Advance Payer"].sort_values("Outstanding")
print(adv[["Customer","Invoiced_Lakhs","Received_Lakhs",
           "Outstanding_Lakhs","Collection_Rate%"]].to_string(index=False))


# ── SAVE CSV ──────────────────────────────────────────────────────────────────
output_cols = [
    "Customer", "Risk_Tier", "Collection_Rate%",
    "Invoiced_Lakhs", "Received_Lakhs", "Outstanding_Lakhs",
    "Num_Invoices", "Num_Receipts",
    "First_Invoice", "Last_Invoice", "Last_Payment", "Days_Since_Pay",
    "Recommended_Action",
]
out_df   = pay[output_cols].sort_values(
    ["Risk_Tier", "Outstanding_Lakhs"], ascending=[True, False]
).reset_index(drop=True)

csv_path = os.path.join(OUTPUT_DIR, "payment_risk.csv")
out_df.to_csv(csv_path, index=False)
print(f"\n  payment_risk.csv saved → {csv_path}")


# ── CHART ─────────────────────────────────────────────────────────────────────
print("\nGenerating chart...")

TIER_COLORS = {
    "Advance Payer" : "#0F6E56",
    "Excellent"     : "#1D9E75",
    "Good"          : "#378ADD",
    "Moderate Risk" : "#BA7517",
    "High Risk"     : "#D85A30",
}

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :])   # Collection rate bar — all customers
ax2 = fig.add_subplot(gs[1, 0])   # Outstanding by tier
ax3 = fig.add_subplot(gs[1, 1])   # Customer count by tier

# ── Chart 1: Collection rate per customer ────────────────────────────────────
pay_sorted = pay.sort_values("Collection_Rate%", ascending=True)
bar_colors = [TIER_COLORS[t] for t in pay_sorted["Risk_Tier"]]

bars = ax1.barh(
    [c[:30] for c in pay_sorted["Customer"]],
    pay_sorted["Collection_Rate%"].clip(upper=150),
    color=bar_colors, edgecolor="white", linewidth=0.3
)

# Reference lines
ax1.axvline(x=100, color="#2C2C2A", linewidth=1.2, linestyle="-",  label="100% collected")
ax1.axvline(x=TIER_EXCELLENT, color="#1D9E75", linewidth=1,
            linestyle="--", label=f"Excellent threshold ({TIER_EXCELLENT}%)")
ax1.axvline(x=TIER_GOOD,      color="#378ADD", linewidth=1,
            linestyle="--", label=f"Good threshold ({TIER_GOOD}%)")
ax1.axvline(x=TIER_MODERATE,  color="#BA7517", linewidth=1,
            linestyle="--", label=f"Moderate threshold ({TIER_MODERATE}%)")

ax1.set_title("Collection Rate by Customer (>100% = advance payer)",
              fontsize=11, fontweight="bold")
ax1.set_xlabel("Collection Rate %")
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax1.legend(fontsize=8, loc="lower right")
ax1.grid(axis="x", alpha=0.25)
ax1.tick_params(axis="y", labelsize=8)

# ── Chart 2: Outstanding amount by customer (positive only) ──────────────────
outstanding_pos = pay[pay["Outstanding"] > 0].sort_values("Outstanding", ascending=True)
bar_colors2     = [TIER_COLORS[t] for t in outstanding_pos["Risk_Tier"]]

ax2.barh(
    [c[:28] for c in outstanding_pos["Customer"]],
    outstanding_pos["Outstanding_Lakhs"],
    color=bar_colors2, edgecolor="white", linewidth=0.3
)
ax2.set_title("Outstanding amount by customer (₹L)",
              fontsize=10, fontweight="bold")
ax2.set_xlabel("Outstanding (₹L)")
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"₹{v:.1f}L"))
ax2.tick_params(axis="y", labelsize=8)
ax2.grid(axis="x", alpha=0.25)

# ── Chart 3: Customer count + revenue by tier ────────────────────────────────
tier_order   = ["Advance Payer", "Excellent", "Good", "Moderate Risk", "High Risk"]
tier_present = [t for t in tier_order if t in pay["Risk_Tier"].values]
counts       = [len(pay[pay["Risk_Tier"] == t]) for t in tier_present]
revenues     = [pay[pay["Risk_Tier"] == t]["Invoiced_Lakhs"].sum() for t in tier_present]
colors3      = [TIER_COLORS[t] for t in tier_present]

x3  = np.arange(len(tier_present))
w   = 0.4

bars3a = ax3.bar(x3 - w/2, counts,   w, color=colors3, alpha=0.9,  label="Customers")
bars3b = ax3.bar(x3 + w/2, revenues, w, color=colors3, alpha=0.55, label="Revenue (₹L)",
                 hatch="//")

ax3.set_xticks(x3)
ax3.set_xticklabels([t.replace(" ", "\n") for t in tier_present], fontsize=8)
ax3.set_title("Customers and revenue by risk tier",
              fontsize=10, fontweight="bold")
ax3.legend(fontsize=8)
ax3.grid(axis="y", alpha=0.25)

for bar, val in zip(bars3a, counts):
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3,
             str(val), ha="center", fontsize=9, fontweight="bold")

fig.suptitle(
    "Challenge Packaging Industries — Customer Payment Risk Analysis",
    fontsize=13, fontweight="bold", y=1.01
)

# Legend for tier colors
from matplotlib.patches import Patch
legend_els = [Patch(color=TIER_COLORS[t], label=t) for t in tier_present]
fig.legend(handles=legend_els, title="Risk Tier",
           loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, -0.04), fontsize=9)

chart_path = os.path.join(OUTPUT_DIR, "payment_risk.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  payment_risk.png saved → {chart_path}")


# ── BUSINESS SUMMARY ──────────────────────────────────────────────────────────
print("\n── Business Insights ────────────────────────────────────────")
print(f"  Overall collection rate  : "
      f"{pay['Total_Received'].sum()/pay['Total_Invoiced'].sum()*100:.1f}%  — very healthy")
print(f"  Total outstanding        : Rs.{pay[pay['Outstanding']>0]['Outstanding'].sum():,.0f}")
print()

adv_count = len(pay[pay["Risk_Tier"] == "Advance Payer"])
exc_count = len(pay[pay["Risk_Tier"] == "Excellent"])
print(f"  {adv_count} customers paid more than invoiced — running advance credit")
print(f"  {exc_count} customers at 95%+ collection rate — extremely reliable")
print()

high_risk = pay[pay["Risk_Tier"] == "High Risk"]
if len(high_risk) > 0:
    print(f"  HIGH RISK ({len(high_risk)} customers):")
    for _, r in high_risk.iterrows():
        print(f"    {r['Customer']:<40} Rs.{r['Outstanding_Lakhs']:.1f}L outstanding "
              f"({r['Collection_Rate%']:.1f}% collected)")
print()

mod_risk = pay[pay["Risk_Tier"] == "Moderate Risk"]
if len(mod_risk) > 0:
    print(f"  MODERATE RISK ({len(mod_risk)} customers):")
    for _, r in mod_risk.iterrows():
        print(f"    {r['Customer']:<40} Rs.{r['Outstanding_Lakhs']:.1f}L outstanding "
              f"({r['Collection_Rate%']:.1f}% collected)")
print()
print("  Note: Negative outstanding = advance payment or prior period credit.")
print("  These customers are trustworthy — not a concern.")

print("\n✅ Payment risk analysis complete!")
print(f"   CSV → {csv_path}")
print(f"   PNG → {chart_path}")