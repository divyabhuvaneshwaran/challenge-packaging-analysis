"""
Challenge Packaging Industries
01 — Customer RFM Segmentation
================================
Input  : day_book_final.csv
Output : outputs/rfm_scores.csv
         outputs/rfm_heatmap.png
         outputs/rfm_segment_summary.png

Run    : python 01_rfm_segmentation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE = r"C:\Users\divya\Documents\challenge-packaging-analysis\data\day_book_final.csv"
OUTPUT_DIR = r"C:\Users\divya\Documents\challenge-packaging-analysis\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── CUSTOMER NAME CONSOLIDATION MAP ──────────────────────────────────────────
# Confirmed with Bhuvanesh — same org, different branches/spellings
NAME_MAP = {
    # Sundram Fasteners — Chennai HQ + Madurai plant = one org
    "Sundram Fasteners Limited ( Madurai )"          : "SUNDRAM FASTENERS LIMITED",

    # ASUS Technology — two name variants
    "M/S ASUS TECHNOLOGY PVT LTD"                    : "ASUS Technology Pvt Ltd",
    "ASUS Technology Pvt. Ltd."                      : "ASUS Technology Pvt Ltd",

    # Red Star Polymers — same company, different branches
    "Red Star Polymers Private Limited ( Mangadu)"   : "Red Star Polymers Pvt Ltd",
    "RED STAR POLYMERS PRIVATE LIMITED( Valasaravakkam)" : "Red Star Polymers Pvt Ltd",
    "RED STAR POLYMERS PRIVATE LIMITED"              : "Red Star Polymers Pvt Ltd",

    # Red Star Plastick — warehouse + main (note: different company from Polymers)
    "Red Star Plastick Pvt Ltd"                      : "Red Star Plastick Pvt Ltd",
    "Redstar Plastick Private Ltd ( Warehouse )"     : "Red Star Plastick Pvt Ltd",
    "REDSTAR PLASTIC LIMITED (Valasaravakkam )"      : "Red Star Plastick Pvt Ltd",

    # Regenix Drugs Group — confirmed group by Bhuvanesh
    "Regenix Drugs Ltd ( Bharti Life )"              : "Regenix Drugs Group",
    "Regenix Drugs Ltd (Queen )"                     : "Regenix Drugs Group",
    "Regenix Drugs Ltd"                              : "Regenix Drugs Group",
    "REGENIX DRUGS MUR & MUR DIVISION"               : "Regenix Drugs Group",
    "Regenix Super Speciality Laboratories Pvt Ltd"  : "Regenix Drugs Group",
    "M/S Regenix Drugs Ltd (B)"                      : "Regenix Drugs Group",
    "M/S Regenix Drugs ( Q)"                         : "Regenix Drugs Group",
}

# ── LOAD & CLEAN ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
sales = df[df["Txn_Category"] == "Sales"].copy()
sales = sales[sales["Debit_Amount"] > 0].copy()

# Apply name consolidation
sales["Customer"] = sales["Particulars"].replace(NAME_MAP)

print(f"  Sales transactions : {len(sales)}")
print(f"  Unique customers   : {sales['Customer'].nunique()} (after consolidation)")

# ── RFM CALCULATION ───────────────────────────────────────────────────────────
# Reference date = day after last transaction (standard RFM practice)
reference_date = sales["Date"].max() + pd.Timedelta(days=1)

rfm = sales.groupby("Customer").agg(
    Recency   = ("Date",          lambda x: (reference_date - x.max()).days),
    Frequency = ("Debit_Amount",  "count"),
    Monetary  = ("Debit_Amount",  "sum"),
    First_Purchase = ("Date",     "min"),
    Last_Purchase  = ("Date",     "max"),
).reset_index()

rfm["Monetary"] = rfm["Monetary"].round(2)

# ── RFM SCORING (1–5 scale) ───────────────────────────────────────────────────
# Recency  : lower days = better = higher score
rfm["R_Score"] = pd.cut(
    rfm["Recency"],
    bins=[-1, 15, 45, 90, 180, 9999],
    labels=[5, 4, 3, 2, 1]
).astype(float)

# Frequency : higher count = better
rfm["F_Score"] = pd.cut(
    rfm["Frequency"],
    bins=[0, 3, 10, 30, 100, 9999],
    labels=[1, 2, 3, 4, 5]
).astype(float)

# Monetary  : higher spend = better (in rupees)
rfm["M_Score"] = pd.cut(
    rfm["Monetary"],
    bins=[0, 100000, 500000, 1000000, 5000000, 999999999],
    labels=[1, 2, 3, 4, 5]
).astype(float)

# Weighted RFM Score — Monetary weighted highest for B2B manufacturing
# R=30%, F=30%, M=40%
rfm["RFM_Score"] = (
    rfm["R_Score"] * 0.30 +
    rfm["F_Score"] * 0.30 +
    rfm["M_Score"] * 0.40
).round(2)

# ── SEGMENT ASSIGNMENT ────────────────────────────────────────────────────────
def assign_segment(row):
    r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
    score = row["RFM_Score"]

    if score >= 4.0:
        return "Champion"
    elif score >= 3.0 and r >= 3:
        return "Loyal"
    elif score >= 3.0 and r < 3:
        return "At Risk"
    elif score >= 2.0 and r >= 3:
        return "Promising"
    elif r <= 2 and m >= 3:
        return "At Risk"
    else:
        return "Lost"

rfm["Segment"] = rfm.apply(assign_segment, axis=1)

# ── ACTION RECOMMENDED ────────────────────────────────────────────────────────
ACTION_MAP = {
    "Champion"  : "Nurture — offer loyalty pricing or early access to new products",
    "Loyal"     : "Retain — regular check-ins, ensure satisfaction",
    "At Risk"   : "Re-engage — personal call from Bhuvanesh within 30 days",
    "Promising" : "Grow — increase order frequency with small incentives",
    "Lost"      : "Investigate — find out why they stopped ordering",
}
rfm["Recommended_Action"] = rfm["Segment"].map(ACTION_MAP)

# ── FORMAT FINAL OUTPUT ───────────────────────────────────────────────────────
rfm["Revenue_Lakhs"]   = (rfm["Monetary"] / 100000).round(2)
rfm["Last_Purchase"]   = rfm["Last_Purchase"].dt.strftime("%Y-%m-%d")
rfm["First_Purchase"]  = rfm["First_Purchase"].dt.strftime("%Y-%m-%d")

output_cols = [
    "Customer", "Segment", "RFM_Score",
    "R_Score", "F_Score", "M_Score",
    "Recency", "Frequency", "Monetary", "Revenue_Lakhs",
    "First_Purchase", "Last_Purchase",
    "Recommended_Action"
]

rfm_out = rfm[output_cols].sort_values("RFM_Score", ascending=False).reset_index(drop=True)

# ── SAVE CSV ──────────────────────────────────────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, "rfm_scores.csv")
rfm_out.to_csv(csv_path, index=False)
print(f"\n  rfm_scores.csv saved → {csv_path}")

# ── PRINT SUMMARY ─────────────────────────────────────────────────────────────
print("\n── Segment Summary ──────────────────────────────────────────")
summary = rfm_out.groupby("Segment").agg(
    Customers     = ("Customer",      "count"),
    Total_Revenue = ("Monetary",      "sum"),
    Avg_Recency   = ("Recency",       "mean"),
    Avg_Frequency = ("Frequency",     "mean"),
).reset_index()
summary["Revenue_Lakhs"] = (summary["Total_Revenue"] / 100000).round(1)
summary["Avg_Recency"]   = summary["Avg_Recency"].round(0).astype(int)
summary["Avg_Frequency"] = summary["Avg_Frequency"].round(1)
print(summary[["Segment","Customers","Revenue_Lakhs","Avg_Recency","Avg_Frequency"]].to_string(index=False))

print("\n── Top Customers by Segment ─────────────────────────────────")
for seg in ["Champion", "Loyal", "At Risk", "Promising", "Lost"]:
    seg_df = rfm_out[rfm_out["Segment"] == seg]
    if len(seg_df) == 0:
        continue
    print(f"\n  {seg} ({len(seg_df)} customers):")
    for _, row in seg_df.head(5).iterrows():
        print(f"    {row['Customer']:<45} ₹{row['Revenue_Lakhs']:>6.1f}L  "
              f"Last: {row['Last_Purchase']}  "
              f"RFM: {row['RFM_Score']}")

# ── CHART 1: RFM HEATMAP ─────────────────────────────────────────────────────
print("\nGenerating charts...")

SEGMENT_COLORS = {
    "Champion"  : "#1D9E75",
    "Loyal"     : "#378ADD",
    "At Risk"   : "#D85A30",
    "Promising" : "#BA7517",
    "Lost"      : "#888780",
}

heatmap_data = rfm_out.set_index("Customer")[["R_Score", "F_Score", "M_Score"]].copy()
heatmap_data.index = heatmap_data.index.str[:30]

seg_order = rfm_out.sort_values(["Segment", "RFM_Score"], ascending=[True, False])
seg_order.index = seg_order["Customer"].str[:30]

fig, ax = plt.subplots(figsize=(10, max(8, len(rfm_out) * 0.28)))

sns.heatmap(
    heatmap_data.loc[seg_order.index[::-1]],
    annot=True, fmt=".0f",
    cmap="RdYlGn",
    linewidths=0.5,
    cbar_kws={"label": "Score  (1 = Poor → 5 = Excellent)"},
    vmin=1, vmax=5,
    ax=ax
)

ax.set_title(
    "Customer RFM Scores — Challenge Packaging Industries",
    fontsize=13, fontweight="bold", pad=15
)
ax.set_xlabel("RFM Dimension", fontsize=10)
ax.set_ylabel("")
ax.set_xticklabels(
    ["Recency\n(recent = good)", "Frequency\n(often = good)", "Monetary\n(spend = good)"],
    fontsize=9
)
ax.tick_params(axis="y", labelsize=8)

# Colour-code y-axis labels by segment
for ytick in ax.get_yticklabels():
    name = ytick.get_text()
    match = seg_order[seg_order["Customer"].str[:30] == name]
    if not match.empty:
        seg = match["Segment"].iloc[0]
        ytick.set_color(SEGMENT_COLORS.get(seg, "#2C2C2A"))
        ytick.set_fontweight("bold")

# Legend
patches = [mpatches.Patch(color=c, label=s) for s, c in SEGMENT_COLORS.items()]
ax.legend(handles=patches, title="Segment", loc="lower right",
          bbox_to_anchor=(1.28, 0), fontsize=8, title_fontsize=8)

plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_DIR, "rfm_heatmap.png")
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  rfm_heatmap.png saved → {heatmap_path}")

# ── CHART 2: SEGMENT SUMMARY BAR ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Customer Segment Summary — Challenge Packaging Industries",
             fontsize=13, fontweight="bold")

seg_counts   = rfm_out["Segment"].value_counts()
seg_revenue  = rfm_out.groupby("Segment")["Revenue_Lakhs"].sum()

order = ["Champion", "Loyal", "Promising", "At Risk", "Lost"]
order_present = [s for s in order if s in seg_counts.index]
colors = [SEGMENT_COLORS[s] for s in order_present]

# Bar 1 — customer count
bars1 = ax1.bar(order_present, [seg_counts.get(s, 0) for s in order_present],
                color=colors, edgecolor="white", linewidth=0.5)
ax1.set_title("Customers per segment", fontsize=11)
ax1.set_ylabel("Number of customers")
ax1.set_ylim(0, max(seg_counts) * 1.2)
for bar, val in zip(bars1, [seg_counts.get(s, 0) for s in order_present]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")

# Bar 2 — revenue contribution
bars2 = ax2.bar(order_present, [seg_revenue.get(s, 0) for s in order_present],
                color=colors, edgecolor="white", linewidth=0.5)
ax2.set_title("Revenue per segment (₹ Lakhs)", fontsize=11)
ax2.set_ylabel("Revenue (₹L)")
for bar, val in zip(bars2, [seg_revenue.get(s, 0) for s in order_present]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"₹{val:.0f}L", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
summary_path = os.path.join(OUTPUT_DIR, "rfm_segment_summary.png")
plt.savefig(summary_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  rfm_segment_summary.png saved → {summary_path}")

print("\n✅ RFM segmentation complete!")
print(f"   CSV  → {csv_path}")
print(f"   PNG1 → {heatmap_path}")
print(f"   PNG2 → {summary_path}")