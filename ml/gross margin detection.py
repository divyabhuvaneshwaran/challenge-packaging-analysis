"""
Challenge Packaging Industries
04 — Gross Margin Prediction
==============================
Predicts Gross Margin % from Sales and Purchase amounts.

Key finding from data exploration:
  Purchase cost is the PRIMARY driver of margin (correlation = -0.694)
  Sales + Purchase together → R² = 0.972, MAE = 2.07%
  Seasonality alone → R² = 0.084 (useless)

Business meaning:
  This model tells Bhuvanesh one thing clearly —
  MARGIN IS CONTROLLED BY PURCHASE COST, NOT REVENUE.
  Months where purchase spiked (Sep-24: 98.7%, Jan-26: 115.8%)
  are the months margin collapsed. Control purchases = control margin.

Input  : day_book_final.csv
Output : outputs/margin_predictions.csv
         outputs/gross_margin.png

Run    : python 04_gross_margin_prediction.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE = r"C:\Users\divya\Documents\challenge-packaging-analysis\data\day_book_final.csv"
OUTPUT_DIR = r"C:\Users\divya\Documents\challenge-packaging-analysis\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Margin health thresholds — confirmed with business context
MARGIN_HEALTHY  = 35   # above this = good month
MARGIN_WARNING  = 20   # below this = concerning
MARGIN_CRITICAL = 0    # below this = losing money

# ── LOAD & PREPARE ────────────────────────────────────────────────────────────
print("Loading data...")
df        = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
sales     = df[df["Txn_Category"] == "Sales"]
purchases = df[df["Txn_Category"] == "Purchase"]

ms = sales.groupby(sales["Date"].dt.to_period("M"))["Debit_Amount"].sum().reset_index()
mp = purchases.groupby(purchases["Date"].dt.to_period("M"))["Credit_Amount"].sum().reset_index()
ms.columns = ["Month", "Sales"]
mp.columns = ["Month", "Purchase"]

m = ms.merge(mp, on="Month")
m["Gross_Margin"]   = m["Sales"] - m["Purchase"]
m["Margin_Pct"]     = (m["Gross_Margin"] / m["Sales"] * 100).round(2)
m["Cost_Ratio"]     = (m["Purchase"] / m["Sales"] * 100).round(2)
m["Month_Num"]      = range(1, len(m) + 1)
m["Month_Of_Year"]  = m["Month"].dt.month
m["FY"]             = (m["Month_Num"] > 12).astype(int)
m["FY_Label"]       = m["FY"].map({0: "2024-25", 1: "2025-26"})
m["Month_Str"]      = m["Month"].astype(str)

print(f"  Total months     : {len(m)}")
print(f"  24-25 avg margin : {m[m['FY']==0]['Margin_Pct'].mean():.1f}%")
print(f"  25-26 avg margin : {m[m['FY']==1]['Margin_Pct'].mean():.1f}%")
print(f"  Best month       : {m.loc[m['Margin_Pct'].idxmax(), 'Month_Str']} "
      f"— {m['Margin_Pct'].max():.1f}%")
print(f"  Worst month      : {m.loc[m['Margin_Pct'].idxmin(), 'Month_Str']} "
      f"— {m['Margin_Pct'].min():.1f}%")


# ── TRAIN MODEL ───────────────────────────────────────────────────────────────
# Sales + Purchase → Margin%  (R²=0.972 confirmed in exploration)
# Ridge regression to avoid overfitting on 24 points
FEATURES = ["Sales", "Purchase"]
X = m[FEATURES].values
y = m["Margin_Pct"].values

model     = Ridge(alpha=1.0).fit(X, y)
y_pred    = model.predict(X)
r2        = r2_score(y, y_pred)
mae       = mean_absolute_error(y, y_pred)
residuals = y - y_pred

m["Margin_Predicted"] = y_pred.round(2)
m["Residual"]         = residuals.round(2)
m["Prediction_Error"] = abs(residuals).round(2)

print(f"\n  Model R²  : {r2:.3f}")
print(f"  MAE       : {mae:.2f}%  (avg prediction error)")
print(f"  Max error : {abs(residuals).max():.2f}%")
print()
print("  Coefficients:")
print(f"    Sales coefficient    : {model.coef_[0]:.6f}")
print(f"    Purchase coefficient : {model.coef_[1]:.6f}")
print(f"    Intercept            : {model.intercept_:.2f}")
print()
print("  Business translation:")
print(f"    Every Rs.1L increase in monthly Sales   → "
      f"+{model.coef_[0]*100000:.2f}% margin")
print(f"    Every Rs.1L increase in monthly Purchase → "
      f"{model.coef_[1]*100000:.2f}% margin")


# ── MARGIN HEALTH FLAGS ───────────────────────────────────────────────────────
def margin_health(pct):
    if pct >= MARGIN_HEALTHY:
        return "Healthy"
    elif pct >= MARGIN_WARNING:
        return "Warning"
    elif pct >= MARGIN_CRITICAL:
        return "Critical"
    else:
        return "Loss"

m["Health_Status"]           = m["Margin_Pct"].apply(margin_health)
m["Predicted_Health_Status"] = m["Margin_Predicted"].apply(margin_health)


# ── WHAT-IF ANALYSIS ──────────────────────────────────────────────────────────
# Answer: "What purchase % would give us 30% margin?"
# Margin% = f(Sales, Purchase)
# Rearranging: what Purchase amount gives target margin?
avg_sales = m["Sales"].mean()
print("\n── What-If: Purchase cost scenarios at avg monthly sales "
      f"(Rs.{avg_sales/100000:.1f}L) ──")
print(f"{'Target Margin%':>15} {'Max Purchase (Rs.)':>20} {'Cost Ratio%':>13}")
print("-" * 52)
for target in [40, 35, 30, 25, 20, 15]:
    # Solve: target = coef[0]*sales + coef[1]*purchase + intercept
    # purchase = (target - coef[0]*sales - intercept) / coef[1]
    max_pur = (target - model.coef_[0] * avg_sales - model.intercept_) / model.coef_[1]
    cost_ratio = max_pur / avg_sales * 100
    print(f"{target:>14}%  {max_pur:>19,.0f}  {cost_ratio:>12.1f}%")


# ── SAVE CSV ──────────────────────────────────────────────────────────────────
output_cols = [
    "Month_Str", "FY_Label", "Sales", "Purchase",
    "Gross_Margin", "Margin_Pct", "Margin_Predicted",
    "Cost_Ratio", "Residual", "Health_Status",
]
out_df   = m[output_cols].copy()
out_df.columns = [
    "Month", "FY", "Sales", "Purchase",
    "Gross_Margin", "Actual_Margin_%", "Predicted_Margin_%",
    "Cost_Ratio_%", "Prediction_Error_%", "Health_Status",
]
csv_path = os.path.join(OUTPUT_DIR, "margin_predictions.csv")
out_df.to_csv(csv_path, index=False)
print(f"\n  margin_predictions.csv saved → {csv_path}")


# ── CHART ─────────────────────────────────────────────────────────────────────
print("\nGenerating chart...")

HEALTH_COLORS = {
    "Healthy"  : "#1D9E75",
    "Warning"  : "#BA7517",
    "Critical" : "#D85A30",
    "Loss"     : "#A32D2D",
}

fig = plt.figure(figsize=(16, 13))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :])   # Margin% actual vs predicted — full width
ax2 = fig.add_subplot(gs[1, :])   # Sales vs Purchase waterfall — full width
ax3 = fig.add_subplot(gs[2, 0])   # Health status bar
ax4 = fig.add_subplot(gs[2, 1])   # Purchase vs Margin scatter

x     = m["Month_Num"]
xlabs = m["Month_Str"]

# ── Chart 1: Actual vs predicted margin% ─────────────────────────────────────
ax1.plot(x, m["Margin_Pct"], color="#185FA5", marker="o",
         markersize=5, linewidth=2, label="Actual Margin%", zorder=3)
ax1.plot(x, m["Margin_Predicted"], color="#EF9F27", linestyle="--",
         linewidth=1.5, label=f"Predicted Margin% (R²={r2:.3f})", zorder=2)

# Health threshold bands
ax1.axhspan(MARGIN_HEALTHY, 100, alpha=0.06, color="#1D9E75", label="Healthy zone (>35%)")
ax1.axhspan(MARGIN_WARNING, MARGIN_HEALTHY, alpha=0.06, color="#BA7517", label="Warning zone (20–35%)")
ax1.axhspan(MARGIN_CRITICAL, MARGIN_WARNING, alpha=0.06, color="#D85A30", label="Critical zone (0–20%)")
ax1.axhspan(-30, MARGIN_CRITICAL, alpha=0.08, color="#A32D2D", label="Loss zone (<0%)")

# Annotate worst months
worst = m.nsmallest(3, "Margin_Pct")
for _, row in worst.iterrows():
    ax1.annotate(
        f"{row['Margin_Pct']:.1f}%",
        xy=(row["Month_Num"], row["Margin_Pct"]),
        xytext=(0, -18), textcoords="offset points",
        ha="center", fontsize=8, color="#A32D2D",
        arrowprops=dict(arrowstyle="-", color="#A32D2D", lw=0.8)
    )

ax1.set_xticks(x)
ax1.set_xticklabels(xlabs, rotation=45, ha="right", fontsize=7)
ax1.set_title("Gross Margin % — Actual vs Predicted", fontsize=11, fontweight="bold")
ax1.set_ylabel("Gross Margin %")
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax1.legend(fontsize=8, loc="upper left", ncol=2)
ax1.grid(axis="y", alpha=0.25)
ax1.axhline(y=0, color="#A32D2D", linewidth=0.8, linestyle="-")

# ── Chart 2: Sales vs Purchase bar chart ─────────────────────────────────────
width = 0.38
bars_s = ax2.bar(x - width/2, m["Sales"]/100000, width,
                 label="Sales", color="#185FA5", alpha=0.85)
bars_p = ax2.bar(x + width/2, m["Purchase"]/100000, width,
                 label="Purchase", color="#D85A30", alpha=0.85)

# Colour code months where purchase > sales
for i, (_, row) in enumerate(m.iterrows()):
    if row["Purchase"] >= row["Sales"]:
        bars_p[i].set_color("#A32D2D")
        bars_p[i].set_alpha(1.0)

ax2.set_xticks(x)
ax2.set_xticklabels(xlabs, rotation=45, ha="right", fontsize=7)
ax2.set_title("Monthly Sales vs Purchase (dark red = purchase exceeded sales)",
              fontsize=11, fontweight="bold")
ax2.set_ylabel("Amount (₹L)")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"₹{v:.0f}L"))
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.25)

# ── Chart 3: Health status distribution ──────────────────────────────────────
health_counts = m["Health_Status"].value_counts()
order         = ["Healthy", "Warning", "Critical", "Loss"]
order_present = [s for s in order if s in health_counts.index]
colors3       = [HEALTH_COLORS[s] for s in order_present]

bars3 = ax3.bar(order_present,
                [health_counts.get(s, 0) for s in order_present],
                color=colors3, edgecolor="white", linewidth=0.5)
ax3.set_title("Months by margin health", fontsize=10, fontweight="bold")
ax3.set_ylabel("Number of months")
for bar, status in zip(bars3, order_present):
    count = health_counts.get(status, 0)
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.1,
             str(count), ha="center", fontsize=11, fontweight="bold")
ax3.set_ylim(0, max(health_counts.values) * 1.25)
ax3.grid(axis="y", alpha=0.25)

# ── Chart 4: Purchase vs Margin scatter ──────────────────────────────────────
scatter_colors = [HEALTH_COLORS[s] for s in m["Health_Status"]]
ax4.scatter(m["Purchase"]/100000, m["Margin_Pct"],
            c=scatter_colors, s=60, alpha=0.85, zorder=3)

# Regression line
xfit = np.linspace(m["Purchase"].min(), m["Purchase"].max(), 100)
# Simple linear for visual
from numpy.polynomial import polynomial as P
coefs = np.polyfit(m["Purchase"]/100000, m["Margin_Pct"], 1)
ax4.plot(xfit/100000, np.polyval(coefs, xfit/100000),
         color="#888780", linestyle="--", linewidth=1.5, alpha=0.7)

ax4.axhline(y=0, color="#A32D2D", linewidth=0.8)
ax4.set_title("Purchase cost vs Margin% relationship",
              fontsize=10, fontweight="bold")
ax4.set_xlabel("Monthly Purchase (₹L)")
ax4.set_ylabel("Gross Margin %")
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"₹{v:.0f}L"))
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

# Label extreme months
for _, row in m.nsmallest(2, "Margin_Pct").iterrows():
    ax4.annotate(row["Month_Str"],
                 xy=(row["Purchase"]/100000, row["Margin_Pct"]),
                 xytext=(5, 5), textcoords="offset points",
                 fontsize=7, color="#A32D2D")
ax4.grid(alpha=0.25)

# Legend for scatter
from matplotlib.patches import Patch
legend_els = [Patch(color=HEALTH_COLORS[s], label=s) for s in order_present]
ax4.legend(handles=legend_els, fontsize=8, loc="upper right")

fig.suptitle(
    "Challenge Packaging Industries — Gross Margin Analysis & Prediction",
    fontsize=13, fontweight="bold", y=1.01
)

chart_path = os.path.join(OUTPUT_DIR, "gross_margin.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  gross_margin.png saved → {chart_path}")


# ── BUSINESS SUMMARY ──────────────────────────────────────────────────────────
print("\n── Business Insights ────────────────────────────────────────")
print(f"  Overall avg margin    : {m['Margin_Pct'].mean():.1f}%")
print(f"  24-25 avg margin      : {m[m['FY']==0]['Margin_Pct'].mean():.1f}%")
print(f"  25-26 avg margin      : {m[m['FY']==1]['Margin_Pct'].mean():.1f}%")
print(f"  Margin declining YoY  : {m[m['FY']==0]['Margin_Pct'].mean() - m[m['FY']==1]['Margin_Pct'].mean():.1f}pp drop")
print()
hc = m["Health_Status"].value_counts()
print(f"  Healthy months  (>35%): {hc.get('Healthy', 0)} out of 24")
print(f"  Warning months (20-35%): {hc.get('Warning', 0)} out of 24")
print(f"  Critical months (0-20%): {hc.get('Critical', 0)} out of 24")
print(f"  Loss months      (<0%) : {hc.get('Loss', 0)} out of 24")
print()
print("  The single most important lever:")
print(f"  Every Rs.1L reduction in monthly purchase cost")
print(f"  → {abs(model.coef_[1])*100000:.2f}% improvement in gross margin")
print()
print("  To consistently achieve 30% margin, monthly purchase")
print(f"  must stay below Rs.{((30 - model.coef_[0]*avg_sales - model.intercept_)/model.coef_[1])/100000:.1f}L")
print(f"  (current avg purchase: Rs.{m['Purchase'].mean()/100000:.1f}L)")

print("\n✅ Gross margin prediction complete!")
print(f"   CSV → {csv_path}")
print(f"   PNG → {chart_path}")