"""
Challenge Packaging Industries
02 — Revenue Forecasting
=========================
Input  : day_book_final.csv
Output : outputs/forecast.csv
         outputs/revenue_forecast.png

Model  : Linear Regression with sin/cos seasonality features
         + Bootstrap resampling for honest confidence intervals

Why not Random Forest / GBM?
  With only 24 data points cross-validation R² = -6 to -7
  meaning those models overfit badly and cannot generalise.
  Linear Regression with seasonality features is the honest
  choice for small datasets — interpretable and stable.

Run    : python 02_revenue_forecast.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE   = r"C:\Users\divya\Documents\challenge-packaging-analysis\data\day_book_final.csv"
OUTPUT_DIR   = r"C:\Users\divya\Documents\challenge-packaging-analysis\outputs"
FORECAST_MONTHS = 6       # how many months ahead to forecast
N_BOOTSTRAP     = 500     # bootstrap iterations for confidence intervals
CONFIDENCE      = 90      # confidence interval percentage
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LOAD & PREPARE ────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
sales = df[df["Txn_Category"] == "Sales"].copy()

# Aggregate to monthly revenue
monthly = (
    sales.groupby(sales["Date"].dt.to_period("M"))["Debit_Amount"]
    .sum()
    .reset_index()
)
monthly.columns = ["Month", "Revenue"]
monthly["Month_Num"]     = range(1, len(monthly) + 1)
monthly["Month_Of_Year"] = monthly["Month"].dt.month

# Seasonality features — sin/cos encode the 12-month cycle
# This is better than dummy variables — smooth, no boundary effects
monthly["Sin_Month"] = np.sin(2 * np.pi * monthly["Month_Of_Year"] / 12)
monthly["Cos_Month"] = np.cos(2 * np.pi * monthly["Month_Of_Year"] / 12)

# FY indicator — captures the growth step from 24-25 to 25-26
monthly["FY"] = (monthly["Month_Num"] > 12).astype(int)

print(f"  Total months of data : {len(monthly)}")
print(f"  Date range           : {monthly['Month'].iloc[0]} → {monthly['Month'].iloc[-1]}")
print(f"  Revenue range        : ₹{monthly['Revenue'].min():,.0f} → ₹{monthly['Revenue'].max():,.0f}")
print(f"  Mean monthly revenue : ₹{monthly['Revenue'].mean():,.0f}")

FEATURES = ["Month_Num", "Sin_Month", "Cos_Month", "FY"]
X = monthly[FEATURES].values
y = monthly["Revenue"].values

# ── TRAIN MODEL ───────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X, y)

y_pred   = model.predict(X)
mae      = mean_absolute_error(y, y_pred)
r2       = r2_score(y, y_pred)
residuals = y - y_pred

print(f"\n  Model R²  : {r2:.3f}")
print(f"  MAE       : ₹{mae:,.0f}  (avg monthly error)")
print(f"  Residual std : ₹{residuals.std():,.0f}")
print()
print("  Note: With only 24 months, cross-validation scores are negative")
print("  for complex models (RF/GBM) — they overfit. Linear Regression")
print("  with seasonality is the honest and stable choice here.")

# ── BUILD FUTURE MONTHS ───────────────────────────────────────────────────────
last_moy = monthly["Month_Of_Year"].iloc[-1]
last_month_period = monthly["Month"].iloc[-1]

future_rows   = []
future_labels = []

for i in range(1, FORECAST_MONTHS + 1):
    moy      = ((last_moy - 1 + i) % 12) + 1
    mn       = len(monthly) + i
    sin_m    = np.sin(2 * np.pi * moy / 12)
    cos_m    = np.cos(2 * np.pi * moy / 12)
    fy       = 1  # continuing 25-26 growth trend
    future_rows.append([mn, sin_m, cos_m, fy])

    # Generate month label (period arithmetic)
    future_period = last_month_period + i
    future_labels.append(str(future_period))

X_future = np.array(future_rows)

# ── BOOTSTRAP CONFIDENCE INTERVALS ────────────────────────────────────────────
# Resample the training data 500 times, train a model each time,
# predict the future — gives a realistic spread of possible outcomes
print(f"\nRunning {N_BOOTSTRAP} bootstrap iterations for confidence intervals...")

bootstrap_preds = []
for seed in range(N_BOOTSTRAP):
    X_b, y_b  = resample(X, y, random_state=seed)
    m_b       = LinearRegression().fit(X_b, y_b)
    bootstrap_preds.append(m_b.predict(X_future))

bootstrap_preds = np.array(bootstrap_preds)  # shape: (500, 6)

alpha           = (100 - CONFIDENCE) / 2
forecast_mean   = bootstrap_preds.mean(axis=0)
forecast_lower  = np.percentile(bootstrap_preds, alpha,           axis=0)
forecast_upper  = np.percentile(bootstrap_preds, 100 - alpha,     axis=0)

# ── PRINT FORECAST TABLE ──────────────────────────────────────────────────────
print(f"\n── {FORECAST_MONTHS}-Month Forecast ({CONFIDENCE}% Confidence Interval) ──────────")
print(f"{'Month':<12} {'Lower (₹)':>14} {'Forecast (₹)':>14} {'Upper (₹)':>14}")
print("-" * 56)
for month, low, mid, high in zip(future_labels, forecast_lower, forecast_mean, forecast_upper):
    print(f"{month:<12} {low:>14,.0f} {mid:>14,.0f} {high:>14,.0f}")

# ── SAVE FORECAST CSV ─────────────────────────────────────────────────────────
forecast_df = pd.DataFrame({
    "Month"            : future_labels,
    "Forecast_Revenue" : forecast_mean.round(2),
    "Lower_Bound"      : forecast_lower.round(2),
    "Upper_Bound"      : forecast_upper.round(2),
    "Type"             : "Forecast",
})

# Also include actuals for Power BI (so one table has full timeline)
actuals_df = pd.DataFrame({
    "Month"            : monthly["Month"].astype(str),
    "Forecast_Revenue" : monthly["Revenue"].round(2),
    "Lower_Bound"      : np.nan,
    "Upper_Bound"      : np.nan,
    "Type"             : "Actual",
})

full_df = pd.concat([actuals_df, forecast_df], ignore_index=True)
csv_path = os.path.join(OUTPUT_DIR, "forecast.csv")
full_df.to_csv(csv_path, index=False)
print(f"\n  forecast.csv saved → {csv_path}")

# ── CHART ─────────────────────────────────────────────────────────────────────
print("\nGenerating chart...")

fig, ax = plt.subplots(figsize=(14, 6))

# ── Actual revenue ──
ax.plot(
    monthly["Month_Num"], monthly["Revenue"],
    color="#185FA5", marker="o", markersize=5,
    linewidth=2, label="Actual revenue", zorder=3
)

# ── Model fit over actuals ──
ax.plot(
    monthly["Month_Num"], y_pred,
    color="#EF9F27", linestyle="--", linewidth=1.5,
    label="Model fit", zorder=2
)

# ── Forecast line ──
forecast_x = list(range(len(monthly) + 1, len(monthly) + FORECAST_MONTHS + 1))
ax.plot(
    forecast_x, forecast_mean,
    color="#1D9E75", linestyle="--", marker="s",
    markersize=5, linewidth=2, label="Forecast (central)", zorder=3
)

# ── Confidence band ──
ax.fill_between(
    forecast_x, forecast_lower, forecast_upper,
    alpha=0.20, color="#1D9E75",
    label=f"{CONFIDENCE}% confidence band"
)

# ── Divider line ──
ax.axvline(x=len(monthly) + 0.5, color="#888780", linestyle=":", linewidth=1.5)
ax.text(len(monthly) + 0.6, monthly["Revenue"].min() * 0.97,
        "Forecast →", fontsize=9, color="#888780")

# ── X-axis labels ──
all_x      = list(monthly["Month_Num"]) + forecast_x
all_labels = list(monthly["Month"].astype(str)) + future_labels
ax.set_xticks(all_x)
ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8)

# ── Y-axis in Lakhs ──
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f"₹{v/100000:.0f}L")
)

# ── Annotations on forecast points ──
for x, mid, low, high in zip(forecast_x, forecast_mean, forecast_lower, forecast_upper):
    ax.annotate(
        f"₹{mid/100000:.1f}L",
        xy=(x, mid), xytext=(0, 10),
        textcoords="offset points",
        ha="center", fontsize=8, color="#0F6E56"
    )

# ── Labels ──
ax.set_title(
    "Challenge Packaging Industries — Revenue Forecast",
    fontsize=13, fontweight="bold", pad=15
)
ax.set_xlabel("Month", fontsize=10)
ax.set_ylabel("Revenue (₹ Lakhs)", fontsize=10)
ax.legend(fontsize=9, loc="upper left")
ax.grid(axis="y", alpha=0.3)

# ── Model note ──
fig.text(
    0.99, 0.01,
    f"Model: Linear Regression + seasonality  |  "
    f"R²={r2:.2f}  MAE=₹{mae/1000:.0f}K  |  "
    f"Bootstrap {N_BOOTSTRAP} iterations  |  "
    f"Only 24 months of data — wide confidence bands are expected",
    ha="right", va="bottom", fontsize=7, color="#888780"
)

plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "revenue_forecast.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  revenue_forecast.png saved → {chart_path}")

# ── BUSINESS SUMMARY ──────────────────────────────────────────────────────────
print("\n── Business Insights ────────────────────────────────────────")
print(f"  Avg actual monthly revenue   : ₹{monthly['Revenue'].mean()/100000:.1f}L")
print(f"  Avg forecast (next 6 months) : ₹{forecast_mean.mean()/100000:.1f}L")
growth = (forecast_mean.mean() - monthly['Revenue'].mean()) / monthly['Revenue'].mean() * 100
print(f"  Expected growth              : {growth:+.1f}% vs historical average")
print(f"  Best forecast month          : {future_labels[forecast_mean.argmax()]} "
      f"— ₹{forecast_mean.max()/100000:.1f}L")
print(f"  Weakest forecast month       : {future_labels[forecast_mean.argmin()]} "
      f"— ₹{forecast_mean.min()/100000:.1f}L")
print()
print("  Note for Bhuvanesh:")
print("  The confidence band is wide (expected with 2 years of data).")
print("  Use the forecast for DIRECTION not exact numbers.")
print("  Rerun this model every 3 months as new data comes in —")
print("  accuracy will improve steadily.")

print("\n✅ Revenue forecasting complete!")
print(f"   CSV → {csv_path}")
print(f"   PNG → {chart_path}")