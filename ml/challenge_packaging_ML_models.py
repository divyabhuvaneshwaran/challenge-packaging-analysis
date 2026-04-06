# ============================================================
# CHALLENGE PACKAGING INDUSTRIES
# Machine Learning Models | DayBook 2024-26
# Author: Divya | Resume Project
# ============================================================

import pandas as pd
import numpy as np
import sqlite3
import warnings
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df_db = pd.read_excel('DayBook_24-25_DIV__1_.xlsx', sheet_name='Day Book', header=7)
df_db.columns = ['Date','Particulars','Vch_Type','Vch_No','Debit_Amount','Credit_Amount']
df_db = df_db[2:].reset_index(drop=True)
df_db['Date'] = pd.to_datetime(df_db['Date'], errors='coerce')
df_db = df_db.dropna(subset=['Date'])
df_db['Debit_Amount'] = pd.to_numeric(df_db['Debit_Amount'], errors='coerce').fillna(0)
df_db['Credit_Amount'] = pd.to_numeric(df_db['Credit_Amount'], errors='coerce').fillna(0)

sales = df_db[df_db['Vch_Type'].str.contains('Sales', case=False, na=False)].copy()
purchases = df_db[df_db['Vch_Type'].str.contains('Purchase', case=False, na=False)].copy()


# ═══════════════════════════════════════════════════════════════
# MODEL 1: MONTHLY REVENUE FORECASTING
# Business Story: "When should we prepare for high-demand months?"
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("MODEL 1: REVENUE FORECASTING")
print("="*60)

monthly = sales.groupby(sales['Date'].dt.to_period('M'))['Debit_Amount'].sum().reset_index()
monthly.columns = ['Period', 'Revenue']
monthly['Month_Num']    = range(1, len(monthly)+1)
monthly['Month_Of_Year'] = monthly['Period'].dt.month
monthly['Sin_Month']     = np.sin(2 * np.pi * monthly['Month_Of_Year'] / 12)
monthly['Cos_Month']     = np.cos(2 * np.pi * monthly['Month_Of_Year'] / 12)
monthly['Quarter']       = monthly['Month_Of_Year'].apply(
    lambda m: 1 if m in [4,5,6] else 2 if m in [7,8,9] else 3 if m in [10,11,12] else 4)

X = monthly[['Month_Num','Sin_Month','Cos_Month','Quarter']].values
y = monthly['Revenue'].values

# Train models
model_lr  = LinearRegression()
model_rf  = RandomForestRegressor(n_estimators=200, random_state=42)
model_gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

for name, model in [('Linear Regression', model_lr), ('Random Forest', model_rf), ('Gradient Boosting', model_gbm)]:
    cv_scores = cross_val_score(model, X, y, cv=min(5, len(X)//2), scoring='r2')
    model.fit(X, y)
    pred = model.predict(X)
    print(f"  {name}: R²={r2_score(y,pred):.3f} | MAE=₹{mean_absolute_error(y,pred):,.0f} | CV-R²={cv_scores.mean():.3f}")

# Forecast next 6 months
last_moy = monthly['Month_Of_Year'].iloc[-1]
future_rows = []
for i in range(1, 7):
    moy = ((last_moy - 1 + i) % 12) + 1
    q   = 1 if moy in [4,5,6] else 2 if moy in [7,8,9] else 3 if moy in [10,11,12] else 4
    future_rows.append([len(monthly)+i, moy, np.sin(2*np.pi*moy/12), np.cos(2*np.pi*moy/12), q])

X_future = np.array(future_rows)
print("\n  FORECAST (next 6 months):")
for i, row in enumerate(X_future):
    lr_p  = model_lr.predict([row])[0]
    rf_p  = model_rf.predict([row])[0]
    gbm_p = model_gbm.predict([row])[0]
    ensemble = (lr_p + rf_p + gbm_p) / 3
    print(f"  Month+{i+1}: Ensemble Forecast = ₹{ensemble:,.0f}")


# ═══════════════════════════════════════════════════════════════
# MODEL 2: CUSTOMER RFM SEGMENTATION
# Business Story: "Who are our Champions vs At-Risk customers?"
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("MODEL 2: CUSTOMER RFM SEGMENTATION")
print("="*60)

ref_date = sales['Date'].max()

rfm = sales.groupby('Particulars').agg(
    Recency   = ('Date', lambda x: (ref_date - x.max()).days),
    Frequency = ('Debit_Amount', 'count'),
    Monetary  = ('Debit_Amount', 'sum')
).reset_index()
rfm.columns = ['Customer', 'Recency', 'Frequency', 'Monetary']

# Normalize
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])

# Score: invert recency (lower = more recent = better)
rfm['R_Score'] = 1 - (rfm['Recency'] - rfm['Recency'].min()) / (rfm['Recency'].max() - rfm['Recency'].min() + 1)
rfm['F_Score'] = (rfm['Frequency'] - rfm['Frequency'].min()) / (rfm['Frequency'].max() - rfm['Frequency'].min() + 1)
rfm['M_Score'] = (rfm['Monetary'] - rfm['Monetary'].min()) / (rfm['Monetary'].max() - rfm['Monetary'].min() + 1)
rfm['RFM_Score'] = rfm['R_Score']*0.35 + rfm['F_Score']*0.25 + rfm['M_Score']*0.40

rfm['Segment'] = pd.cut(rfm['RFM_Score'],
    bins=[0, 0.30, 0.60, 1.01],
    labels=['🔴 At Risk', '🟡 Growing', '🟢 Champion'])

print(rfm.sort_values('RFM_Score', ascending=False)[[
    'Customer','Recency','Frequency','Monetary','RFM_Score','Segment']].to_string(index=False))

print("\n  Segment Distribution:")
print(rfm['Segment'].value_counts().to_string())

# Business insight
at_risk = rfm[rfm['Segment'] == '🔴 At Risk']
print(f"\n  ⚠️  {len(at_risk)} customers are AT RISK. Recommended action: personal outreach within 30 days.")
print(f"  Top at-risk customers:")
print(at_risk.sort_values('Monetary', ascending=False)[['Customer','Recency','Monetary']].head(5).to_string())


# ═══════════════════════════════════════════════════════════════
# MODEL 3: ANOMALY DETECTION (Isolation Forest)
# Business Story: "Which transactions need a second look?"
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("MODEL 3: ANOMALY DETECTION")
print("="*60)

sales_pos = sales[sales['Debit_Amount'] > 0].copy()
sales_pos['Month'] = sales_pos['Date'].dt.month
sales_pos['DayOfWeek'] = sales_pos['Date'].dt.dayofweek

features = sales_pos[['Debit_Amount', 'Month', 'DayOfWeek']].values
iso_forest = IsolationForest(contamination=0.05, random_state=42)
sales_pos['Anomaly'] = iso_forest.fit_predict(features)
sales_pos['Anomaly_Score'] = iso_forest.score_samples(features)

anomalies = sales_pos[sales_pos['Anomaly'] == -1].sort_values('Debit_Amount', ascending=False)
print(f"  Total Sales Transactions: {len(sales_pos)}")
print(f"  Detected Anomalies: {len(anomalies)} ({len(anomalies)/len(sales_pos)*100:.1f}%)")
print("\n  Top Anomalous Transactions:")
print(anomalies[['Date','Particulars','Debit_Amount','Anomaly_Score']].head(10).to_string(index=False))


# ═══════════════════════════════════════════════════════════════
# MODEL 4: GROSS MARGIN PREDICTION
# Business Story: "Can we predict next month's margin?"
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("MODEL 4: GROSS MARGIN PREDICTION")
print("="*60)

monthly_sales    = sales.groupby(sales['Date'].dt.to_period('M'))['Debit_Amount'].sum()
monthly_purchase = purchases.groupby(purchases['Date'].dt.to_period('M'))['Credit_Amount'].sum()
margin_df = pd.DataFrame({'Sales': monthly_sales, 'Purchase': monthly_purchase}).dropna()
margin_df['Gross_Margin']  = margin_df['Sales'] - margin_df['Purchase']
margin_df['Margin_Pct']    = (margin_df['Gross_Margin'] / margin_df['Sales'] * 100).round(2)
margin_df['Month_Num']     = range(1, len(margin_df)+1)
margin_df['Month_Of_Year'] = margin_df.index.month

Xm = margin_df[['Month_Num','Sales','Month_Of_Year']].values
ym = margin_df['Margin_Pct'].values

model_margin = Ridge(alpha=1.0)
model_margin.fit(Xm, ym)
pred_margin = model_margin.predict(Xm)

print(f"  Margin Model R²: {r2_score(ym, pred_margin):.3f}")
print(f"  Average Gross Margin: {ym.mean():.1f}%")
print(f"  Best Month: {margin_df['Margin_Pct'].idxmax()} = {margin_df['Margin_Pct'].max():.1f}%")
print(f"  Worst Month: {margin_df['Margin_Pct'].idxmin()} = {margin_df['Margin_Pct'].min():.1f}%")
print("\n  Monthly Margin Table:")
print(margin_df[['Sales','Purchase','Gross_Margin','Margin_Pct']].to_string())


# ═══════════════════════════════════════════════════════════════
# MODEL 5: PAYMENT DELAY RISK CLASSIFIER
# Business Story: "Which customers might delay payment?"
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("MODEL 5: PAYMENT BEHAVIOR ANALYSIS")
print("="*60)

# Compare invoice date to receipt date per customer
receipts = df_db[df_db['Vch_Type'].str.contains('Receipt', case=False, na=False)].copy()

customer_sales    = sales.groupby('Particulars')['Debit_Amount'].agg(['sum','count']).rename(columns={'sum':'Total_Sales','count':'Num_Invoices'})
customer_receipts = receipts.groupby('Particulars')['Credit_Amount'].sum().rename('Total_Received')

payment_df = pd.merge(customer_sales, customer_receipts, left_index=True, right_index=True, how='left')
payment_df['Total_Received'] = payment_df['Total_Received'].fillna(0)
payment_df['Outstanding']    = payment_df['Total_Sales'] - payment_df['Total_Received']
payment_df['Collection_Rate'] = (payment_df['Total_Received'] / payment_df['Total_Sales'] * 100).round(1)
payment_df['Risk_Flag'] = payment_df['Collection_Rate'].apply(
    lambda x: 'High Risk' if x < 50 else 'Medium Risk' if x < 80 else 'Low Risk')

print("  Customer Payment Behavior:")
print(payment_df.sort_values('Outstanding', ascending=False)[[
    'Total_Sales','Total_Received','Outstanding','Collection_Rate','Risk_Flag']].head(12).to_string())

print("\n  Risk Summary:")
print(payment_df['Risk_Flag'].value_counts().to_string())

# Total outstanding
print(f"\n  Total Outstanding (all customers): ₹{payment_df['Outstanding'].sum():,.0f}")
print("\n✅ All 5 ML models complete!")
