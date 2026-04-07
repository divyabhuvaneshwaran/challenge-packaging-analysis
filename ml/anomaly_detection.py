
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# ── Load data ──
df = pd.read_csv(r'C:\Users\divya\Documents\challenge-packaging-analysis\data\day_book_clean.csv')
df['Date'] = pd.to_datetime(df['Date'])

# ── Filter Sales only ──
sales = df[df['Txn_Category'] == 'Sales'].copy()
sales = sales[sales['Debit_Amount'] > 0].copy()

print("Sales transactions to scan:", len(sales))
print("Average transaction: ₹", round(sales['Debit_Amount'].mean(), 0))
print("Max transaction: ₹", round(sales['Debit_Amount'].max(), 0))
print("Min transaction: ₹", round(sales['Debit_Amount'].min(), 0))

# ── Prepare features for the model ──
sales['Month'] = sales['Date'].dt.month
sales['Day_of_Week'] = sales['Date'].dt.dayofweek  # 0=Monday, 6=Sunday

features = sales[['Debit_Amount', 'Month', 'Day_of_Week']].values

# ── Train Isolation Forest ──
model = IsolationForest(
    contamination=0.05,  # expect 5% of transactions to be anomalies
    random_state=42
)
model.fit(features)

# ── Predict: -1 = anomaly, 1 = normal ──
sales['Anomaly'] = model.predict(features)
sales['Anomaly_Score'] = model.score_samples(features)

# ── Separate anomalies ──
anomalies = sales[sales['Anomaly'] == -1].copy()
normal = sales[sales['Anomaly'] == 1].copy()

print(f"Total transactions scanned: {len(sales)}")
print(f"Normal transactions: {len(normal)}")
print(f"Anomalies detected: {len(anomalies)}")
print(f"\n── Top 10 Suspicious Transactions ──")
print(anomalies[['Date','Particulars','Debit_Amount','Anomaly_Score']]
      .sort_values('Debit_Amount', ascending=False)
      .head(10)
      .to_string())

# ── Draw the Anomaly Chart ──
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# ── Chart 1: Scatter plot — normal vs anomaly ──
ax1.scatter(normal['Date'], normal['Debit_Amount'],
            c='steelblue', alpha=0.5, s=20, label='Normal')
ax1.scatter(anomalies['Date'], anomalies['Debit_Amount'],
            c='red', alpha=0.8, s=50, marker='x', label='Anomaly')

ax1.set_title('Transaction Anomaly Detection — Challenge Packaging',
              fontsize=13, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Transaction Amount (₹)')
ax1.legend()
ax1.axhline(y=sales['Debit_Amount'].mean(),
            color='orange', linestyle='--',
            linewidth=1, label='Average')

# Annotate top 3 anomalies
top3 = anomalies.sort_values('Debit_Amount', ascending=False).head(3)
for _, row in top3.iterrows():
    ax1.annotate(f"₹{row['Debit_Amount']:,.0f}",
                xy=(row['Date'], row['Debit_Amount']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=8, color='red')

# ── Chart 2: Distribution — normal vs anomaly ──
ax2.hist(normal['Debit_Amount'], bins=50,
         color='steelblue', alpha=0.7, label='Normal transactions')
ax2.hist(anomalies['Debit_Amount'], bins=20,
         color='red', alpha=0.7, label='Anomalies')

ax2.set_title('Amount Distribution — Normal vs Anomalous',
              fontsize=13, fontweight='bold')
ax2.set_xlabel('Transaction Amount (₹)')
ax2.set_ylabel('Number of Transactions')
ax2.legend()

plt.tight_layout()
plt.savefig(r'C:\Users\divya\Documents\challenge-packaging-analysis\ml\anomaly_detection.png', dpi=150)
plt.show()
print("Chart saved!")

# ── Business Summary ──
print("\n── Business Action Items ──")
print(f"1. Review {len(anomalies)} flagged transactions manually")
print(f"2. Fix customer name: 'NEST PACCAGING\\t\\t\\t\\n' has dirty data")
print(f"3. Daeseung Autoparts bulk orders are legitimate but worth credit-checking")
print(f"4. Anomaly rate: {len(anomalies)/len(sales)*100:.1f}% of all transactions")