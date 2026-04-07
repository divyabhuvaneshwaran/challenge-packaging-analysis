##Customer segmentation##

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load data ──
df = pd.read_csv(r'C:\Users\divya\Documents\challenge-packaging-analysis\data\day_book_clean.csv')
df['Date'] = pd.to_datetime(df['Date'])

# ── Filter Sales only ──
sales = df[df['Txn_Category'] == 'Sales'].copy()

print("Total sales transactions:", len(sales))
print("Unique customers:", sales['Particulars'].nunique())

# ── Step 1: Calculate RFM for each customer ──
reference_date = sales['Date'].max()

rfm = sales.groupby('Particulars').agg(
    Recency   = ('Date', lambda x: (reference_date - x.max()).days),
    Frequency = ('Debit_Amount', 'count'),
    Monetary  = ('Debit_Amount', 'sum')
).reset_index()

rfm.columns = ['Customer', 'Recency', 'Frequency', 'Monetary']
rfm['Monetary'] = rfm['Monetary'].round(0)

print(rfm.sort_values('Monetary', ascending=False).head(10))

# ── Step 2: Score each customer 1-5 on each metric ──

# Recency: LOWER is better (bought recently = good)
rfm['R_Score'] = pd.cut(rfm['Recency'],
    bins=[-1, 10, 30, 60, 120, 999],
    labels=[5, 4, 3, 2, 1])

# Frequency: HIGHER is better
rfm['F_Score'] = pd.cut(rfm['Frequency'],
    bins=[0, 2, 5, 15, 50, 999],
    labels=[1, 2, 3, 4, 5])

# Monetary: HIGHER is better
rfm['M_Score'] = pd.cut(rfm['Monetary'],
    bins=[0, 50000, 200000, 500000, 1000000, 99999999],
    labels=[1, 2, 3, 4, 5])

# Convert to numbers
rfm['R_Score'] = rfm['R_Score'].astype(float)
rfm['F_Score'] = rfm['F_Score'].astype(float)
rfm['M_Score'] = rfm['M_Score'].astype(float)

# ── Step 3: Calculate final RFM score ──
# Monetary weighted highest (35%), Recency (35%), Frequency (30%)
rfm['RFM_Score'] = (rfm['R_Score'] * 0.35 +
                    rfm['F_Score'] * 0.30 +
                    rfm['M_Score'] * 0.35)

print(rfm[['Customer','R_Score','F_Score','M_Score','RFM_Score']]
      .sort_values('RFM_Score', ascending=False).head(10))

# ── Step 4: Assign Segments ──
def assign_segment(score):
    if score >= 4.5:
        return 'Champion'
    elif score >= 3.5:
        return 'Growing'
    elif score >= 2.5:
        return 'Needs Attention'
    else:
        return 'At Risk'

rfm['Segment'] = rfm['RFM_Score'].apply(assign_segment)

print("\n── Segment Summary ──")
print(rfm['Segment'].value_counts())

print("\n── At Risk Customers ──")
at_risk = rfm[rfm['Segment'] == 'At Risk']
print(at_risk[['Customer','Recency','Frequency','Monetary']].to_string())

# ── Step 5: Seaborn Heatmap ──
plt.figure(figsize=(10, 8))

# Prepare heatmap data
heatmap_data = rfm[['Customer','R_Score','F_Score','M_Score']].copy()
heatmap_data = heatmap_data.set_index('Customer')

# Shorten long names for display
heatmap_data.index = heatmap_data.index.str[:25]

sns.heatmap(heatmap_data,
            annot=True,
            fmt='.0f',
            cmap='RdYlGn',
            linewidths=0.5,
            cbar_kws={'label': 'Score (1=Poor, 5=Excellent)'},
            vmin=1, vmax=5)

plt.title('Customer RFM Scores — Challenge Packaging Industries',
          fontsize=13, fontweight='bold', pad=15)
plt.xlabel('RFM Dimension')
plt.ylabel('Customer')
plt.xticks(['R_Score','F_Score','M_Score'],
           ['Recency\n(recent=good)', 'Frequency\n(often=good)', 'Monetary\n(spend=good)'])
plt.tight_layout()
plt.savefig(r'C:\Users\divya\Documents\challenge-packaging-analysis\ml\rfm_heatmap.png', dpi=150)
plt.show()
print("\nHeatmap saved!")