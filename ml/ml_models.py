import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("All libraries loaded!")

##Load the data##
df = pd.read_csv(r'C:\Users\divya\Documents\challenge-packaging-analysis\data\day_book_clean.csv')

##Clean the date column##
df['Date'] = pd.to_datetime(df['Date'])

# Check it loaded correctly
print("Shape:", df.shape)
print("Date range:", df['Date'].min(), "to", df['Date'].max())
print(df.head(3))

# ── Filter only Sales transactions ──
sales = df[df['Txn_Category'] == 'Sales'].copy()

# ── Group by Month ──
sales['Date'] = pd.to_datetime(sales['Date'])
monthly = sales.groupby(sales['Date'].dt.to_period('M'))['Debit_Amount'].sum().reset_index()
monthly.columns = ['Month', 'Revenue']

# ── Add a simple month number (1, 2, 3...) ──
monthly['Month_Num'] = range(1, len(monthly) + 1)

print(monthly)
print("\nTotal months of data:", len(monthly))

# ── Prepare X (input) and y (output) ──
X = monthly[['Month_Num']].values  # input: month number
y = monthly['Revenue'].values       # output: revenue

# ── Create and train the model ──
model = LinearRegression()
model.fit(X, y)

# ── Check how accurate it is ──
y_predicted = model.predict(X)
mae = mean_absolute_error(y, y_predicted)
r2 = r2_score(y, y_predicted)

print(f"R² Score: {r2:.3f}")
print(f"MAE: ₹{mae:,.0f}")

# ── Upgrade: add seasonality features ──
monthly['Month_Of_Year'] = monthly['Month'].dt.month

# Sin and Cos capture the seasonal cycle
monthly['Sin_Month'] = np.sin(2 * np.pi * monthly['Month_Of_Year'] / 12)
monthly['Cos_Month'] = np.cos(2 * np.pi * monthly['Month_Of_Year'] / 12)

# ── Retrain with more features ──
X_upgraded = monthly[['Month_Num', 'Sin_Month', 'Cos_Month']].values

model_v2 = LinearRegression()
model_v2.fit(X_upgraded, y)

y_predicted_v2 = model_v2.predict(X_upgraded)
mae_v2 = mean_absolute_error(y, y_predicted_v2)
r2_v2 = r2_score(y, y_predicted_v2)

print(f"V1 → R²: 0.447  MAE: ₹1,48,478")
print(f"V2 → R²: {r2_v2:.3f}  MAE: ₹{mae_v2:,.0f}")
print(f"Improvement: {((148478 - mae_v2)/148478*100):.1f}% better!")

# ── Predict next 6 months ──
future_months = []
for i in range(1, 7):
    month_num = len(monthly) + i
    month_of_year = ((monthly['Month_Of_Year'].iloc[-1] - 1 + i) % 12) + 1
    sin_m = np.sin(2 * np.pi * month_of_year / 12)
    cos_m = np.cos(2 * np.pi * month_of_year / 12)
    future_months.append([month_num, sin_m, cos_m])

X_future = np.array(future_months)
forecast = model_v2.predict(X_future)

print("\n── 6 Month Forecast ──")
for i, rev in enumerate(forecast):
    print(f"  Month +{i+1}: ₹{rev:,.0f}")

# ── Draw the chart ──
plt.figure(figsize=(12, 6))

# Actual revenue line
plt.plot(monthly['Month_Num'], monthly['Revenue'],
         color='steelblue', marker='o', linewidth=2, label='Actual Revenue')

# Predicted line over actual months
plt.plot(monthly['Month_Num'], y_predicted_v2,
         color='orange', linestyle='--', linewidth=2, label='Model Fit')

# Forecast line
forecast_months = list(range(len(monthly) + 1, len(monthly) + 7))
plt.plot(forecast_months, forecast,
         color='green', linestyle='--', marker='s',
         linewidth=2, label='Forecast')

# Vertical line separating actual vs forecast
plt.axvline(x=len(monthly), color='gray', linestyle=':', linewidth=1.5)
plt.text(len(monthly) + 0.1, min(y), '← Forecast starts', fontsize=9, color='gray')

# Labels
plt.title('Challenge Packaging — Revenue Forecast', fontsize=14, fontweight='bold')
plt.xlabel('Month Number')
plt.ylabel('Revenue (₹)')
plt.xticks(monthly['Month_Num'], monthly['Month'].astype(str), rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(r'C:\Users\divya\Documents\challenge-packaging-analysis\ml\revenue_forecast.png', dpi=150)
plt.show()

print("\nChart saved!")

# ── MODEL V3: Adding Attendance Data ──
print("\n── Model V3: With Attendance Feature ──")

attendance_data = {
    'Month_Num':        [1,2,3,4,5,6,7,8,9,10,11,12],
    'Month_Of_Year':    [4,5,6,7,8,9,10,11,12,1,2,3],
    'Avg_Days_Present': [21.1,24.8,27.4,29.1,28.1,28.6,
                         29.0,26.6,28.4,28.6,26.6,26.7]
}
monthly_v3 = pd.DataFrame(attendance_data)
monthly_v3['Sin_Month'] = np.sin(2*np.pi*monthly_v3['Month_Of_Year']/12)
monthly_v3['Cos_Month'] = np.cos(2*np.pi*monthly_v3['Month_Of_Year']/12)
monthly_v3['Revenue'] = y

X_v3 = monthly_v3[['Month_Num','Sin_Month',
                    'Cos_Month','Avg_Days_Present']].values
model_v3 = LinearRegression()
model_v3.fit(X_v3, y)

y_pred_v3 = model_v3.predict(X_v3)
print(f"V2 → R²: 0.604  MAE: ₹1,39,104")
print(f"V3 → R²: {r2_score(y, y_pred_v3):.3f}  "
      f"MAE: ₹{mean_absolute_error(y, y_pred_v3):,.0f}")
print("New feature added: Avg worker attendance days per month")
print("Business insight: April low revenue explained by low attendance (21.1 days)")