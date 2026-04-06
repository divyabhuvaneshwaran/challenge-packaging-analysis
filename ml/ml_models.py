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
