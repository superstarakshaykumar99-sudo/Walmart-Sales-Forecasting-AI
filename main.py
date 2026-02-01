import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import datetime, timedelta

# 1. Dummy Data Generate Karna
np.random.seed(42)
dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(100)]
df = pd.DataFrame({
    'Date': dates,
    'Store': np.random.randint(1, 5, 100),
    'Holiday_Flag': np.random.choice([0, 1], 100, p=[0.8, 0.2]),
    'Weekly_Sales': np.random.randint(15000, 45000, 100)
})

# 2. AI Prediction Logic
df['Month'] = df['Date'].dt.month
X = df[['Store', 'Holiday_Flag', 'Month']]
y = df['Weekly_Sales']

model = XGBRegressor()
model.fit(X, y)
df['Predicted_Sales'] = model.predict(X)

# 3. CSV Save Karna
df.to_csv('Walmart_Final_Project.csv', index=False)
print("Project File 'Walmart_Final_Project.csv' taiyar hai! ✅")