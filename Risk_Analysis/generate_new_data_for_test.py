import numpy as np
import pandas as pd

# Simulated new data
new_data = {
    "debt_to_equity": np.random.uniform(0.5, 3.0, 20),
    "current_ratio": np.random.uniform(0.5, 5.0, 20),
    "altman_z_score": np.random.uniform(1.0, 4.5, 20),
    "revenue": np.random.uniform(1e6, 1e9, 20),
    "market_cap": np.random.uniform(5e6, 1e10, 20),
    "on_time_delivery": np.random.uniform(0.5, 1.0, 20),
    "order_volume": np.random.uniform(100, 10000, 20),
    "inflation_rate": np.random.uniform(0.5, 5.0, 20),
    "currency_volatility": np.random.uniform(0.5, 3.0, 20),
    "news_risk_score": np.random.uniform(0, 1.0, 20),
}

df_new_suppliers = pd.DataFrame(new_data)
df_new_suppliers.to_csv("new_supplier_data.csv", index=False)

print("\n New supplier data saved as 'new_supplier_data.csv'.")
