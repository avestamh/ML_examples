import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Generate mock supplier financial data
num_suppliers = 100

supplier_names = [f"Supplier_{i}" for i in range(1, num_suppliers + 1)]
duns_numbers = [f"DUNS-{100000 + i}" for i in range(1, num_suppliers + 1)]
regions = np.random.choice(["North America", "Europe", "Asia", "South America"], num_suppliers)
countries = np.random.choice(["USA", "Germany", "China", "Mexico"], num_suppliers)

# Financial data
debt_to_equity = np.random.normal(1.5, 0.5, num_suppliers)
current_ratio = np.random.normal(1.2, 0.3, num_suppliers)
altman_z_score = np.random.normal(2.5, 0.7, num_suppliers)
revenue = np.random.randint(50_000_000, 5_000_000_000, num_suppliers)
market_cap = np.random.randint(100_000_000, 10_000_000_000, num_suppliers)

# Risk indicators
inflation_rate = np.random.normal(3.0, 1.5, num_suppliers)
currency_volatility = np.random.normal(0.02, 0.01, num_suppliers)
news_risk_score = np.random.uniform(0.1, 0.7, num_suppliers)

# Bankruptcy risk
bankruptcy_risk = np.random.choice([0, 1], num_suppliers, p=[0.85, 0.15])

df = pd.DataFrame({
    "supplier_name": supplier_names,
    "duns_number": duns_numbers,
    "region": regions,
    "country": countries,
    "debt_to_equity": debt_to_equity,
    "current_ratio": current_ratio,
    "altman_z_score": altman_z_score,
    "revenue": revenue,
    "market_cap": market_cap,
    "inflation_rate": inflation_rate,
    "currency_volatility": currency_volatility,
    "news_risk_score": news_risk_score,
    "bankruptcy_risk": bankruptcy_risk
})

df.to_csv("supplier_financial_data.csv", index=False)
print("Mock data generated.")
