import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from cnn_model import RiskCNN
from sklearn.preprocessing import StandardScaler

# Load new supplier data
df_new = pd.read_csv("new_supplier_data.csv")

# Load final model type
with open("final_model_type.txt", "r") as f:
    final_model = f.read().strip()

# Preprocess new data
features = ["debt_to_equity", "current_ratio", "altman_z_score", "revenue", "market_cap",
            "on_time_delivery", "order_volume", "inflation_rate", "currency_volatility", "news_risk_score"]
X_new = df_new[features]

# Normalize using the same scaler
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

# **CNN Prediction**
if final_model == "CNN":
    print("🔍 Loading CNN model for prediction...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = RiskCNN(input_dim=len(features)).to(device)
    cnn_model.load_state_dict(torch.load("final_model.pth"))
    cnn_model.eval()

    X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred_proba = torch.sigmoid(cnn_model(X_new_tensor)).cpu().numpy()
    
    y_pred = (y_pred_proba > 0.3).astype(int)  # Apply threshold

# **LightGBM Prediction**
elif final_model == "LightGBM":
    print("🔍 Loading LightGBM model for prediction...")
    lgb_model = lgb.Booster(model_file="final_model.txt")
    y_pred_proba = lgb_model.predict(X_new_scaled)
    y_pred = (y_pred_proba > 0.3).astype(int)  # Apply threshold

else:
    raise ValueError("❌ No valid final model found!")

# Save new predictions
df_new["bankruptcy_risk"] = y_pred
df_new.to_csv("new_supplier_predictions.csv", index=False)

print("\n✅ Predictions saved to 'new_supplier_predictions.csv'.")
