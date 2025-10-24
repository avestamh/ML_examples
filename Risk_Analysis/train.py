from cnn_model import RiskCNN
from lstm_model import RiskLSTM
import numpy as np
import pandas as pd
import shap
import torch
import torch.optim as optim
import torch.nn as nn
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE




# ✅ GPU Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
df = pd.read_csv("supplier_financial_data.csv")
features = ["debt_to_equity", "current_ratio", "altman_z_score", "revenue", "market_cap",
            "on_time_delivery", "order_volume", "inflation_rate", "currency_volatility", "news_risk_score"]
X = df[features]
y = df["bankruptcy_risk"]

# ✅ Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Handle Class Imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_shap_summary_and_bar(shap_values, X_test, model_name):
    """Generates a SHAP summary beeswarm plot and a mean absolute SHAP bar plot side by side."""

    # 🔹 Handle LightGBM SHAP Output (Ensure correct class is selected)
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values = shap_values[1]  # Select SHAP values for class 1 (bankruptcy risk)

    shap_values = np.array(shap_values)  # Convert to NumPy array
    mean_shap = np.abs(shap_values).mean(axis=0)  # Compute mean absolute SHAP values

    # ✅ Debugging: Ensure feature size matches SHAP values
    if mean_shap.shape[0] != len(features):
        print(f"❌ SHAP values do not match the number of features! Expected {len(features)}, but got {mean_shap.shape[0]}")
        return  # Skip plotting if there's a mismatch

    sorted_indices = np.argsort(mean_shap)[::-1]  # Sort in descending order
    sorted_feature_names = np.array(features)[sorted_indices]  # Ensure correct indexing

    # 🔹 Step 1: Generate and Save SHAP Summary Plot
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
    plt.title(f"SHAP Summary Plot - {model_name}")
    plt.savefig(f"shap_{model_name}_summary.png", bbox_inches="tight")
    plt.close()  # ✅ Ensure the figure is cleared before moving to the next plot

    # 🔹 Step 2: Generate and Save SHAP Bar Plot
    plt.figure(figsize=(8, 6))  # 🔥 FIX: Create a separate figure for the bar plot
    plt.barh(sorted_feature_names[::-1], mean_shap[sorted_indices][::-1], color="crimson")  # Reverse for better display
    plt.xlabel("Mean |SHAP Value|")
    plt.ylabel("Features")
    plt.title(f"Mean SHAP Value - {model_name}")
    plt.tight_layout()
    plt.savefig(f"shap_{model_name}_summary_bar.png")  # ✅ Save correctly
    plt.close()  # ✅ FIX: Close the figure before `plt.show()`

#     print(f"✅ SHAP Summary and Bar Plots saved for {model_name}.")

# def plot_shap_summary_and_bar_combined(shap_values, X_test, model_name):
#     """Generates a properly aligned SHAP summary plot and SHAP mean absolute bar plot side by side."""
    
#     # 🔹 Fix for LightGBM SHAP Output (Binary Classification Case)
#     if isinstance(shap_values, list) and len(shap_values) > 1:
#         shap_values = shap_values[1]  # Use SHAP values for class 1

#     shap_values = np.array(shap_values)  # Convert to NumPy array
#     mean_shap = np.abs(shap_values).mean(axis=0)  # Compute mean absolute SHAP values

#     # ✅ Ensure feature size matches SHAP values
#     if mean_shap.shape[0] != len(features):
#         print(f"❌ SHAP values do not match the number of features! Expected {len(features)}, but got {mean_shap.shape[0]}")
#         return

#     sorted_indices = np.argsort(mean_shap)[::-1]  # Sort in descending order
#     sorted_feature_names = np.array(features)[sorted_indices]  # Ensure correct indexing

#     # ✅ Create a single figure with 2 properly aligned subplots
#     fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]})  # Left wider than right

#     # 🔹 Left: SHAP Summary Beeswarm Plot
#     plt.sca(ax[0])  # ✅ Set current axis to left subplot
#     shap.summary_plot(shap_values, X_test, feature_names=features, show=False)  
#     ax[0].set_title(f"SHAP Summary Plot - {model_name}")

#     # 🔹 Right: SHAP Mean Feature Importance Bar Plot
#     ax[1].barh(sorted_feature_names[::-1], mean_shap[sorted_indices][::-1], color="crimson")  
#     ax[1].set_xlabel("Mean |SHAP Value|")
#     ax[1].set_ylabel("Features")
#     ax[1].set_title(f"Mean SHAP Value - {model_name}")

#     plt.tight_layout()
#     plt.savefig(f"shap_{model_name}_combined.png", bbox_inches="tight")  # ✅ Save figure
#     plt.show()  # ✅ Show combined figure


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
### =================== CNN Model Training =================== ###
cnn_model = RiskCNN(input_dim=len(features)).to(device)

# ✅ Class Weighting for Loss
pos_weight = torch.tensor([y_train.value_counts()[0] / y_train.value_counts()[1]]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(cnn_model.parameters(), lr=0.0005)

# ✅ Convert Data to Torch Tensors
X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_resampled.values, dtype=torch.float32).view(-1, 1).to(device)

for epoch in range(100):
    cnn_model.train()
    optimizer.zero_grad()
    outputs = cnn_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# ✅ CNN Model Evaluation
cnn_model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred_proba_cnn = torch.sigmoid(cnn_model(X_test_tensor)).cpu().numpy()

### =================== LightGBM Model Training =================== ###
lgb_train = lgb.Dataset(X_train_resampled, y_train_resampled)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 50,
    'learning_rate': 0.02,
    'min_data_in_leaf': 10,
    'max_depth': 6,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'scale_pos_weight': np.sqrt(y_train.value_counts()[0] / y_train.value_counts()[1]),
    'verbose': -1
}

lgb_model = lgb.train(params, lgb_train, num_boost_round=100)

y_pred_proba_lgb = lgb_model.predict(X_test)
y_pred_lgb = (y_pred_proba_lgb > 0.3).astype(int)

### =================== LSTM Model Training =================== ###
lstm_model = RiskLSTM(input_dim=len(features), hidden_dim=64, num_layers=2).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(lstm_model.parameters(), lr=0.0005)

X_train_tensor_lstm = torch.tensor(X_train_resampled, dtype=torch.float32).view(-1, 1, len(features)).to(device)
y_train_tensor_lstm = torch.tensor(y_train_resampled.values, dtype=torch.float32).view(-1, 1).to(device)

for epoch in range(100):
    lstm_model.train()
    optimizer.zero_grad()
    outputs = lstm_model(X_train_tensor_lstm)
    loss = criterion(outputs, y_train_tensor_lstm)
    loss.backward()
    optimizer.step()

# ✅ LSTM Model Evaluation
lstm_model.eval()
X_test_tensor_lstm = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, len(features)).to(device)

with torch.no_grad():
    y_pred_proba_lstm = torch.sigmoid(lstm_model(X_test_tensor_lstm)).cpu().numpy()

### =================== Final Model Selection =================== ###
cnn_f1 = f1_score(y_test, (y_pred_proba_cnn > 0.3).astype(int))
lgb_f1 = f1_score(y_test, y_pred_lgb)
lstm_f1 = f1_score(y_test, (y_pred_proba_lstm > 0.3).astype(int))

# ✅ Determine Best Model
best_model = None
best_f1 = max(cnn_f1, lgb_f1, lstm_f1)

if best_f1 == cnn_f1:
    best_model = "CNN"
    torch.save(cnn_model.state_dict(), "final_model_cnn.pth")
elif best_f1 == lgb_f1:
    best_model = "LightGBM"
    lgb_model.save_model("final_model_lgb.txt")
else:
    best_model = "LSTM"
    torch.save(lstm_model.state_dict(), "final_model_lstm.pth")

# ✅ Save Model Type
with open("final_model_type.txt", "w") as f:
    f.write(best_model)

### =================== SHAP Analysis for Best Model =================== ###
# if best_model == "CNN":
#     explainer = shap.KernelExplainer(lambda x: torch.sigmoid(cnn_model(torch.tensor(x, dtype=torch.float32).to(device))).cpu().detach().numpy(), X_train[:100])
#     shap_values = explainer.shap_values(X_test[:100])
# elif best_model == "LightGBM":
#     explainer = shap.TreeExplainer(lgb_model)
#     shap_values = explainer.shap_values(X_test)
# elif best_model == "LSTM":
#     explainer = shap.KernelExplainer(lambda x: torch.sigmoid(lstm_model(torch.tensor(x, dtype=torch.float32).view(-1, 1, len(features)).to(device))).cpu().detach().numpy(), X_train[:100])
#     shap_values = explainer.shap_values(X_test[:100])

### =================== SHAP Analysis for Best Model =================== ###
if best_model == "CNN":
    explainer = shap.KernelExplainer(lambda x: torch.sigmoid(cnn_model(torch.tensor(x, dtype=torch.float32).to(device))).cpu().detach().numpy(), X_train[:100])
    shap_values = explainer.shap_values(X_test[:100])

elif best_model == "LightGBM":
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_test)
    
    # ✅ Fix: Ensure SHAP values for LightGBM are extracted correctly
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values = shap_values[1]  # Take SHAP values for the bankruptcy risk class
    else:
        shap_values = np.array(shap_values)  # Convert to array if not already

elif best_model == "LSTM":
    explainer = shap.KernelExplainer(lambda x: torch.sigmoid(lstm_model(torch.tensor(x, dtype=torch.float32).view(-1, 1, len(features)).to(device))).cpu().detach().numpy(), X_train[:100])
    shap_values = explainer.shap_values(X_test[:100])

##____________________________________________________________
# # ✅ Plot and Save SHAP Summary Plot
# plt.figure(figsize=(8, 6))
# shap.summary_plot(shap_values, X_test[:100], feature_names=features, show=False)  # ✅ Avoid showing for automation
# shap_filename = f"shap_{best_model.lower()}.png"
# plt.savefig(shap_filename, bbox_inches="tight")
# print(f"✅ SHAP plot saved as {shap_filename}")

# print(f"✅ Final Model: {best_model} Saved Successfully!")

##__________________________________________________________________
# ✅ Generate and Save SHAP Summary + Bar Plot
# plot_shap_summary_and_bar(shap_values, X_test[:100], best_model)
# plot_shap_summary_and_bar(shap_values, X_test[:100], best_model)
plot_shap_summary_and_bar_combined(shap_values, X_test[:100], best_model)

print(f"✅ SHAP Summary and Bar Plots saved for {best_model}.")

