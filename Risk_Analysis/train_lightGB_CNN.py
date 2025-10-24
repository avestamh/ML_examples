import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# 📌 **Enable GPU if available**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 📌 **Load Data**
df_financials = pd.read_csv("financial_data.csv")
df_suppliers = pd.read_csv("supplier_data.csv")

# 📌 **Merge Financial & Supplier Data**
df = pd.merge(df_suppliers, df_financials, how="left", left_on="supplier", right_on="symbol")

# **Select Features**
features_cnn = ["debt_to_equity", "current_ratio", "altman_z_score", "inflation_rate", "currency_volatility", "market_cap"]
X = df[features_cnn].values
y = df["bankruptcy_risk"].values  # 0 = stable, 1 = bankrupt

# **Scale Features**
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# **Prepare Data for CNN (reshape into time series)**
num_quarters = 6
X_cnn = np.tile(X_scaled[:, np.newaxis, :], (1, num_quarters, 1))

# **Ensure Stratified Sampling (Fix Class Imbalance Issue)**
X_train_cnn, X_test_cnn, y_train, y_test = train_test_split(
    X_cnn, y, test_size=0.2, random_state=42, stratify=y
)

# **Convert to PyTorch Tensors & Move to GPU**
X_train_cnn = torch.tensor(X_train_cnn, dtype=torch.float32).to(device)
X_test_cnn = torch.tensor(X_test_cnn, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# ------------------------ 📌 **Define CNN Model (on GPU)** ------------------------

class CNNBankruptcyModel(nn.Module):
    def __init__(self):
        super(CNNBankruptcyModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=len(features_cnn), out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * num_quarters, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape for Conv1D (batch, channels, time_steps)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return x

# **Initialize CNN Model on GPU**
cnn_model = CNNBankruptcyModel().to(device)

# **Train CNN**
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
criterion = nn.MSELoss()  

num_epochs = 10
for epoch in range(num_epochs):
    cnn_model.train()
    optimizer.zero_grad()
    embeddings = cnn_model(X_train_cnn)  # Get embeddings
    loss = criterion(embeddings, embeddings)  # Dummy loss (Autoencoder-like)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# ------------------------ 📌 **Extract CNN Features for LightGBM** ------------------------

cnn_model.eval()
with torch.no_grad():
    train_embeddings = cnn_model(X_train_cnn).cpu().numpy()
    test_embeddings = cnn_model(X_test_cnn).cpu().numpy()

# **Prepare Final Dataset for LightGBM**
feature_names = [f"cnn_feature_{i}" for i in range(train_embeddings.shape[1])]
X_train_lgb = np.hstack((X_train_cnn[:, 0, :].cpu().numpy(), train_embeddings))
X_test_lgb = np.hstack((X_test_cnn[:, 0, :].cpu().numpy(), test_embeddings))

X_train_lgb_df = pd.DataFrame(X_train_lgb, columns=feature_names)
X_test_lgb_df = pd.DataFrame(X_test_lgb, columns=feature_names)

# ------------------------ 📌 **Train LightGBM Model** ------------------------

lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
lgb_model.fit(X_train_lgb_df, y_train.cpu().numpy())

# **Evaluate LightGBM**
y_pred_lgb = lgb_model.predict(X_test_lgb_df)
accuracy = accuracy_score(y_test.cpu().numpy(), y_pred_lgb)

# **Fix ROC-AUC Issue (Avoid `nan`)**
if len(set(y_test.cpu().numpy().flatten())) > 1:
    roc_auc = roc_auc_score(y_test.cpu().numpy(), y_pred_lgb)
else:
    roc_auc = float("nan")

print(f"✅ Hybrid Model (CNN + LightGBM) Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
