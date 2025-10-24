import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_lightgbm(X_train, y_train, X_test, y_test):
    scale_pos_weight = np.sqrt(y_train.value_counts()[0] / y_train.value_counts()[1])

    params = {
         'objective': 'binary',
         'metric': 'binary_logloss',
         'boosting_type': 'gbdt',
         'num_leaves': 64,  # Increased from 50 to allow more complex splits
         'learning_rate': 0.02,
         'min_data_in_leaf': 5,  # Reduced from 10 to prevent missing splits
         'max_depth': -1,  # Allow full tree growth
         'feature_fraction': 0.9,  # Allow slightly more features per tree
         'bagging_fraction': 0.9,  # More bagging to improve stability
         'scale_pos_weight': scale_pos_weight,  # Balance class weights
         'verbose': -1  # Suppresses warnings
        
    }

    lgb_train = lgb.Dataset(X_train, y_train)
    model = lgb.train(params, lgb_train, num_boost_round=100)

    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.3).astype(int)


    model.save_model("lightgbm_model.txt")
