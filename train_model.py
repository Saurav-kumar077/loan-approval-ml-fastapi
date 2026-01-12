import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier


FINAL_THRESHOLD = 0.3
RANDOM_STATE = 42


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")



# Numeric → median
for col in train_df.select_dtypes(include=["int64", "float64"]).columns:
    median_val = train_df[col].median()
    train_df[col] = train_df[col].fillna(median_val)
    test_df[col]  = test_df[col].fillna(median_val)

# Categorical → mode
for col in train_df.select_dtypes(include=["object"]).columns:
    mode_val = train_df[col].mode()[0]
    train_df[col] = train_df[col].fillna(mode_val)
    if col != "Loan_Status":
        test_df[col]=test_df[col].fillna(mode_val)

# Drop ID column
train_df.drop("Loan_ID", axis=1, inplace=True)
test_df.drop("Loan_ID", axis=1, inplace=True)

# Label Encoding
le = LabelEncoder()
for col in train_df.columns:
    if col != "Loan_Status" and train_df[col].dtype == "object":
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])


X = train_df.drop("Loan_Status", axis=1)
y = train_df["Loan_Status"].map({"N": 0, "Y": 1})

pos = (y == 1).sum()
neg = (y == 0).sum()
scale_pos_weight = neg / pos

model = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

acc_scores, prec_scores, rec_scores, f1_scores, roc_auc_scores = [], [], [], [], []

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model.fit(X_train, y_train)

    # Probabilities
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # FINAL threshold
    y_pred = (y_pred_proba >= FINAL_THRESHOLD).astype(int)

    # Metrics
    acc_scores.append(accuracy_score(y_val, y_pred))
    prec_scores.append(precision_score(y_val, y_pred))
    rec_scores.append(recall_score(y_val, y_pred))
    f1_scores.append(f1_score(y_val, y_pred))
    roc_auc_scores.append(roc_auc_score(y_val, y_pred_proba))

print("\nFINAL CROSS-VALIDATION RESULTS (Threshold = 0.3)\n")
print("Average Accuracy :", np.mean(acc_scores))
print("Average Precision:", np.mean(prec_scores))
print("Average Recall   :", np.mean(rec_scores))
print("Average F1 Score :", np.mean(f1_scores))
print("Average ROC-AUC  :", np.mean(roc_auc_scores))
