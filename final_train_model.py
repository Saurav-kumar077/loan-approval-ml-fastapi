import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
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
        test_df[col] = test_df[col].fillna(mode_val)

# Drop ID column
train_df.drop("Loan_ID", axis=1, inplace=True)
test_df.drop("Loan_ID", axis=1, inplace=True)

# Label Encoding
le = LabelEncoder()
for col in train_df.columns:
    if col != "Loan_Status" and train_df[col].dtype == "object":
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col]  = le.transform(test_df[col])

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


model.fit(X, y)

joblib.dump(model, "loan_xgboost_model.pkl")
joblib.dump(FINAL_THRESHOLD, "loan_threshold.pkl")

print(" Final model trained on full data")
print(" Model saved as loan_xgboost_model.pkl")
print(" Threshold saved as loan_threshold.pkl")
print(" Final Threshold:", FINAL_THRESHOLD)
