import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load saved model & threshold
model = joblib.load("loan_xgboost_model.pkl")
threshold = joblib.load("loan_threshold.pkl")

# Load data
df = pd.read_csv("train.csv")


# Numeric → median
for col in df.select_dtypes(include=["int64", "float64"]).columns:
    df[col] = df[col].fillna(df[col].median())

# Categorical → mode
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Drop ID
df.drop("Loan_ID", axis=1, inplace=True)

# Label Encoding (IMPORTANT)
le = LabelEncoder()
for col in df.columns:
    if col != "Loan_Status" and df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# Encode target
df["Loan_Status"] = df["Loan_Status"].map({"N": 0, "Y": 1})

X = df.drop("Loan_Status", axis=1)

# Take one sample
sample = X.iloc[[0]]

proba = model.predict_proba(sample)[:, 1]
prediction = (proba >= threshold).astype(int)

print("Predicted Probability:", proba[0])
print("Final Prediction (1=Approve, 0=Reject):", prediction[0])
