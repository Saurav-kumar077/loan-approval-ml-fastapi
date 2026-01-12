from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

import os

app = FastAPI(title="Loan Approval Prediction API")




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "loan_xgboost_model.pkl")
threshold_path = os.path.join(BASE_DIR,"loan_threshold.pkl")

model = joblib.load(model_path)
THRESHOLD = joblib.load(threshold_path)




# Input schema
class LoanInput(BaseModel):
    Gender: int
    Married: int
    Dependents: int
    Education: int
    Self_Employed: int
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: int
    Property_Area: int

@app.get("/")
def home():
    return {"message": "Loan Approval API running"}

@app.post("/predict")
def predict_loan(data: LoanInput):
    features = np.array([[
        data.Gender,
        data.Married,
        data.Dependents,
        data.Education,
        data.Self_Employed,
        data.ApplicantIncome,
        data.CoapplicantIncome,
        data.LoanAmount,
        data.Loan_Amount_Term,
        data.Credit_History,
        data.Property_Area
    ]])

    approval_prob = model.predict_proba(features)[0][1]

    decision = "Approved" if approval_prob >= THRESHOLD else "Rejected"

    return {
    "loan_status": decision,
    "approval_probability": round(float(approval_prob), 3),
    "threshold_used": THRESHOLD
}

