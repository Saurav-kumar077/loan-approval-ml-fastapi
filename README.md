# Loan Approval Prediction API (FastAPI + ML)

A Loan Approval Prediction API built with FastAPI and a trained XGBoost machine learning model to classify loan applications as approved or rejected. This project includes training scripts, model artifacts, threshold logic, automated tests, and a REST API to serve predictions. Ideal for learning end-to-end ML + backend deployment workflows.

---

## 🚀 Features

- Train a loan approval classification model using applicant financial data  
- Predict loan approval with probability scoring  
- REST API built using FastAPI  
- Includes training scripts and saved model artifacts (.pkl)  
- Automated testing using pytest  
- Threshold logic for probability-based decisions  

---

## 📂 Repository Structure

```text
loan-approval-ml-fastapi/
├── app.py
├── train_model.py
├── final_train_model.py
├── threshold.py
├── loan_xgboost_model.pkl
├── loan_threshold.pkl
├── train.csv
├── test.csv
├── test_model.py
├── requirements.txt
├── __pycache__/
└── .gitignore 
```


---

## 🧠 Model Training

The model is trained using an XGBoost classifier on historical loan application data.

Run training locally:

```bash
pip install -r requirements.txt
python train_model.py
```
---

## ⚡ Run the FastAPI Server

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```
Open API docs in browser:

http://localhost:8000/docs


## 🧾 API Endpoint

### POST `/predict`

Example request:

```json
{
  "age": 35,
  "income": 5600,
  "loan_amount": 120000,
  "credit_score": 690,
  "existing_loans_count": 1,
  "employment_years": 3,
  "dependents": 1,
  "education_level": "Graduate",
  "property_area": "Urban"
}
```

Example response:
```json
{
  "approved": true,
  "approval_probability": 0.78,
  "message": "Loan likely approved"
}
```



