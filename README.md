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


## 🛠 Environment Setup

- Python 3.9 or higher
- Recommended to use a virtual environment

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```



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


## 📊 Prediction Logic

- The model outputs a probability score between 0 and 1.
- A configurable threshold is applied to determine loan approval.
- If probability ≥ threshold → **Approved**
- If probability < threshold → **Rejected**



## ⚠️ Input Validation & Error Handling

- Request validation is handled using **Pydantic**.
- Invalid or missing fields return a `422 Unprocessable Entity` error.
- Ensures safe and consistent API behavior.


## 🚀 Future Improvements

- Dockerize the application
- Deploy API to cloud (Render / Railway / AWS)
- Add CI/CD using GitHub Actions
- Improve model performance with hyperparameter tuning
- Add authentication and rate limiting


## 🌟 Project Highlights

- End-to-end ML project from data preprocessing to API deployment
- XGBoost-based loan approval classifier with probability scoring
- FastAPI backend with validation and testing
- Clean project structure and production-ready codebase


## 🧪 Testing

Automated tests are written using **pytest** to validate the model and API behavior.

Run tests locally:

```bash
pip install -r requirements.txt
pytest test_model.py
```

## 📄 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this project for personal and commercial purposes.




