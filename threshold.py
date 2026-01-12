import joblib
threshold = 0.65
joblib.dump(threshold,"loan_threshold.pkl")
print("loan_threshold.pkl saved")