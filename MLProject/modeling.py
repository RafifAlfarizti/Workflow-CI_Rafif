# modelling.py

import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature

# Load Processed Data
X_train = np.load("heart_preprocessing/X_train.npy")
X_test = np.load("heart_preprocessing/X_test.npy")
y_train = np.load("heart_preprocessing/y_train.npy")
y_test = np.load("heart_preprocessing/y_test.npy")

# MLflow Autolog (optional)
mlflow.sklearn.autolog()

with mlflow.start_run():
    # Train Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predict
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc}")
    
    # Create signature & input example
    signature = infer_signature(X_test, preds)
    input_example = X_test[0:1]
    
    # Manual log with signature & input example
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="logreg_model",
        signature=signature,
        input_example=input_example
    )