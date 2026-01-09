# Telco Customer Churn Prediction

End-to-end ML project to predict telecom customer churn using Python, scikit-learn/XGBoost, FastAPI, MLflow, and Docker.

## Project structure

- src/data/prepare.py: load Telco data, clean, engineer features, encode, split, scale, save.
- src/features/build_features.py: reusable feature engineering functions.
- src/models/train.py: train model, evaluate, log to MLflow, save models/churn_model.pkl.
- src/api/app.py: FastAPI service exposing /predict endpoint.
- src/visualization/shap_summary.py: generates reports/shap_summary.png for feature importance.
- mlruns/, mlflow.db: MLflow tracking.
- Dockerfile: container for API + model.

## How to run

pip install -r requirements.txt

python -m src.data.prepare
python -m src.models.train
uvicorn src.api.app:app --reload

optional explainability
python -m src.visualization.shap_summary



## Example request:

POST /predict
{
"gender": "Female",
"SeniorCitizen": 0,
"Partner": "Yes",
"Dependents": "No",
"tenure": 12,
"PhoneService": "Yes",
"InternetService": "Fiber optic",
"Contract": "Month-to-month",
"MonthlyCharges": 75.5,
"TotalCharges": 900.0
}


undefined

## Run with Docker

docker build -t churn-api .
docker run --rm -p 8000:8000 churn-api