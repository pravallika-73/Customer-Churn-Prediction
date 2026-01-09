import joblib
import mlflow
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

print("🚀 Churn Prediction - Model Training Starting...")

# 1. Load processed data created by prepare.py
X_train, X_test, y_train, y_test = joblib.load("models/data.pkl")

# 2. Define base models
xgb_model = xgb.XGBClassifier(
    max_depth=6,
    n_estimators=200,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss",
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# 3. Define stacking ensemble (enterprise-style)
stack_model = StackingClassifier(
    estimators=[("xgb", xgb_model), ("rf", rf_model)],
    final_estimator=LogisticRegression(max_iter=1000),
    n_jobs=-1,
    passthrough=False,
)

# 4. Train with MLflow tracking
mlflow.set_experiment("TCS_Customer_Churn")

with mlflow.start_run(run_name="stacking_xgb_rf"):
    stack_model.fit(X_train, y_train)

    # Predictions
    y_proba = stack_model.predict_proba(X_test)[:, 1]
    y_pred = stack_model.predict(X_test)

    auc = roc_auc_score(y_test, y_proba)
    print(f"AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred))

    # Log metric and model
    mlflow.log_metric("AUC", auc)
    mlflow.sklearn.log_model(stack_model, "stacking_model")

    # Save to disk for API
    joblib.dump(stack_model, "models/churn_model.pkl")

print("✅ MODEL READY! Saved to models/churn_model.pkl")