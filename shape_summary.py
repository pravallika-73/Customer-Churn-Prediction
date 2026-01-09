import joblib
import shap
import matplotlib.pyplot as plt

def main():
    # 1. Load processed data and model
    X_train, X_test, y_train, y_test = joblib.load("models/data.pkl")
    model = joblib.load("models/churn_model.pkl")

    # 2. Take a small sample
    X_sample = X_test[:200]

    # 3. If model is a stacking classifier, take the XGBoost base estimator
    try:
        # adjust the key/name if your base estimator name is different
        xgb_model = model.named_estimators_["xgb"]
    except AttributeError:
        # if it is already a single tree model, just use it directly
        xgb_model = model

    # 4. Use TreeExplainer for tree-based model
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)

    # 5. Summary plot
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_summary.png")
    print("SHAP summary plot saved to reports/shap_summary.png")

if __name__ == "__main__":
    main()