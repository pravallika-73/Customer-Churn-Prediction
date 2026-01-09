import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from category_encoders import WOEEncoder
import joblib

from src.features.build_features import engineer_features, woe_encode

print("🚀 Churn Prediction - Data Prep Starting...")

# 1. Load dataset
df = pd.read_csv('data/raw/telco.csv')
print(f"Dataset loaded: {df.shape}")

# 2–4. Cleaning + feature engineering + WOE
df = engineer_features(df)
df_woe, woe = woe_encode(df, 'Churn')

# 5. Final feature matrix X and target y
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges',
            'tenure_group', 'charges_ratio']
X = pd.concat([df[num_cols], df_woe], axis=1)
y = df['Churn']

# 6. Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Save processed data and objects
joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test), 'models/data.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(woe, 'models/woe.pkl')

print("✅ DATA READY! Saved to models/data.pkl")