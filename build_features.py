import pandas as pd
from category_encoders import WOEEncoder

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    df['tenure_group'] = pd.cut(df['tenure'], bins=5, labels=False)
    df['charges_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    return df

def woe_encode(df: pd.DataFrame, target_col: str):
    cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                'InternetService', 'Contract']
    encoder = WOEEncoder(cols=cat_cols).fit(df[cat_cols], df[target_col])
    df_woe = encoder.transform(df[cat_cols])
    return df_woe, encoder