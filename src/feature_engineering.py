import numpy as np

def create_risk_features(df):
    if (
        'Loan_Amount' in df.columns and
        'Annual_Household_Income' in df.columns
    ):
        df['Loan_to_Annual_Income'] = (
            df['Loan_Amount']
            /
            df['Annual_Household_Income']
        )
    return df

def create_high_risk_target(df):
    df['High_Risk'] = np.where(
        df['Default_History_Count'] > 0,
        1,
        0
    )
    return df