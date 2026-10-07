import pandas as pd
import numpy as np
import joblib


# ============================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================

PREPROCESSOR_PATH = "models/preprocessor.joblib"
MODEL_PATH = "models/xgboost_tuned.joblib"

# Load the preprocessing pipeline that was fitted during training.
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Load the tuned XGBoost model.
model = joblib.load(MODEL_PATH)


# ============================================================
# PREDICTION CONFIGURATION
# ============================================================

# Classification threshold selected from threshold analysis.
# We selected 0.45 because it gave the best F1 among the
# tested thresholds while maintaining stronger recall.
THRESHOLD = 0.45


# ============================================================
# FEATURE ENGINEERING FOR NEW APPLICATIONS
# ============================================================

def create_features(input_data: dict) -> pd.DataFrame:
    """
    Convert raw loan application data into the same feature
    structure used during model training.

    The API receives raw customer/loan information.
    This function creates the engineered features required
    by the trained model.
    """

    # Convert the input dictionary into a one-row DataFrame.
    df = pd.DataFrame([input_data])

    # --------------------------------------------------------
    # Convert dates
    # --------------------------------------------------------

    df["Date.of.Birth"] = pd.to_datetime(
        df["Date.of.Birth"],
        errors="coerce"
    )

    df["DisbursalDate"] = pd.to_datetime(
        df["DisbursalDate"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Age
    # --------------------------------------------------------

    # Calculate customer's age at the time of loan disbursal.
    df["Age"] = (
        (
            df["DisbursalDate"]
            - df["Date.of.Birth"]
        ).dt.days / 365.25
    ).round()

    # --------------------------------------------------------
    # Disbursal date features
    # --------------------------------------------------------

    df["Disbursal_Year"] = df["DisbursalDate"].dt.year

    df["Disbursal_Month"] = df["DisbursalDate"].dt.month

    # --------------------------------------------------------
    # Loan / asset ratio
    # --------------------------------------------------------

    # Avoid division by zero by replacing zero asset cost
    # with NumPy NaN.
    df["Loan_Asset_Ratio"] = (
        df["disbursed_amount"]
        / df["asset_cost"].replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Primary account ratios
    # --------------------------------------------------------

    # Avoid division by zero when the customer has no
    # primary credit accounts.
    primary_accounts = df["PRI.NO.OF.ACCTS"].replace(0, np.nan)

    df["Primary_Active_Ratio"] = (
        df["PRI.ACTIVE.ACCTS"]
        / primary_accounts
    )

    df["Primary_Overdue_Ratio"] = (
        df["PRI.OVERDUE.ACCTS"]
        / primary_accounts
    )

    # --------------------------------------------------------
    # Secondary account ratios
    # --------------------------------------------------------

    # Avoid division by zero when there are no secondary
    # credit accounts.
    secondary_accounts = df["SEC.NO.OF.ACCTS"].replace(0, np.nan)

    df["Secondary_Active_Ratio"] = (
        df["SEC.ACTIVE.ACCTS"]
        / secondary_accounts
    )

    df["Secondary_Overdue_Ratio"] = (
        df["SEC.OVERDUE.ACCTS"]
        / secondary_accounts
    )

    # --------------------------------------------------------
    # Total account features
    # --------------------------------------------------------

    df["Total_Accounts"] = (
        df["PRI.NO.OF.ACCTS"]
        + df["SEC.NO.OF.ACCTS"]
    )

    df["Total_Active_Accounts"] = (
        df["PRI.ACTIVE.ACCTS"]
        + df["SEC.ACTIVE.ACCTS"]
    )

    df["Total_Overdue_Accounts"] = (
        df["PRI.OVERDUE.ACCTS"]
        + df["SEC.OVERDUE.ACCTS"]
    )

    # --------------------------------------------------------
    # Total current balance
    # --------------------------------------------------------

    # Missing current balances are treated as zero for the
    # aggregate feature.
    df["Total_Current_Balance"] = (
        df["PRI.CURRENT.BALANCE"].fillna(0)
        + df["SEC.CURRENT.BALANCE"].fillna(0)
    )

    # --------------------------------------------------------
    # Total sanctioned amount
    # --------------------------------------------------------

    df["Total_Sanctioned_Amount"] = (
        df["PRI.SANCTIONED.AMOUNT"]
        + df["SEC.SANCTIONED.AMOUNT"]
    )

    # --------------------------------------------------------
    # Total previous disbursed amount
    # --------------------------------------------------------

    df["Total_Previous_Disbursed_Amount"] = (
        df["PRI.DISBURSED.AMOUNT"]
        + df["SEC.DISBURSED.AMOUNT"]
    )

    # --------------------------------------------------------
    # Credit history features
    # --------------------------------------------------------

    df["Total_Credit_History_Months"] = (
        df["CREDIT.HISTORY.LENGTH.MONTHS"]
    )

    df["Average_Account_Age_Months"] = (
        df["AVERAGE.ACCT.AGE.MONTHS"]
    )

    # --------------------------------------------------------
    # Recent credit activity
    # --------------------------------------------------------

    df["Recent_Credit_Activity"] = (
        df["NEW.ACCTS.IN.LAST.SIX.MONTHS"]
        + df["DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS"]
    )

    # --------------------------------------------------------
    # Remove columns that were removed during model training
    # --------------------------------------------------------

    df.drop(
        columns=[
            # Raw dates were converted into engineered features.
            "Date.of.Birth",
            "DisbursalDate",

            # Identifiers were removed during feature engineering.
            "UniqueID",
            "MobileNo_Avl_Flag",
            "branch_id",
            "supplier_id",
            "manufacturer_id",
            "Current_pincode_ID",
            "State_ID",
            "Employee_code_ID",

            # Original duration strings were converted into
            # month-based numerical features.
            "AVERAGE.ACCT.AGE",
            "CREDIT.HISTORY.LENGTH",

            # Target variable must never be used for prediction.
            "loan_default"
        ],
        inplace=True,
        errors="ignore"
    )

    # --------------------------------------------------------
    # Clean infinite and nullable values
    # --------------------------------------------------------

    # Ratio calculations can create positive/negative infinity.
    # Convert those values to NumPy NaN.
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Convert Pandas nullable values such as pd.NA into
    # NumPy NaN so that scikit-learn can process them.
    df = df.replace(
        {pd.NA: np.nan}
    )

    # --------------------------------------------------------
    # Convert numerical columns to numeric data types
    # --------------------------------------------------------

    # These two columns are categorical and should remain strings.
    categorical_columns = [
        "Employment.Type",
        "PERFORM_CNS.SCORE.DESCRIPTION"
    ]

    for column in df.columns:

        if column not in categorical_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Align columns with the training data
    # --------------------------------------------------------

    # The saved preprocessor knows exactly which columns were
    # present during training.
    #
    # Reindexing guarantees that:
    # 1. Columns are in the correct order.
    # 2. Unexpected columns are removed.
    # 3. Missing expected columns become NaN and are handled
    #    by the trained SimpleImputer.
    expected_columns = list(
        preprocessor.feature_names_in_
    )

    df = df.reindex(
        columns=expected_columns
    )

    return df


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_loan_default(input_data: dict):
    """
    Predict the probability of loan default for a new
    automotive loan application.
    """

    # --------------------------------------------------------
    # Step 1: Create model features
    # --------------------------------------------------------

    feature_df = create_features(input_data)

    # --------------------------------------------------------
    # Step 2: Apply the fitted preprocessing pipeline
    # --------------------------------------------------------

    # This performs:
    # - Missing value imputation
    # - Standard scaling for numerical features
    # - One-hot encoding for categorical features
    processed_data = preprocessor.transform(
        feature_df
    )

    # --------------------------------------------------------
    # Step 3: Generate default probability
    # --------------------------------------------------------

    default_probability = model.predict_proba(
        processed_data
    )[0, 1]

    # --------------------------------------------------------
    # Step 4: Apply business classification threshold
    # --------------------------------------------------------

    prediction = int(
        default_probability >= THRESHOLD
    )

    # --------------------------------------------------------
    # Step 5: Generate business interpretation
    # --------------------------------------------------------

    if prediction == 1:

        risk_category = "High Risk"
        decision = "Potential Default"

    else:

        risk_category = "Lower Risk"
        decision = "Potential Non-Default"

    # --------------------------------------------------------
    # Step 6: Return API-friendly response
    # --------------------------------------------------------

    return {
        "default_probability": round(
            float(default_probability),
            4
        ),
        "prediction": prediction,
        "risk_category": risk_category,
        "decision": decision,
        "threshold": THRESHOLD
    }








# The below code is used to build the ML model only

# import pandas as pd
# import joblib


# # ============================================================
# # 1. LOAD MODEL AND PREPROCESSOR
# # ============================================================

# # Load these once when the module starts.
# #
# # This is better than loading the files for every prediction
# # request.

# PREPROCESSOR_PATH = "models/preprocessor.joblib"
# MODEL_PATH = "models/xgboost_tuned.joblib"

# preprocessor = joblib.load(
#     PREPROCESSOR_PATH
# )

# model = joblib.load(
#     MODEL_PATH
# )


# # ============================================================
# # 2. FINAL CLASSIFICATION THRESHOLD
# # ============================================================

# # Based on our threshold analysis, 0.45 provides the best
# # F1 score among the thresholds we tested while maintaining
# # relatively strong recall.

# THRESHOLD = 0.45


# # ============================================================
# # 3. PREDICTION FUNCTION
# # ============================================================

# def predict_loan_default(input_data: dict):
#     """
#     Predict whether a loan application is likely to default.

#     Parameters
#     ----------
#     input_data : dict
#         Loan application features.

#     Returns
#     -------
#     dict
#         Default probability, prediction and risk category.
#     """

#     # Convert incoming dictionary into a one-row DataFrame.
#     input_df = pd.DataFrame([input_data])


#     # --------------------------------------------------------
#     # Apply the SAME preprocessing used during model training.
#     # --------------------------------------------------------

#     input_processed = preprocessor.transform(
#         input_df
#     )


#     # --------------------------------------------------------
#     # Get probability of loan default.
#     # --------------------------------------------------------

#     default_probability = model.predict_proba(
#         input_processed
#     )[0, 1]


#     # --------------------------------------------------------
#     # Apply our selected threshold.
#     #
#     # Probability >= 0.45 -> Default Risk
#     # Probability <  0.45 -> Non-Default Risk
#     # --------------------------------------------------------

#     prediction = int(
#         default_probability >= THRESHOLD
#     )


#     # --------------------------------------------------------
#     # Convert prediction into a business-friendly result.
#     # --------------------------------------------------------

#     if prediction == 1:
#         risk_category = "High Risk"
#         decision = "Potential Default"
#     else:
#         risk_category = "Lower Risk"
#         decision = "Potential Non-Default"


#     return {
#         "default_probability": round(
#             float(default_probability),
#             4
#         ),
#         "prediction": prediction,
#         "risk_category": risk_category,
#         "decision": decision,
#         "threshold": THRESHOLD
#     }


# if __name__ == "__main__":

#     # Load one real test record.
#     test_data = pd.read_csv(
#         "data/processed/X_test.csv"
#     )

#     sample = test_data.iloc[0].to_dict()

#     result = predict_loan_default(sample)

#     print("\nPrediction Result:")
#     print(result)