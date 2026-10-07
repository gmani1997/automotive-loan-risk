import pandas as pd
import requests

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"

FEATURE_DATA_PATH = (
    "data/processed/feature_engineered_loan_data.csv"
)

CLEAN_DATA_PATH = (
    "data/processed/cleaned_loan_data.csv"
)

# Number of real test records to send through the API.
SAMPLE_SIZE = 20

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

feature_df = pd.read_csv(FEATURE_DATA_PATH)

clean_df = pd.read_csv(CLEAN_DATA_PATH)

print("Feature-engineered shape:", feature_df.shape)
print("Cleaned data shape:", clean_df.shape)


# ============================================================
# RECREATE THE SAME TRAIN / TEST SPLIT
# ============================================================

# We reproduce the exact split used during model development.
# This allows us to identify records belonging to the held-out
# test set while still using the original raw columns for API input.

X = feature_df.drop(
    columns=["loan_default"]
)

y = feature_df["loan_default"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


print("\nTest dataset size:", len(X_test))


# ============================================================
# GET ORIGINAL RAW RECORDS FOR TEST SET
# ============================================================

# X_test retains the original row indexes from the
# feature-engineered dataset.
#
# The cleaned dataset has the same row order because the
# feature-engineering process did not reorder records.
test_indices = X_test.index

raw_test_df = clean_df.iloc[test_indices].copy()

raw_test_df["loan_default"] = y_test.values


# ============================================================
# SELECT TEST RECORDS WITHOUT MISSING API VALUES
# ============================================================

required_columns = [
    "disbursed_amount",
    "asset_cost",
    "ltv",
    "Date.of.Birth",
    "Employment.Type",
    "DisbursalDate",
    "Aadhar_flag",
    "PAN_flag",
    "VoterID_flag",
    "Driving_flag",
    "Passport_flag",
    "PERFORM_CNS.SCORE",
    "PERFORM_CNS.SCORE.DESCRIPTION",
    "PRI.NO.OF.ACCTS",
    "PRI.ACTIVE.ACCTS",
    "PRI.OVERDUE.ACCTS",
    "PRI.CURRENT.BALANCE",
    "PRI.SANCTIONED.AMOUNT",
    "PRI.DISBURSED.AMOUNT",
    "SEC.NO.OF.ACCTS",
    "SEC.ACTIVE.ACCTS",
    "SEC.OVERDUE.ACCTS",
    "SEC.CURRENT.BALANCE",
    "SEC.SANCTIONED.AMOUNT",
    "SEC.DISBURSED.AMOUNT",
    "PRIMARY.INSTAL.AMT",
    "SEC.INSTAL.AMT",
    "NEW.ACCTS.IN.LAST.SIX.MONTHS",
    "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS",
    "AVERAGE.ACCT.AGE.MONTHS",
    "CREDIT.HISTORY.LENGTH.MONTHS",
    "NO.OF_INQUIRIES"
]

# For this first API validation, use complete records.
# This avoids sending null values to the Pydantic API schema.
complete_test_df = raw_test_df.dropna(
    subset=required_columns
)

sample_df = complete_test_df.head(
    SAMPLE_SIZE
).copy()


print(
    "Records selected for API validation:",
    len(sample_df)
)


# ============================================================
# CONVERT RAW DATA TO API PAYLOAD
# ============================================================

def create_api_payload(row):
    """
    Convert one raw dataset row into the JSON structure
    expected by the FastAPI /predict endpoint.
    """

    return {
        "disbursed_amount": float(
            row["disbursed_amount"]
        ),

        "asset_cost": float(
            row["asset_cost"]
        ),

        "ltv": float(
            row["ltv"]
        ),

        "Date_of_Birth": str(
            row["Date.of.Birth"]
        ),

        "Employment_Type": str(
            row["Employment.Type"]
        ),

        "DisbursalDate": str(
            row["DisbursalDate"]
        ),

        "PERFORM_CNS_SCORE": float(
            row["PERFORM_CNS.SCORE"]
        ),

        "PERFORM_CNS_SCORE_DESCRIPTION": str(
            row["PERFORM_CNS.SCORE.DESCRIPTION"]
        ),

        "PRI_NO_OF_ACCTS": float(
            row["PRI.NO.OF.ACCTS"]
        ),

        "PRI_ACTIVE_ACCTS": float(
            row["PRI.ACTIVE.ACCTS"]
        ),

        "PRI_OVERDUE_ACCTS": float(
            row["PRI.OVERDUE.ACCTS"]
        ),

        "PRI_CURRENT_BALANCE": float(
            row["PRI.CURRENT.BALANCE"]
        ),

        "PRI_SANCTIONED_AMOUNT": float(
            row["PRI.SANCTIONED.AMOUNT"]
        ),

        "PRI_DISBURSED_AMOUNT": float(
            row["PRI.DISBURSED.AMOUNT"]
        ),

        "SEC_NO_OF_ACCTS": float(
            row["SEC.NO.OF.ACCTS"]
        ),

        "SEC_ACTIVE_ACCTS": float(
            row["SEC.ACTIVE.ACCTS"]
        ),

        "SEC_OVERDUE_ACCTS": float(
            row["SEC.OVERDUE.ACCTS"]
        ),

        "SEC_CURRENT_BALANCE": float(
            row["SEC.CURRENT.BALANCE"]
        ),

        "SEC_SANCTIONED_AMOUNT": float(
            row["SEC.SANCTIONED.AMOUNT"]
        ),

        "SEC_DISBURSED_AMOUNT": float(
            row["SEC.DISBURSED.AMOUNT"]
        ),

        "PRIMARY_INSTAL_AMT": float(
            row["PRIMARY.INSTAL.AMT"]
        ),

        "SEC_INSTAL_AMT": float(
            row["SEC.INSTAL.AMT"]
        ),

        "NEW_ACCTS_IN_LAST_SIX_MONTHS": float(
            row["NEW.ACCTS.IN.LAST.SIX.MONTHS"]
        ),

        "DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS": float(
            row["DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS"]
        ),

        "AVERAGE_ACCT_AGE_MONTHS": float(
            row["AVERAGE.ACCT.AGE.MONTHS"]
        ),

        "CREDIT_HISTORY_LENGTH_MONTHS": float(
            row["CREDIT.HISTORY.LENGTH.MONTHS"]
        ),

        "NO_OF_INQUIRIES": float(
            row["NO.OF_INQUIRIES"]
        ),

        "Aadhar_flag": float(
            row["Aadhar_flag"]
        ),

        "PAN_flag": float(
            row["PAN_flag"]
        ),

        "VoterID_flag": float(
            row["VoterID_flag"]
        ),

        "Driving_flag": float(
            row["Driving_flag"]
        ),

        "Passport_flag": float(
            row["Passport_flag"]
        )
    }


# ============================================================
# SEND RECORDS TO FASTAPI
# ============================================================

results = []

print("\nSending records to FastAPI...")
print("-" * 70)


for index, row in sample_df.iterrows():

    actual = int(
        row["loan_default"]
    )

    payload = create_api_payload(row)

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        prediction_result = response.json()

        prediction = int(
            prediction_result["prediction"]
        )

        probability = float(
            prediction_result["default_probability"]
        )

        results.append({
            "row_index": index,
            "actual": actual,
            "prediction": prediction,
            "probability": probability,
            "risk_category": prediction_result[
                "risk_category"
            ]
        })

        print(
            f"Row {index}: "
            f"Actual={actual}, "
            f"Prediction={prediction}, "
            f"Probability={probability:.4f}"
        )

    except Exception as e:

        print(
            f"Row {index}: API request failed"
        )

        print("Error:", e)


# ============================================================
# CREATE VALIDATION RESULTS
# ============================================================

results_df = pd.DataFrame(results)


if results_df.empty:

    print("\nNo successful API predictions were returned.")
    print("Check whether FastAPI is running.")

    raise SystemExit


# ============================================================
# CALCULATE VALIDATION METRICS
# ============================================================

y_actual = results_df["actual"]

y_pred = results_df["prediction"]


accuracy = accuracy_score(
    y_actual,
    y_pred
)

precision = precision_score(
    y_actual,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_actual,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_actual,
    y_pred,
    zero_division=0
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FASTAPI MODEL VALIDATION RESULTS")
print("=" * 70)

print(
    f"Records tested : {len(results_df)}"
)

print(
    f"Accuracy       : {accuracy:.4f}"
)

print(
    f"Precision      : {precision:.4f}"
)

print(
    f"Recall         : {recall:.4f}"
)

print(
    f"F1 Score       : {f1:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_actual,
        y_pred
    )
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_actual,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# SAVE VALIDATION RESULTS
# ============================================================

output_path = (
    "data/processed/api_validation_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print(
    f"\nValidation results saved to: {output_path}"
)