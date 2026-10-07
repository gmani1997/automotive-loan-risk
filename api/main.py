from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_loan_default


# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Automotive Loan Credit Risk API",
    description="API for predicting automotive loan default risk.",
    version="1.0.0"
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class LoanApplication(BaseModel):

    # -------------------------
    # Loan / asset information
    # -------------------------

    disbursed_amount: float
    asset_cost: float
    ltv: float

    # -------------------------
    # Customer information
    # -------------------------

    Date_of_Birth: str
    Employment_Type: str

    # -------------------------
    # Loan disbursal
    # -------------------------

    DisbursalDate: str

    # -------------------------
    # Credit score
    # -------------------------

    PERFORM_CNS_SCORE: float
    PERFORM_CNS_SCORE_DESCRIPTION: str

    # -------------------------
    # Primary accounts
    # -------------------------

    PRI_NO_OF_ACCTS: float
    PRI_ACTIVE_ACCTS: float
    PRI_OVERDUE_ACCTS: float

    PRI_CURRENT_BALANCE: float
    PRI_SANCTIONED_AMOUNT: float
    PRI_DISBURSED_AMOUNT: float

    # -------------------------
    # Secondary accounts
    # -------------------------

    SEC_NO_OF_ACCTS: float
    SEC_ACTIVE_ACCTS: float
    SEC_OVERDUE_ACCTS: float

    SEC_CURRENT_BALANCE: float
    SEC_SANCTIONED_AMOUNT: float
    SEC_DISBURSED_AMOUNT: float

    # -------------------------
    # Installment information
    # -------------------------

    PRIMARY_INSTAL_AMT: float
    SEC_INSTAL_AMT: float

    # -------------------------
    # Recent credit activity
    # -------------------------

    NEW_ACCTS_IN_LAST_SIX_MONTHS: float
    DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS: float

    # -------------------------
    # Credit history
    # -------------------------

    AVERAGE_ACCT_AGE_MONTHS: float
    CREDIT_HISTORY_LENGTH_MONTHS: float

    NO_OF_INQUIRIES: float

    # -------------------------
    # Document verification flags
    # -------------------------

    Aadhar_flag: float
    PAN_flag: float
    VoterID_flag: float
    Driving_flag: float
    Passport_flag: float


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "automotive-loan-risk-api"
    }


# ============================================================
# LOAN DEFAULT PREDICTION
# ============================================================

@app.post("/predict")
def predict(application: LoanApplication):

    # Convert the validated Pydantic object into a dictionary.
    input_data = application.model_dump()

    # Convert API field names into the exact column names
    # expected by our ML feature-engineering function.

    input_data = {
        "disbursed_amount": input_data["disbursed_amount"],
        "asset_cost": input_data["asset_cost"],
        "ltv": input_data["ltv"],

        "Date.of.Birth": input_data["Date_of_Birth"],
        "Employment.Type": input_data["Employment_Type"],
        "DisbursalDate": input_data["DisbursalDate"],

        "PERFORM_CNS.SCORE": input_data[
            "PERFORM_CNS_SCORE"
        ],

        "PERFORM_CNS.SCORE.DESCRIPTION": input_data[
            "PERFORM_CNS_SCORE_DESCRIPTION"
        ],

        "PRI.NO.OF.ACCTS": input_data[
            "PRI_NO_OF_ACCTS"
        ],

        "PRI.ACTIVE.ACCTS": input_data[
            "PRI_ACTIVE_ACCTS"
        ],

        "PRI.OVERDUE.ACCTS": input_data[
            "PRI_OVERDUE_ACCTS"
        ],

        "PRI.CURRENT.BALANCE": input_data[
            "PRI_CURRENT_BALANCE"
        ],

        "PRI.SANCTIONED.AMOUNT": input_data[
            "PRI_SANCTIONED_AMOUNT"
        ],

        "PRI.DISBURSED.AMOUNT": input_data[
            "PRI_DISBURSED_AMOUNT"
        ],

        "SEC.NO.OF.ACCTS": input_data[
            "SEC_NO_OF_ACCTS"
        ],

        "SEC.ACTIVE.ACCTS": input_data[
            "SEC_ACTIVE_ACCTS"
        ],

        "SEC.OVERDUE.ACCTS": input_data[
            "SEC_OVERDUE_ACCTS"
        ],

        "SEC.CURRENT.BALANCE": input_data[
            "SEC_CURRENT_BALANCE"
        ],

        "SEC.SANCTIONED.AMOUNT": input_data[
            "SEC_SANCTIONED_AMOUNT"
        ],

        "SEC.DISBURSED.AMOUNT": input_data[
            "SEC_DISBURSED_AMOUNT"
        ],

        "PRIMARY.INSTAL.AMT": input_data[
            "PRIMARY_INSTAL_AMT"
        ],

        "SEC.INSTAL.AMT": input_data[
            "SEC_INSTAL_AMT"
        ],

        "NEW.ACCTS.IN.LAST.SIX.MONTHS": input_data[
            "NEW_ACCTS_IN_LAST_SIX_MONTHS"
        ],

        "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS": input_data[
            "DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS"
        ],

        "AVERAGE.ACCT.AGE.MONTHS": input_data[
            "AVERAGE_ACCT_AGE_MONTHS"
        ],

        "CREDIT.HISTORY.LENGTH.MONTHS": input_data[
            "CREDIT_HISTORY_LENGTH_MONTHS"
        ],

        "NO.OF_INQUIRIES": input_data[
            "NO_OF_INQUIRIES"
        ],

        "Aadhar_flag": input_data[
            "Aadhar_flag"
        ],

        "PAN_flag": input_data[
            "PAN_flag"
        ],

        "VoterID_flag": input_data[
            "VoterID_flag"
        ],

        "Driving_flag": input_data[
            "Driving_flag"
        ],

        "Passport_flag": input_data[
            "Passport_flag"
        ],
    }

    # Send the raw application to the ML prediction module.
    result = predict_loan_default(input_data)

    return result













# The below code is used to build the ML model only


# from fastapi import FastAPI
# from pydantic import BaseModel


# # ============================================================
# # CREATE FASTAPI APPLICATION
# # ============================================================

# app = FastAPI(
#     title = "Automotive Loan Credit Risk API",
#     description = "API for predicting automotive loan default risk.",
#     version = "1.0.0"
# )

# # ============================================================
# # REQUEST DATA MODEL
# # ============================================================

# class LoanApplication(BaseModel):
#     """
#     Defines the input structure expected from a new automotive loan application.

#     Pydantic automatically validates the incoming JSON.
#     """

#     disbursed_amount: float
#     asset_cost: float
#     ltv: float

#     Employment_Type: str

#     PERFORM_CNS_SCORE: float
#     PERFORM_CNS_SCORE_DESCRIPTION: str

#     PRI_NO_OF_ACCTS: float
#     PRI_ACTIVE_ACCTS: float
#     PRI_OVERDUE_ACCTS: float

#     PRI_CURRENT_BALANCE: float
#     PRI_SANCTIONED_AMOUNT: float
#     PRI_DISBURSED_AMOUNT: float

#     SEC_NO_OF_ACCTS: float
#     SEC_ACTIVE_ACCTS: float
#     SEC_OVERDUE_ACCTS: float

#     SEC_CURRENT_BALANCE: float
#     SEC_SANCTIONED_AMOUNT: float
#     SEC_DISBURSED_AMOUNT: float

#     PRIMARY_INSTAL_AMT: float
#     SEC_INSTAL_AMT: float

#     NEW_ACCTS_IN_LAST_SIX_MONTHS: float
#     DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS: float

#     AVERAGE_ACCT_AGE_MONTHS: float
#     CREDIT_HISTORY_LENGTH_MONTHS: float

#     NO_OF_INQUIRIES: float

#     Age: float
#     Disbursal_Year: float
#     Disbursal_Month: float

#     Loan_Asset_Ratio: float

#     Primary_Active_Ratio: float
#     Primary_Overdue_Ratio: float

#     Secondary_Active_Ratio: float
#     Secondary_Overdue_Ratio: float

#     Total_Accounts: float
#     Total_Active_Accounts: float
#     Total_Overdue_Accounts: float

#     Total_Current_Balance: float
#     Total_Sanctioned_Amount: float
#     Total_Previous_Disbursed_Amount: float

#     Total_Credit_History_Months: float
#     Average_Account_Age_Months: float

#     Recent_Credit_Activity: float


# # ============================================================
# # HEALTH CHECK ENDPOINT
# # ============================================================

# @app.get("/health")
# def health_check():
#     """
#     Simple endpoint used to verify that the API is running
#     """

#     return {
#         "status": "healthy",
#         "service": "automotive-loan-risk-api",
#     }


