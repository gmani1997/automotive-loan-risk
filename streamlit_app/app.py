import streamlit as st
import requests


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Automotive Loan Risk Prediction",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("🚗 Automotive Loan Credit Risk Prediction")

st.write(
    "Enter the applicant and loan details below to estimate "
    "the probability of loan default."
)

st.divider()


# ============================================================
# FASTAPI CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"


# ============================================================
# LOAN INFORMATION
# ============================================================

st.subheader("Loan Information")

col1, col2, col3 = st.columns(3)

with col1:

    disbursed_amount = st.number_input(
        "Disbursed Amount",
        min_value=0.0,
        value=50578.0
    )

with col2:

    asset_cost = st.number_input(
        "Asset Cost",
        min_value=0.0,
        value=58400.0
    )

with col3:

    ltv = st.number_input(
        "Loan-to-Value (LTV)",
        min_value=0.0,
        value=89.55
    )


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    date_of_birth = st.date_input(
        "Date of Birth"
    )

with col2:

    employment_type = st.selectbox(
        "Employment Type",
        [
            "Salaried",
            "Self employed",
            "Unknown"
        ]
    )

with col3:

    disbursal_date = st.date_input(
        "Disbursal Date"
    )


# ============================================================
# CREDIT BUREAU INFORMATION
# ============================================================

st.subheader("Credit Bureau Information")

col1, col2 = st.columns(2)

with col1:

    cns_score = st.number_input(
        "Credit Bureau Score",
        min_value=0.0,
        value=0.0
    )

with col2:

    cns_description = st.text_input(
        "Credit Score Description",
        value="No Bureau History Available"
    )


# ============================================================
# PRIMARY CREDIT ACCOUNT INFORMATION
# ============================================================

st.subheader("Primary Credit Accounts")

col1, col2, col3 = st.columns(3)

with col1:

    pri_no_of_accts = st.number_input(
        "Primary Accounts",
        min_value=0.0,
        value=0.0
    )

with col2:

    pri_active_accts = st.number_input(
        "Primary Active Accounts",
        min_value=0.0,
        value=0.0
    )

with col3:

    pri_overdue_accts = st.number_input(
        "Primary Overdue Accounts",
        min_value=0.0,
        value=0.0
    )

col1, col2, col3 = st.columns(3)

with col1:

    pri_current_balance = st.number_input(
        "Primary Current Balance",
        min_value=0.0,
        value=0.0
    )

with col2:

    pri_sanctioned_amount = st.number_input(
        "Primary Sanctioned Amount",
        min_value=0.0,
        value=0.0
    )

with col3:

    pri_disbursed_amount = st.number_input(
        "Primary Disbursed Amount",
        min_value=0.0,
        value=0.0
    )


# ============================================================
# SECONDARY CREDIT ACCOUNT INFORMATION
# ============================================================

st.subheader("Secondary Credit Accounts")

col1, col2, col3 = st.columns(3)

with col1:

    sec_no_of_accts = st.number_input(
        "Secondary Accounts",
        min_value=0.0,
        value=0.0
    )

with col2:

    sec_active_accts = st.number_input(
        "Secondary Active Accounts",
        min_value=0.0,
        value=0.0
    )

with col3:

    sec_overdue_accts = st.number_input(
        "Secondary Overdue Accounts",
        min_value=0.0,
        value=0.0
    )

col1, col2, col3 = st.columns(3)

with col1:

    sec_current_balance = st.number_input(
        "Secondary Current Balance",
        min_value=0.0,
        value=0.0
    )

with col2:

    sec_sanctioned_amount = st.number_input(
        "Secondary Sanctioned Amount",
        min_value=0.0,
        value=0.0
    )

with col3:

    sec_disbursed_amount = st.number_input(
        "Secondary Disbursed Amount",
        min_value=0.0,
        value=0.0
    )


# ============================================================
# INSTALLMENT INFORMATION
# ============================================================

st.subheader("Installment Information")

col1, col2 = st.columns(2)

with col1:

    primary_instal_amt = st.number_input(
        "Primary Installment Amount",
        min_value=0.0,
        value=0.0
    )

with col2:

    sec_instal_amt = st.number_input(
        "Secondary Installment Amount",
        min_value=0.0,
        value=0.0
    )


# ============================================================
# CREDIT HISTORY
# ============================================================

st.subheader("Credit History")

col1, col2, col3 = st.columns(3)

with col1:

    new_accts = st.number_input(
        "New Accounts in Last 6 Months",
        min_value=0.0,
        value=0.0
    )

with col2:

    delinquent_accts = st.number_input(
        "Delinquent Accounts in Last 6 Months",
        min_value=0.0,
        value=0.0
    )

with col3:

    no_of_inquiries = st.number_input(
        "Number of Credit Inquiries",
        min_value=0.0,
        value=0.0
    )

col1, col2 = st.columns(2)

with col1:

    average_acct_age = st.number_input(
        "Average Account Age (Months)",
        min_value=0.0,
        value=0.0
    )

with col2:

    credit_history_length = st.number_input(
        "Credit History Length (Months)",
        min_value=0.0,
        value=0.0
    )


# ============================================================
# DOCUMENT VERIFICATION FLAGS
# ============================================================

st.subheader("Document Verification")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    aadhar_flag = st.selectbox(
        "Aadhar",
        [0, 1]
    )

with col2:

    pan_flag = st.selectbox(
        "PAN",
        [0, 1]
    )

with col3:

    voter_flag = st.selectbox(
        "Voter ID",
        [0, 1]
    )

with col4:

    driving_flag = st.selectbox(
        "Driving License",
        [0, 1]
    )

with col5:

    passport_flag = st.selectbox(
        "Passport",
        [0, 1]
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

if st.button(
    "Predict Loan Risk",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # Prepare API request
    # --------------------------------------------------------

    payload = {

        "disbursed_amount": disbursed_amount,

        "asset_cost": asset_cost,

        "ltv": ltv,

        "Date_of_Birth": date_of_birth.strftime(
            "%Y-%m-%d"
        ),

        "Employment_Type": employment_type,

        "DisbursalDate": disbursal_date.strftime(
            "%Y-%m-%d"
        ),

        "PERFORM_CNS_SCORE": cns_score,

        "PERFORM_CNS_SCORE_DESCRIPTION":
            cns_description,

        "PRI_NO_OF_ACCTS": pri_no_of_accts,

        "PRI_ACTIVE_ACCTS": pri_active_accts,

        "PRI_OVERDUE_ACCTS": pri_overdue_accts,

        "PRI_CURRENT_BALANCE":
            pri_current_balance,

        "PRI_SANCTIONED_AMOUNT":
            pri_sanctioned_amount,

        "PRI_DISBURSED_AMOUNT":
            pri_disbursed_amount,

        "SEC_NO_OF_ACCTS":
            sec_no_of_accts,

        "SEC_ACTIVE_ACCTS":
            sec_active_accts,

        "SEC_OVERDUE_ACCTS":
            sec_overdue_accts,

        "SEC_CURRENT_BALANCE":
            sec_current_balance,

        "SEC_SANCTIONED_AMOUNT":
            sec_sanctioned_amount,

        "SEC_DISBURSED_AMOUNT":
            sec_disbursed_amount,

        "PRIMARY_INSTAL_AMT":
            primary_instal_amt,

        "SEC_INSTAL_AMT":
            sec_instal_amt,

        "NEW_ACCTS_IN_LAST_SIX_MONTHS":
            new_accts,

        "DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS":
            delinquent_accts,

        "AVERAGE_ACCT_AGE_MONTHS":
            average_acct_age,

        "CREDIT_HISTORY_LENGTH_MONTHS":
            credit_history_length,

        "NO_OF_INQUIRIES":
            no_of_inquiries,

        "Aadhar_flag":
            aadhar_flag,

        "PAN_flag":
            pan_flag,

        "VoterID_flag":
            voter_flag,

        "Driving_flag":
            driving_flag,

        "Passport_flag":
            passport_flag
    }


    # --------------------------------------------------------
    # Call FastAPI
    # --------------------------------------------------------

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        result = response.json()


        # ----------------------------------------------------
        # Display prediction
        # ----------------------------------------------------

        st.subheader("Prediction Result")

        probability = (
            result["default_probability"] * 100
        )

        prediction = result["prediction"]

        risk_category = result["risk_category"]

        decision = result["decision"]


        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Default Probability",
                f"{probability:.2f}%"
            )

        with col2:

            st.metric(
                "Risk Category",
                risk_category
            )

        with col3:

            st.metric(
                "Decision",
                decision
            )


        # ----------------------------------------------------
        # Business interpretation
        # ----------------------------------------------------

        if prediction == 1:

            st.error(
                "⚠️ Potential Default: "
                "The applicant is classified as High Risk."
            )

        else:

            st.success(
                "✅ Potential Non-Default: "
                "The applicant is classified as Lower Risk."
            )


        st.caption(
            f"Classification threshold: "
            f"{result['threshold']}"
        )


    except requests.exceptions.ConnectionError:

        st.error(
            "Unable to connect to FastAPI. "
            "Make sure the FastAPI server is running."
        )

    except requests.exceptions.Timeout:

        st.error(
            "The prediction API request timed out."
        )

    except requests.exceptions.HTTPError as e:

        st.error(
            f"FastAPI returned an error: {e}"
        )

    except Exception as e:

        st.error(
            f"Unexpected error: {e}"
        )

        