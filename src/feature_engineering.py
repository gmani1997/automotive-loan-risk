import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

df = pd.read_csv(
    "data/processed/cleaned_loan_data.csv"
)

print("Original shape:", df.shape)


# ============================================================
# 2. CONVERT DATE COLUMNS
# ============================================================

# Convert date strings back to datetime because CSV storage
# changes datetime columns into strings.
df["Date.of.Birth"] = pd.to_datetime(
    df["Date.of.Birth"],
    errors="coerce"
)

df["DisbursalDate"] = pd.to_datetime(
    df["DisbursalDate"],
    errors="coerce"
)


# ============================================================
# 3. CREATE AGE FEATURE
# ============================================================

# The dataset contains Date.of.Birth.
# We derive approximate age using the loan disbursal date.
#
# Age is more useful to an ML model than the raw date of birth.

df["Age"] = (
    (df["DisbursalDate"] - df["Date.of.Birth"])
    .dt.days / 365.25
)

# Convert to integer age.
df["Age"] = df["Age"].round().astype("Int64")


# ============================================================
# 4. CREATE DISBURSAL DATE FEATURES
# ============================================================

# Extract useful information from the loan disbursal date.

df["Disbursal_Year"] = df["DisbursalDate"].dt.year

df["Disbursal_Month"] = df["DisbursalDate"].dt.month


# ============================================================
# 5. LOAN-TO-ASSET RATIO
# ============================================================

# This ratio represents the proportion of the asset cost
# covered by the disbursed loan.

df["Loan_Asset_Ratio"] = (
    df["disbursed_amount"] /
    df["asset_cost"].replace(0, np.nan)
)


# ============================================================
# 6. PRIMARY ACCOUNT ACTIVITY RATIO
# ============================================================

# Calculate the proportion of primary accounts that are active.

df["Primary_Active_Ratio"] = (
    df["PRI.ACTIVE.ACCTS"] /
    df["PRI.NO.OF.ACCTS"].replace(0, np.nan)
)


# ============================================================
# 7. PRIMARY OVERDUE RATIO
# ============================================================

# Percentage of primary accounts that are overdue.

df["Primary_Overdue_Ratio"] = (
    df["PRI.OVERDUE.ACCTS"] /
    df["PRI.NO.OF.ACCTS"].replace(0, np.nan)
)


# ============================================================
# 8. SECONDARY ACCOUNT ACTIVITY RATIO
# ============================================================

df["Secondary_Active_Ratio"] = (
    df["SEC.ACTIVE.ACCTS"] /
    df["SEC.NO.OF.ACCTS"].replace(0, np.nan)
)


# ============================================================
# 9. SECONDARY OVERDUE RATIO
# ============================================================

df["Secondary_Overdue_Ratio"] = (
    df["SEC.OVERDUE.ACCTS"] /
    df["SEC.NO.OF.ACCTS"].replace(0, np.nan)
)


# ============================================================
# 10. TOTAL ACCOUNT FEATURES
# ============================================================

# Combine primary and secondary account information.

df["Total_Accounts"] = (
    df["PRI.NO.OF.ACCTS"] +
    df["SEC.NO.OF.ACCTS"]
)

df["Total_Active_Accounts"] = (
    df["PRI.ACTIVE.ACCTS"] +
    df["SEC.ACTIVE.ACCTS"]
)

df["Total_Overdue_Accounts"] = (
    df["PRI.OVERDUE.ACCTS"] +
    df["SEC.OVERDUE.ACCTS"]
)


# ============================================================
# 11. TOTAL CURRENT BALANCE
# ============================================================

# Fill the suspicious negative balances that were converted
# to NaN during preprocessing with zero for this aggregate.
#
# We keep the original columns separately as well.

df["Total_Current_Balance"] = (
    df["PRI.CURRENT.BALANCE"].fillna(0) +
    df["SEC.CURRENT.BALANCE"].fillna(0)
)


# ============================================================
# 12. TOTAL SANCTIONED AMOUNT
# ============================================================

df["Total_Sanctioned_Amount"] = (
    df["PRI.SANCTIONED.AMOUNT"] +
    df["SEC.SANCTIONED.AMOUNT"]
)


# ============================================================
# 13. TOTAL DISBURSED AMOUNT
# ============================================================

df["Total_Previous_Disbursed_Amount"] = (
    df["PRI.DISBURSED.AMOUNT"] +
    df["SEC.DISBURSED.AMOUNT"]
)


# ============================================================
# 14. CREDIT HISTORY FEATURES
# ============================================================

# These columns were already converted to months
# during preprocessing.

df["Total_Credit_History_Months"] = (
    df["CREDIT.HISTORY.LENGTH.MONTHS"]
)

df["Average_Account_Age_Months"] = (
    df["AVERAGE.ACCT.AGE.MONTHS"]
)


# ============================================================
# 15. CREDIT ACTIVITY FEATURES
# ============================================================

# Combine recent account activity and delinquency.

df["Recent_Credit_Activity"] = (
    df["NEW.ACCTS.IN.LAST.SIX.MONTHS"] +
    df["DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS"]
)


# ============================================================
# 16. DROP RAW DATE COLUMNS
# ============================================================

# We have extracted the useful information from these dates.
# The raw dates themselves are not required by the ML model.

df.drop(
    columns=[
        "Date.of.Birth",
        "DisbursalDate"
    ],
    inplace=True
)


# ============================================================
# 17. DROP IDENTIFIER COLUMNS
# ============================================================

# These are identifiers rather than meaningful continuous
# measurements. Treating them as numerical values can cause
# the model to learn meaningless relationships.

identifier_columns = [
    "branch_id",
    "supplier_id",
    "manufacturer_id",
    "Current_pincode_ID",
    "State_ID",
    "Employee_code_ID"
]

df.drop(
    columns=identifier_columns,
    inplace=True,
    errors="ignore"
)


# ============================================================
# 18. HANDLE INFINITE VALUES
# ============================================================

# Ratios can produce infinity when their denominator is zero.
# Convert infinite values to NaN.

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# ============================================================
# 19. DISPLAY FEATURE INFORMATION
# ============================================================

print("\nFeature-engineered shape:", df.shape)

print("\nNew features:")
new_features = [
    "Age",
    "Disbursal_Year",
    "Disbursal_Month",
    "Loan_Asset_Ratio",
    "Primary_Active_Ratio",
    "Primary_Overdue_Ratio",
    "Secondary_Active_Ratio",
    "Secondary_Overdue_Ratio",
    "Total_Accounts",
    "Total_Active_Accounts",
    "Total_Overdue_Accounts",
    "Total_Current_Balance",
    "Total_Sanctioned_Amount",
    "Total_Previous_Disbursed_Amount",
    "Total_Credit_History_Months",
    "Average_Account_Age_Months",
    "Recent_Credit_Activity"
]

print(new_features)


# ============================================================
# 20. SAVE FEATURE-ENGINEERED DATA
# ============================================================

output_path = (
    "data/processed/"
    "feature_engineered_loan_data.csv"
)

df.to_csv(
    output_path,
    index=False
)

print(
    f"\nFeature-engineered dataset saved to: {output_path}"
)