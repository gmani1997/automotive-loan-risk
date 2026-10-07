import pandas as pd
from src.database import engine


# ============================================================
# 1. LOAD DATA FROM MYSQL
# ============================================================

# Read the complete training dataset from MySQL.
query = """
SELECT *
FROM loan_data
"""

df = pd.read_sql(query, engine)

print("Original dataset shape:", df.shape)


# ============================================================
# 2. HANDLE MISSING VALUES
# ============================================================

# Employment.Type contains 7,661 missing values.
# We don't want to delete these rows because they contain
# useful information in other columns.
#
# "Unknown" is better than using the mode because we don't
# want to incorrectly assume that missing employment data
# belongs to the most common employment category.

df["Employment.Type"] = df["Employment.Type"].fillna("Unknown")

print(
    "\nMissing Employment.Type after cleaning:",
    df["Employment.Type"].isna().sum()
)


# ============================================================
# 3. CONVERT DATE COLUMNS
# ============================================================

# Date.of.Birth and DisbursalDate are currently stored
# as strings.
#
# Convert them to Pandas datetime objects so that we can
# later extract useful features such as age and month/year.

df["Date.of.Birth"] = pd.to_datetime(
    df["Date.of.Birth"],
    format="%d-%m-%Y",
    errors="coerce"
)

df["DisbursalDate"] = pd.to_datetime(
    df["DisbursalDate"],
    format="%d-%m-%Y",
    errors="coerce"
)


# ============================================================
# 4. CONVERT ACCOUNT AGE TO MONTHS
# ============================================================

# AVERAGE.ACCT.AGE contains values such as:
#
# "0yrs 0mon"
# "1yrs 11mon"
# "2yrs 6mon"
#
# ML models cannot directly use these strings.
# Convert them into total months.

def convert_to_months(value):
    """
    Convert a value such as '1yrs 11mon'
    into total months.

    Example:
    1 year 11 months = 23 months
    """

    if pd.isna(value):
        return None

    value = str(value)

    years = 0
    months = 0

    # Extract the year value.
    if "yrs" in value:
        years = int(value.split("yrs")[0].strip())

    # Extract the month value.
    if "mon" in value:
        month_part = value.split("yrs")[-1]
        month_part = month_part.replace("mon", "").strip()

        if month_part:
            months = int(month_part)

    return years * 12 + months


df["AVERAGE.ACCT.AGE.MONTHS"] = (
    df["AVERAGE.ACCT.AGE"].apply(convert_to_months)
)


df["CREDIT.HISTORY.LENGTH.MONTHS"] = (
    df["CREDIT.HISTORY.LENGTH"].apply(convert_to_months)
)


# ============================================================
# 5. HANDLE SUSPICIOUS NEGATIVE BALANCES
# ============================================================

# Our validation showed negative values in:
#
# PRI.CURRENT.BALANCE
# SEC.CURRENT.BALANCE
#
# Since a negative current balance is suspicious for this
# project, we treat these values as missing rather than
# deleting the entire customer record.

df.loc[
    df["PRI.CURRENT.BALANCE"] < 0,
    "PRI.CURRENT.BALANCE"
] = pd.NA

df.loc[
    df["SEC.CURRENT.BALANCE"] < 0,
    "SEC.CURRENT.BALANCE"
] = pd.NA


# ============================================================
# 6. DROP IDENTIFIER / CONSTANT COLUMNS
# ============================================================

# UniqueID is an identifier and should not be used as a
# predictive feature.
#
# MobileNo_Avl_Flag contains only the value 1 for all records,
# so it provides no predictive information.

df.drop(
    columns=[
        "UniqueID",
        "MobileNo_Avl_Flag"
    ],
    inplace=True
)


# ============================================================
# 7. REMOVE ORIGINAL STRING-BASED TIME COLUMNS
# ============================================================

# We created numerical versions of these columns in months.
# The original string columns are no longer required by ML.

df.drop(
    columns=[
        "AVERAGE.ACCT.AGE",
        "CREDIT.HISTORY.LENGTH"
    ],
    inplace=True
)


# ============================================================
# 8. DISPLAY CLEANED DATA INFORMATION
# ============================================================

print("\nCleaned dataset shape:", df.shape)

print("\nRemaining missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nCleaned data types:")
print(df.dtypes)


# ============================================================
# 9. SAVE CLEANED DATA
# ============================================================

# Save the cleaned dataset locally.
# We will use this file for EDA and ML later.

output_path = "data/processed/cleaned_loan_data.csv"

df.to_csv(
    output_path,
    index=False
)

print(f"\nCleaned dataset saved to: {output_path}")