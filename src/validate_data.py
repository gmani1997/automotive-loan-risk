import pandas as pd

# Import the SQLAlchemy engine from our database connection file.
from src.database import engine


# ============================================================
# 1. LOAD DATA FROM MYSQL
# ============================================================

# SQL query to retrieve the training data from MySQL.
query = """
SELECT *
FROM loan_data
"""

# Execute the query and load the result into a Pandas DataFrame.
df = pd.read_sql(query, engine)

print("=" * 70)
print("DATA VALIDATION REPORT")
print("=" * 70)


# ============================================================
# 2. BASIC DATASET INFORMATION
# ============================================================

print("\n1. DATASET SHAPE")
print("-" * 70)

# Shape returns:
# (number_of_rows, number_of_columns)
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\n2. MISSING VALUES")
print("-" * 70)

# isnull().sum() counts missing values in every column.
missing_values = df.isnull().sum()

# Display only columns that actually contain missing values.
missing_values = missing_values[missing_values > 0]

if missing_values.empty:
    print("No missing values found.")
else:
    print(missing_values)


# ============================================================
# 4. CHECK DUPLICATE ROWS
# ============================================================

print("\n3. DUPLICATE ROWS")
print("-" * 70)

# duplicated().sum() counts completely duplicated records.
duplicate_rows = df.duplicated().sum()

print(f"Duplicate rows: {duplicate_rows}")


# ============================================================
# 5. CHECK DUPLICATE UNIQUE IDs
# ============================================================

print("\n4. DUPLICATE UniqueID")
print("-" * 70)

# UniqueID should identify each loan/customer record.
duplicate_ids = df["UniqueID"].duplicated().sum()

print(f"Duplicate UniqueID values: {duplicate_ids}")


# ============================================================
# 6. CHECK TARGET VARIABLE
# ============================================================

print("\n5. TARGET VARIABLE - loan_default")
print("-" * 70)

# Display the number of records for each target class.
target_counts = df["loan_default"].value_counts().sort_index()

print(target_counts)


# Calculate the percentage distribution.
target_percentage = (
    df["loan_default"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
)

print("\nTarget percentage:")
print(target_percentage)


# ============================================================
# 7. CHECK TARGET VALUES
# ============================================================

print("\n6. TARGET VALUE VALIDATION")
print("-" * 70)

# Our target should contain only 0 and 1.
valid_target_values = {0, 1}

actual_target_values = set(df["loan_default"].dropna().unique())

invalid_target_values = actual_target_values - valid_target_values

if invalid_target_values:
    print("Invalid target values found:")
    print(invalid_target_values)
else:
    print("Target contains only valid values: 0 and 1.")


# ============================================================
# 8. CHECK NUMERICAL COLUMNS
# ============================================================

print("\n7. NUMERICAL COLUMNS")
print("-" * 70)

# Select all numerical columns.
numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

print(f"Number of numerical columns: {len(numerical_columns)}")
print("\nNumerical columns:")
print(list(numerical_columns))


# ============================================================
# 9. CHECK NEGATIVE VALUES
# ============================================================

print("\n8. NEGATIVE VALUES")
print("-" * 70)

# Negative values can be suspicious for financial quantities.
# We check numerical columns and report how many negative
# values each column contains.
negative_values = {}

for column in numerical_columns:
    count = (df[column] < 0).sum()

    if count > 0:
        negative_values[column] = count

if negative_values:
    print("Negative values found:")
    for column, count in negative_values.items():
        print(f"{column}: {count}")
else:
    print("No negative values found in numerical columns.")


# ============================================================
# 10. CHECK CATEGORICAL COLUMNS
# ============================================================

print("\n9. CATEGORICAL / TEXT COLUMNS")
print("-" * 70)

# Select columns stored as strings.
categorical_columns = df.select_dtypes(
    include=["object", "string"]
).columns

print(f"Number of categorical/text columns: {len(categorical_columns)}")

for column in categorical_columns:
    print(f"\n{column}")
    print(f"Unique values: {df[column].nunique(dropna=False)}")

    # Show the first few most frequent values.
    print(df[column].value_counts(dropna=False).head(5))


# ============================================================
# 11. CHECK NUMERICAL SUMMARY
# ============================================================

print("\n10. NUMERICAL SUMMARY")
print("-" * 70)

# describe() gives statistical information such as:
# count, mean, standard deviation, minimum, maximum, etc.
print(df[numerical_columns].describe().T)


# ============================================================
# 12. FINAL VALIDATION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION COMPLETED")
print("=" * 70)