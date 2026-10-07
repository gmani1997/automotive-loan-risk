import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

# Read the cleaned dataset produced by preprocessing.py.
df = pd.read_csv("data/processed/cleaned_loan_data.csv")

print("=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# ============================================================
# 2. BASIC DATASET INFORMATION
# ============================================================

print("\n1. DATASET SHAPE")
print("-" * 70)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


print("\n2. DATA TYPES")
print("-" * 70)
print(df.dtypes)


# ============================================================
# 3. MISSING VALUE ANALYSIS
# ============================================================

print("\n3. MISSING VALUES")
print("-" * 70)

missing = df.isnull().sum()
missing = missing[missing > 0]

if missing.empty:
    print("No missing values found.")
else:
    print(missing)


# ============================================================
# 4. TARGET VARIABLE ANALYSIS
# ============================================================

print("\n4. TARGET VARIABLE ANALYSIS")
print("-" * 70)

target_counts = df["loan_default"].value_counts().sort_index()

print(target_counts)

target_percentage = (
    df["loan_default"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print("\nTarget percentage:")
print(target_percentage)


# Plot target distribution
plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="loan_default"
)

plt.title("Loan Default Distribution")
plt.xlabel("Loan Default")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ============================================================
# 5. NUMERICAL FEATURE SUMMARY
# ============================================================

print("\n5. NUMERICAL FEATURE SUMMARY")
print("-" * 70)

numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

print(df[numerical_columns].describe().T)


# ============================================================
# 6. CATEGORICAL FEATURE ANALYSIS
# ============================================================

print("\n6. CATEGORICAL FEATURES")
print("-" * 70)

categorical_columns = df.select_dtypes(
    include=["object", "string"]
).columns

for column in categorical_columns:

    print(f"\n{column}")
    print("-" * 50)

    print(df[column].value_counts(dropna=False).head(10))


# ============================================================
# 7. DEFAULT RATE BY EMPLOYMENT TYPE
# ============================================================

print("\n7. DEFAULT RATE BY EMPLOYMENT TYPE")
print("-" * 70)

employment_default = (
    df.groupby("Employment.Type")["loan_default"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print(employment_default)


plt.figure(figsize=(8, 5))

sns.barplot(
    x=employment_default.index,
    y=employment_default.values
)

plt.title("Loan Default Rate by Employment Type")
plt.xlabel("Employment Type")
plt.ylabel("Default Rate (%)")

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()


# ============================================================
# 8. DEFAULT RATE BY LTV BUCKET
# ============================================================

# LTV is an important financial/risk variable.
# Create business-friendly LTV groups.

df["LTV_Group"] = pd.cut(
    df["ltv"],
    bins=[0, 60, 70, 80, 90, 100],
    labels=[
        "<=60",
        "60-70",
        "70-80",
        "80-90",
        "90-100"
    ]
)

ltv_default = (
    df.groupby("LTV_Group", observed=True)["loan_default"]
    .mean()
    .mul(100)
)

print("\n8. DEFAULT RATE BY LTV GROUP")
print("-" * 70)
print(ltv_default)


plt.figure(figsize=(8, 5))

sns.barplot(
    x=ltv_default.index,
    y=ltv_default.values
)

plt.title("Loan Default Rate by LTV Group")
plt.xlabel("LTV Group")
plt.ylabel("Default Rate (%)")

plt.tight_layout()
plt.show()


# ============================================================
# 9. DEFAULT RATE BY CREDIT SCORE
# ============================================================

# Create credit-score groups to make the analysis easier
# to interpret from a business perspective.

df["Credit_Score_Group"] = pd.cut(
    df["PERFORM_CNS.SCORE"],
    bins=[-1, 0, 300, 500, 700, 900],
    labels=[
        "No Score",
        "Low",
        "Medium",
        "High",
        "Very High"
    ]
)

credit_default = (
    df.groupby(
        "Credit_Score_Group",
        observed=True
    )["loan_default"]
    .mean()
    .mul(100)
)

print("\n9. DEFAULT RATE BY CREDIT SCORE GROUP")
print("-" * 70)
print(credit_default)


plt.figure(figsize=(8, 5))

sns.barplot(
    x=credit_default.index,
    y=credit_default.values
)

plt.title("Loan Default Rate by Credit Score")
plt.xlabel("Credit Score Group")
plt.ylabel("Default Rate (%)")

plt.tight_layout()
plt.show()


# ============================================================
# 10. CORRELATION ANALYSIS
# ============================================================

print("\n10. CORRELATION ANALYSIS")
print("-" * 70)

# Select numerical columns.
correlation_data = df[numerical_columns]

correlation_matrix = correlation_data.corr()

# Find correlation specifically with the target.
target_correlation = (
    correlation_matrix["loan_default"]
    .sort_values(ascending=False)
)

print("\nCorrelation with loan_default:")
print(target_correlation)


# Heatmap
plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0
)

plt.title("Numerical Feature Correlation Matrix")

plt.tight_layout()
plt.show()


# ============================================================
# 11. LOAN AMOUNT DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="disbursed_amount",
    bins=50,
    kde=True
)

plt.title("Disbursed Loan Amount Distribution")
plt.xlabel("Disbursed Amount")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ============================================================
# 12. LTV DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="ltv",
    bins=40,
    kde=True
)

plt.title("LTV Distribution")
plt.xlabel("Loan-to-Value (LTV)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ============================================================
# 13. CREDIT SCORE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="PERFORM_CNS.SCORE",
    bins=50,
    kde=True
)

plt.title("Credit Score Distribution")
plt.xlabel("Credit Score")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ============================================================
# 14. FINAL EDA MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETED")
print("=" * 70)