import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD FEATURE-ENGINEERED DATA
# ============================================================

df = pd.read_csv(
    "data/processed/feature_engineered_loan_data.csv"
)

print("Dataset shape:", df.shape)


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

# X contains all input variables used by the ML model.
X = df.drop(columns=["loan_default"])

# y contains the target we want to predict.
#
# 0 = Loan will not default
# 1 = Loan will default
y = df["loan_default"]


print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

# 80% of the data is used for training.
# 20% is kept completely separate for final testing.
#
# stratify=y ensures that the proportion of defaulters
# remains approximately the same in both datasets.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. DISPLAY DATASET SIZES
# ============================================================

print("\nTrain/Test Split")
print("-" * 50)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# 5. CHECK TARGET DISTRIBUTION
# ============================================================

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).sort_index())

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).sort_index())


# ============================================================
# 6. SAVE SPLIT DATA
# ============================================================

X_train.to_csv(
    "data/processed/X_train.csv",
    index=False
)

X_test.to_csv(
    "data/processed/X_test.csv",
    index=False
)

y_train.to_csv(
    "data/processed/y_train.csv",
    index=False
)

y_test.to_csv(
    "data/processed/y_test.csv",
    index=False
)


print("\nTrain/test datasets saved successfully.")