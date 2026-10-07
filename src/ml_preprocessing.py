import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

# We use X_train only to learn preprocessing parameters.
# This is important because the test data must remain unseen.

X_train = pd.read_csv(
    "data/processed/X_train.csv"
)

print("X_train shape:", X_train.shape)


# ============================================================
# 2. IDENTIFY COLUMN TYPES
# ============================================================

# Numerical columns are processed differently from
# categorical/text columns.

numerical_columns = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object"]
).columns.tolist()


print("\nNumber of numerical columns:", len(numerical_columns))
print("Number of categorical columns:", len(categorical_columns))

print("\nCategorical columns:")
print(categorical_columns)


# ============================================================
# 3. NUMERICAL PREPROCESSING
# ============================================================

# Some numerical columns still contain NaN values.
#
# Median imputation is commonly used for numerical features
# because it is less sensitive to extreme outliers than mean.

# numerical_pipeline = Pipeline(
#     steps=[
#         (
#             "imputer",
#             SimpleImputer(strategy="median")
#         )
#     ]
# )

numerical_pipeline = Pipeline(
    steps=[
        # Fill missing numerical values with the median.
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        # Put numerical features on a comparable scale.
        # This helps Logistic Regression converge properly.
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# 4. CATEGORICAL PREPROCESSING
# ============================================================

# Categorical columns such as Employment.Type cannot be
# directly passed to most ML algorithms.
#
# OneHotEncoder converts categories into binary columns.
#
# handle_unknown="ignore" is important for production:
# if FastAPI receives a category that wasn't present during
# training, the model will not crash.

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ]
)


# ============================================================
# 5. COMBINE PREPROCESSING
# ============================================================

# ColumnTransformer applies the correct transformation
# to each type of feature.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# ============================================================
# 6. FIT PREPROCESSOR ON TRAINING DATA
# ============================================================

# IMPORTANT:
# We fit ONLY on X_train.
#
# We do NOT fit on X_test because that would introduce
# data leakage.

X_train_processed = preprocessor.fit_transform(X_train)


# ============================================================
# 7. DISPLAY RESULT
# ============================================================

print("\nPreprocessing completed successfully.")

print(
    "Processed X_train shape:",
    X_train_processed.shape
)


# ============================================================
# 8. SAVE PREPROCESSOR
# ============================================================

# Save the fitted preprocessing object.
#
# Later, FastAPI can load this same object and apply exactly
# the same transformations to a new loan application.

import joblib

joblib.dump(
    preprocessor,
    "models/preprocessor.joblib"
)

print(
    "\nPreprocessor saved to: "
    "models/preprocessor.joblib"
)