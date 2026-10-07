import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. LOAD TEST DATA
# ============================================================

X_test = pd.read_csv(
    "data/processed/X_test.csv"
)

y_test = pd.read_csv(
    "data/processed/y_test.csv"
).squeeze()


# ============================================================
# 2. LOAD PREPROCESSOR AND TUNED MODEL
# ============================================================

preprocessor = joblib.load(
    "models/preprocessor.joblib"
)

model = joblib.load(
    "models/xgboost_tuned.joblib"
)


# ============================================================
# 3. PREPROCESS TEST DATA
# ============================================================

X_test_processed = preprocessor.transform(
    X_test
)


# ============================================================
# 4. GET DEFAULT PROBABILITIES
# ============================================================

# Instead of immediately predicting 0 or 1,
# we first obtain the probability of default.

default_probability = model.predict_proba(
    X_test_processed
)[:, 1]


# ============================================================
# 5. TEST DIFFERENT CLASSIFICATION THRESHOLDS
# ============================================================

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]


print("\n" + "=" * 75)
print("THRESHOLD ANALYSIS")
print("=" * 75)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'False Negatives':<18}"
)

print("-" * 75)


for threshold in thresholds:

    # Convert probabilities into binary predictions.
    #
    # Example:
    # threshold = 0.40
    # probability >= 0.40 -> default
    # probability < 0.40  -> non-default

    y_pred = (
        default_probability >= threshold
    ).astype(int)


    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    # Confusion matrix:
    #
    # [[TN, FP],
    #  [FN, TP]]

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    false_negatives = cm[1, 0]


    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{false_negatives:<18}"
    )


print("\nThreshold analysis completed.")
