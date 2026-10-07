import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD TRAINING AND TEST DATA
# ============================================================

X_train = pd.read_csv(
    "data/processed/X_train.csv"
)

X_test = pd.read_csv(
    "data/processed/X_test.csv"
)

y_train = pd.read_csv(
    "data/processed/y_train.csv"
).squeeze()

y_test = pd.read_csv(
    "data/processed/y_test.csv"
).squeeze()


print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# ============================================================
# 2. LOAD FITTED PREPROCESSOR
# ============================================================

# We already fitted the preprocessing pipeline using X_train.
#
# We load the saved object instead of fitting it again.

preprocessor = joblib.load(
    "models/preprocessor.joblib"
)


# ============================================================
# 3. TRANSFORM TRAINING AND TEST DATA
# ============================================================

# IMPORTANT:
# Do NOT call fit_transform() on X_test.
#
# The preprocessor has already learned everything from X_train.
# Therefore:
#
# X_train -> transform
# X_test  -> transform

X_train_processed = preprocessor.transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


print("\nProcessed training shape:")
print(X_train_processed.shape)

print("\nProcessed testing shape:")
print(X_test_processed.shape)


# ============================================================
# 4. CREATE LOGISTIC REGRESSION MODEL
# ============================================================

# Logistic Regression is our baseline classification model.
#
# class_weight="balanced":
# The dataset contains fewer default cases than non-default
# cases. This option gives more importance to the minority
# class during training.

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)


# ============================================================
# 5. TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression...")

model.fit(
    X_train_processed,
    y_train
)

print("Training completed.")


# ============================================================
# 6. MAKE PREDICTIONS
# ============================================================

# Predicted class: 0 or 1
y_pred = model.predict(
    X_test_processed
)

# Probability of class 1 (default).
#
# ROC-AUC requires probability scores rather than
# only predicted class labels.

y_probability = model.predict_proba(
    X_test_processed
)[:, 1]


# ============================================================
# 7. CALCULATE EVALUATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 11. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/logistic_regression.joblib"
)

print(
    "\nModel saved to: "
    "models/logistic_regression.joblib"
)