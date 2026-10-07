import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
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
# 1. LOAD TRAIN / TEST DATA
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

# Use the same preprocessing that was fitted on X_train.
preprocessor = joblib.load(
    "models/preprocessor.joblib"
)


# ============================================================
# 3. TRANSFORM DATA
# ============================================================

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
# 4. CREATE RANDOM FOREST
# ============================================================

# Random Forest can capture nonlinear relationships and
# feature interactions that Logistic Regression cannot easily
# capture.
#
# class_weight="balanced" gives additional importance to the
# minority default class.

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 5. TRAIN MODEL
# ============================================================

print("\nTraining Random Forest...")

model.fit(
    X_train_processed,
    y_train
)

print("Training completed.")


# ============================================================
# 6. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test_processed
)

# Probability that the loan will default.
y_probability = model.predict_proba(
    X_test_processed
)[:, 1]


# ============================================================
# 7. CALCULATE METRICS
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
print("RANDOM FOREST RESULTS")
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
    "models/random_forest.joblib"
)

print(
    "\nModel saved to: "
    "models/random_forest.joblib"
)
