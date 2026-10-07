import pandas as pd
import joblib

from xgboost import XGBClassifier

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

# The preprocessor was fitted only on X_train.
#
# We reuse the same fitted preprocessor so that all models
# receive exactly the same input representation.

preprocessor = joblib.load(
    "models/preprocessor.joblib"
)


# ============================================================
# 3. TRANSFORM TRAIN / TEST DATA
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
# 4. CALCULATE CLASS IMBALANCE WEIGHT
# ============================================================

# XGBoost does not use class_weight="balanced" like
# Random Forest.
#
# scale_pos_weight gives more importance to the positive
# class (loan default = 1).

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nClass counts:")
print("Non-default:", negative_count)
print("Default    :", positive_count)

print(
    "\nscale_pos_weight:",
    round(scale_pos_weight, 4)
)


# ============================================================
# 5. CREATE XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 6. TRAIN MODEL
# ============================================================

print("\nTraining XGBoost...")

model.fit(
    X_train_processed,
    y_train
)

print("Training completed.")


# ============================================================
# 7. MAKE PREDICTIONS
# ============================================================

# Predicted class.
y_pred = model.predict(
    X_test_processed
)

# Probability of loan default.
y_probability = model.predict_proba(
    X_test_processed
)[:, 1]


# ============================================================
# 8. CALCULATE METRICS
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
# 9. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("XGBOOST RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 12. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/xgboost.joblib"
)

print(
    "\nModel saved to: "
    "models/xgboost.joblib"
)
