import pandas as pd
import joblib

from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
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
# 2. LOAD PREPROCESSOR
# ============================================================

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


# ============================================================
# 4. CALCULATE CLASS WEIGHT
# ============================================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print(
    "\nscale_pos_weight:",
    round(scale_pos_weight, 4)
)


# ============================================================
# 5. CREATE BASE XGBOOST MODEL
# ============================================================

base_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 6. DEFINE HYPERPARAMETER SEARCH SPACE
# ============================================================

param_distributions = {

    # Number of boosting trees
    "n_estimators": [
        200,
        300,
        400,
        500
    ],

    # Maximum depth of each tree
    "max_depth": [
        3,
        4,
        5,
        6,
        7
    ],

    # Step size used during boosting
    "learning_rate": [
        0.01,
        0.03,
        0.05,
        0.1
    ],

    # Percentage of training rows used by each tree
    "subsample": [
        0.7,
        0.8,
        0.9,
        1.0
    ],

    # Percentage of features used by each tree
    "colsample_bytree": [
        0.7,
        0.8,
        0.9,
        1.0
    ],

    # Minimum sum of instance weight needed in a child
    "min_child_weight": [
        1,
        3,
        5,
        10
    ]
}


# ============================================================
# 7. RANDOMIZED SEARCH
# ============================================================

# RandomizedSearchCV tests a limited number of combinations
# instead of testing every possible combination.
#
# scoring="roc_auc" means we select the model that gives the
# best ranking ability for default vs non-default loans.

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=15,
    scoring="roc_auc",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 8. RUN HYPERPARAMETER SEARCH
# ============================================================

print("\nStarting XGBoost hyperparameter tuning...")
print("This may take some time because the dataset is large.")


random_search.fit(
    X_train_processed,
    y_train
)


# ============================================================
# 9. DISPLAY BEST PARAMETERS
# ============================================================

print("\n" + "=" * 60)
print("BEST XGBOOST PARAMETERS")
print("=" * 60)

print(
    random_search.best_params_
)

print(
    "\nBest Cross-Validation ROC-AUC:",
    round(random_search.best_score_, 4)
)


# ============================================================
# 10. GET BEST MODEL
# ============================================================

best_model = random_search.best_estimator_


# ============================================================
# 11. EVALUATE ON UNSEEN TEST DATA
# ============================================================

y_pred = best_model.predict(
    X_test_processed
)

y_probability = best_model.predict_proba(
    X_test_processed
)[:, 1]


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
# 12. DISPLAY FINAL TUNED RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TUNED XGBOOST TEST RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 13. SAVE TUNED MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/xgboost_tuned.joblib"
)

print(
    "\nTuned model saved to: "
    "models/xgboost_tuned.joblib"
)
