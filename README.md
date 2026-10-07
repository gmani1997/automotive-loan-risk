# Automotive Loan Credit Risk & Default Prediction

An end-to-end machine learning project for predicting the probability of **automotive loan default** and classifying loan applications into risk categories.

The project demonstrates a production-oriented data science workflow covering:

* Data ingestion from MySQL
* Data validation and preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Statistical and machine learning modeling
* Model comparison and hyperparameter tuning
* Classification threshold optimization
* Model serialization
* FastAPI prediction API
* Streamlit user interface
* Git/GitHub version control

## Business Objective

The objective is to help an automotive lending organization identify customers who have a higher probability of defaulting on their vehicle loan.

The model produces:

* **Default Probability**
* **Risk Category**
* **Prediction**
* **Business Decision**

This can support credit-risk teams in prioritizing applications for further review and improving risk-based lending decisions.



## Dataset

The project uses the **L&T Vehicle Loan Default Prediction** dataset.

The training dataset contains:

* **233,154 records**
* **41 original columns**
* Target variable: `loan_default`

Target distribution:

| Target | Meaning     | Records | Percentage |
| ------ | ----------- | ------: | ---------: |
| 0      | Non-Default | 182,543 |     78.29% |
| 1      | Default     |  50,611 |     21.71% |

The dataset contains information related to:

* Loan and asset details
* Loan-to-value ratio
* Customer employment
* Credit bureau history
* Primary and secondary credit accounts
* Previous loan activity
* Delinquencies and overdue accounts
* Credit history length
* Customer document verification

## End-to-End Architecture

```text
                ┌─────────────────────┐
                │   Loan Dataset      │
                │     train.csv       │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │       MySQL         │
                │   automotive_risk   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Data Validation &   │
                │   Preprocessing     │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Feature Engineering │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Train/Test Split    │
                │    Stratified       │
                └──────────┬──────────┘
                           │
                           ▼
        ┌────────────────────────────────────┐
        │       Machine Learning Models       │
        │                                    │
        │ Logistic Regression                │
        │ Random Forest                      │
        │ XGBoost                            │
        └──────────────────┬─────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Model Evaluation &   │
                │ Hyperparameter Tune  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Tuned XGBoost Model │
                │   + Preprocessor    │
                └──────────┬──────────┘
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
        ┌─────────────────┐  ┌─────────────────┐
        │    FastAPI      │  │    Streamlit    │
        │ Prediction API  │  │   Web Interface │
        └─────────────────┘  └─────────────────┘
```

## Technology Stack

### Programming & Data

* Python
* Pandas
* NumPy
* SQL
* MySQL
* SQLAlchemy

### Machine Learning

* Scikit-learn
* XGBoost
* Logistic Regression
* Random Forest
* Classification metrics
* Hyperparameter tuning
* Classification threshold optimization

### API & Application

* FastAPI
* Uvicorn
* Pydantic
* Streamlit

### Model Management

* Joblib

### Development & Version Control

* VS Code
* Git
* GitHub
* Python virtual environment

### Deployment Preparation

* Docker
* Dockerfile
* Google Cloud CLI / GCP Cloud Run configuration


## Data Processing

The raw loan data was loaded from MySQL and validated before model development.

### Data Validation

The following checks were performed:

* Dataset shape and column validation
* Missing-value analysis
* Duplicate-row detection
* Duplicate `UniqueID` detection
* Target-value validation
* Negative-value detection in financial fields
* Constant-column detection
* Data-type validation
* Outlier and skewness inspection

Validation results included:

* **233,154 rows**
* **0 duplicate rows**
* **0 duplicate UniqueID values**
* `Employment.Type` contained missing values, which were handled during preprocessing
* `loan_default` contained only binary values: `0` and `1`
* Negative current-balance values were treated as invalid and converted to missing values for downstream imputation

### Data Preprocessing

The preprocessing pipeline included:

1. Missing-value handling
2. Date conversion
3. Employment-type handling
4. Financial-value validation
5. Duration conversion from formats such as `1yrs 11mon` into months
6. Removal of identifier columns that were not suitable as predictive features
7. Removal of constant features
8. Numerical imputation using the median
9. Categorical imputation using the most frequent category
10. One-hot encoding of categorical variables
11. Feature scaling for numerical variables

The preprocessing transformation was fitted **only on the training data** to prevent data leakage.

The trained preprocessing pipeline was saved as:

```text
models/preprocessor.joblib
```

## Feature Engineering

Business-oriented features were created from the original loan and credit attributes.

### Loan Features

* `Loan_Asset_Ratio`
* `Age`
* `Disbursal_Year`
* `Disbursal_Month`

### Primary Credit Features

* `Primary_Active_Ratio`
* `Primary_Overdue_Ratio`

### Secondary Credit Features

* `Secondary_Active_Ratio`
* `Secondary_Overdue_Ratio`

### Aggregated Credit Features

* `Total_Accounts`
* `Total_Active_Accounts`
* `Total_Overdue_Accounts`
* `Total_Current_Balance`
* `Total_Sanctioned_Amount`
* `Total_Previous_Disbursed_Amount`

### Credit History Features

* `Total_Credit_History_Months`
* `Average_Account_Age_Months`
* `Recent_Credit_Activity`

These features were designed to represent customer credit exposure, repayment behavior, existing obligations, and recent credit activity in a form suitable for predictive modeling.

After feature engineering, the dataset contained **48 columns**, including the target variable, resulting in **47 model features**.

## Train/Test Split

The data was divided using a stratified train/test split:

* Training data: **80%**
* Test data: **20%**
* Random state: `42`
* Stratification: based on `loan_default`

Final shapes:

```text
X_train: 186,523 × 47
X_test:   46,631 × 47
y_train: 186,523
y_test:   46,631
```

Stratification was used to preserve the default/non-default class distribution in both datasets.


## Model Development

Three classification algorithms were evaluated for predicting automotive loan default:

1. Logistic Regression
2. Random Forest
3. XGBoost

Because the target variable is imbalanced, class weighting / class imbalance handling was incorporated during model development.

### Model Comparison

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   0.5622 |    0.2773 | 0.6330 |   0.3857 |  0.6245 |
| Random Forest       |   0.5875 |    0.2880 | 0.6117 |   0.3917 |  0.6343 |
| XGBoost             |   0.5824 |    0.2888 | 0.6318 |   0.3964 |  0.6408 |
| Tuned XGBoost       |   0.5772 |    0.2877 | 0.6422 |   0.3974 |  0.6408 |

The tuned XGBoost model was selected as the final prediction model because it provided the strongest overall balance among the evaluated models, particularly for **F1 score and recall**, while maintaining the highest ROC-AUC observed in testing.

## XGBoost Hyperparameter Tuning

Randomized cross-validation was used to tune the XGBoost model.

Configuration:

* Search method: `RandomizedSearchCV`
* Cross-validation: 3-fold
* Number of candidates: 15
* Optimization metric: ROC-AUC
* Random state: `42`

Best parameters identified:

```text id="8ks1e7"
subsample = 0.8
n_estimators = 200
min_child_weight = 1
max_depth = 5
learning_rate = 0.05
colsample_bytree = 0.7
```

Best cross-validation ROC-AUC:

```text
0.6456
```

The final tuned model was saved as:

```text id="9kq2fj"
models/xgboost_tuned.joblib
```

## Classification Threshold Optimization

The default classification threshold of `0.50` was evaluated along with alternative thresholds.

| Threshold |  Precision |     Recall |   F1 Score |
| --------: | ---------: | ---------: | ---------: |
|      0.30 |     0.2323 |     0.9665 |     0.3745 |
|      0.35 |     0.2427 |     0.9176 |     0.3838 |
|      0.40 |     0.2569 |     0.8543 |     0.3950 |
|  **0.45** | **0.2699** | **0.7611** | **0.3985** |
|      0.50 |     0.2877 |     0.6422 |     0.3974 |
|      0.55 |     0.3088 |     0.4857 |     0.3775 |
|      0.60 |     0.3442 |     0.2732 |     0.3046 |

A threshold of **0.45** was selected as the operating threshold because it produced the highest F1 score among the tested thresholds while maintaining stronger recall.

This threshold is a **business operating choice**, not a universal optimum. In a real lending environment, the threshold would be determined using the relative cost of false positives versus false negatives, credit policy, approval rates, and expected financial loss.

## Final Model Output

For each loan application, the prediction service returns:

```json
{
  "default_probability": 0.5874,
  "prediction": 1,
  "risk_category": "High Risk",
  "decision": "Potential Default",
  "threshold": 0.45
}
```

Where:

* `default_probability` = estimated probability of loan default
* `prediction = 1` = classified as potential default
* `prediction = 0` = classified as potential non-default
* `risk_category` = business-friendly risk classification
* `decision` = model-based decision signal
* `threshold` = probability threshold used for classification



## Prediction API

The trained model is exposed through a **FastAPI REST API**.

The API separates the machine learning logic from the user interface and provides a reusable prediction endpoint.

### API Workflow

```text id="8p3k7c"
Client Request
      │
      ▼
FastAPI /predict
      │
      ▼
Pydantic Input Validation
      │
      ▼
Convert API Fields
to Training Feature Names
      │
      ▼
Feature Engineering
      │
      ▼
Saved Preprocessor
      │
      ▼
Tuned XGBoost Model
      │
      ▼
Default Probability
      │
      ▼
Threshold = 0.45
      │
      ▼
Risk Category + Decision
      │
      ▼
JSON Response
```

### Available Endpoints

#### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy"
}
```

#### Loan Risk Prediction

```http
POST /predict
```

The endpoint accepts customer, loan, credit-history, and document-verification information and returns the predicted default probability and risk classification.

Example response:

```json
{
  "default_probability": 0.5874,
  "prediction": 1,
  "risk_category": "High Risk",
  "decision": "Potential Default",
  "threshold": 0.45
}
```

### Interactive API Documentation

FastAPI automatically provides Swagger documentation at:

```text
http://127.0.0.1:8000/docs
```

This allows users or developers to test the `/predict` endpoint interactively.

## Streamlit Application

A Streamlit web interface was developed on top of the FastAPI prediction service.

The UI collects:

* Loan information
* Customer information
* Credit bureau information
* Primary credit account details
* Secondary credit account details
* Installment information
* Credit history
* Document verification information

When the user clicks **Predict Loan Risk**, Streamlit sends the input to the FastAPI `/predict` endpoint.

The API processes the request using the same preprocessing pipeline and trained XGBoost model used during model development.

### Application Flow

```text id="x7jv3q"
User
  │
  ▼
Streamlit UI
  │
  │ HTTP POST /predict
  ▼
FastAPI
  │
  ▼
Feature Engineering
  │
  ▼
Preprocessor
  │
  ▼
XGBoost Model
  │
  ▼
Prediction
  │
  ▼
FastAPI JSON Response
  │
  ▼
Streamlit Risk Display
```

The Streamlit application displays:

* Default Probability
* Risk Category
* Prediction Decision
* Classification Threshold

## Local Application Setup

### Start FastAPI

From the project root:

```bash
uvicorn api.main:app --reload --port 8000
```

FastAPI will be available at:

```text
http://127.0.0.1:8000
```

### Start Streamlit

Open another terminal and run:

```bash
streamlit run streamlit_app/app.py
```

The Streamlit application will be available at:

```text
http://localhost:8501
```

The Streamlit application communicates with the FastAPI backend rather than loading the model



## Project Structure

```text id="v4c8f2"
automotive-loan-risk/
│
├── api/
│   └── main.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── logistic_regression.joblib
│   ├── preprocessor.joblib
│   ├── xgboost.joblib
│   └── xgboost_tuned.joblib
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── database.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── import_data.py
│   ├── ml_preprocessing.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── threshold_analysis.py
│   ├── train_logistic_regression.py
│   ├── train_random_forest.py
│   ├── train_test_split.py
│   ├── train_xgboost.py
│   ├── tune_xgboost.py
│   ├── validate_api_predictions.py
│   └── validate_data.py
│
├── streamlit_app/
│   └── app.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

## Installation

### 1. Clone the Repository

```bash id="j5xk7m"
git clone https://github.com/gmani1997/automotive-loan-risk.git
cd automotive-loan-risk
```

### 2. Create a Virtual Environment

Windows:

```bash id="5o4f7x"
python -m venv loanenv
loanenv\Scripts\activate
```

### 3. Install Dependencies

```bash id="f8y2zq"
pip install -r requirements.txt
```

### 4. Configure Database Connection

Create a `.env` file in the project root:

```env id="m3n5bx"
DB_HOST=localhost
DB_PORT=3306
DB_NAME=automotive_risk
DB_USER=root
DB_PASSWORD=your_mysql_password
```

The `.env` file is intentionally excluded from Git to protect database credentials.

### 5. Prepare the Database

Create the MySQL database:

```sql id="2x4y8k"
CREATE DATABASE automotive_risk;
```

Import the training dataset into the `loan_data` table.

The Python database layer uses SQLAlchemy and PyMySQL to connect to MySQL.

### 6. Run the Prediction API

```bash id="x6p4mz"
uvicorn api.main:app --reload --port 8000
```

Open the API documentation:

```text id="n4r7kx"
http://127.0.0.1:8000/docs
```

### 7. Run the Streamlit Application

In a second terminal:

```bash id="w9t3qa"
streamlit run streamlit_app/app.py
```

Open:

```text id="j7v2mc"
http://localhost:8501
```

## Reproducibility

The project separates the machine learning workflow into independent stages:

```text id="a8f6yc"
Data Loading
     ↓
Validation
     ↓
Preprocessing
     ↓
EDA
     ↓
Feature Engineering
     ↓
Train/Test Split
     ↓
ML Preprocessing
     ↓
Model Training
     ↓
Hyperparameter Tuning
     ↓
Threshold Analysis
     ↓
Model Serialization
     ↓
FastAPI Prediction
```

Each stage is implemented as a separate Python module under `src/`.

## Model Artifacts

The following trained artifacts are included in the repository:

* `preprocessor.joblib`
* `logistic_regression.joblib`
* `xgboost.joblib`
* `xgboost_tuned.joblib`

The large Random Forest model is intentionally excluded from the repository to keep the Git repository lightweight.

## Data Privacy and Repository Hygiene

The following files and directories are excluded from version control:

* `.env`
* `loanenv/`
* `data/raw/`
* `data/processed/`
* `.vscode/`
* Python cache files
* Logs
* Large unused model artifacts

This prevents credentials, local environments, and large datasets from being accidentally committed.


## Business Interpretation

The model is intended to support, not replace, credit-risk decision-making.

A lending organization can use the predicted default probability to prioritize applications for different levels of review.

A possible operational strategy is:

```text id="b9x4qn"
Low predicted risk
       ↓
Standard processing

Higher predicted risk
       ↓
Additional credit-risk review

Very high predicted risk
       ↓
Potential manual review / stricter credit policy
```

The exact business thresholds should be determined using historical loss data, credit policy, approval-rate targets, and the financial cost of false positives and false negatives.

### Why Recall Matters

In loan default prediction, a false negative means:

> A customer who is actually likely to default is classified as non-default.

This can expose the lender to potential financial loss.

Therefore, recall is an important metric for identifying potentially risky borrowers.

However, maximizing recall alone can produce too many false positives. This is why the project evaluates **precision, recall, F1 score, ROC-AUC, and classification thresholds** together.

### Model Evaluation Considerations

The model achieved a test ROC-AUC of approximately **0.641**.

This indicates that the model has meaningful predictive signal, but it is not a perfect predictor.

In a real production credit-risk system, additional work would be required, including:

* Probability calibration
* Cost-sensitive evaluation
* Population Stability Index (PSI)
* Data drift monitoring
* Model drift monitoring
* Fairness and bias assessment
* Explainability using techniques such as SHAP
* Periodic model retraining
* Champion/challenger model monitoring
* Business-policy validation

## Key Business Features

The feature engineering process focuses on characteristics that can represent credit risk, including:

* Loan-to-asset relationship
* Existing credit exposure
* Number of active accounts
* Number of overdue accounts
* Recent credit activity
* Previous disbursed amounts
* Credit history length
* Average account age
* Customer age
* Credit bureau information

These features allow the model to capture multiple dimensions of a customer's existing credit profile rather than relying only on the requested loan amount.

## Interview Discussion Points

### Why was XGBoost selected?

XGBoost was selected after comparing Logistic Regression, Random Forest, and XGBoost.

It provided the strongest overall performance in terms of ROC-AUC and F1 score among the evaluated models and was therefore selected for the final prediction pipeline.

### Why was the threshold changed from 0.50 to 0.45?

The classification threshold was evaluated because the business problem is sensitive to missed defaults.

A threshold of `0.45` provided the highest F1 score among the tested thresholds while maintaining stronger recall.

The threshold can be changed later according to business cost and risk appetite.

### How was data leakage prevented?

Data leakage was controlled by:

1. Splitting the data into training and test sets before fitting the preprocessing pipeline.
2. Fitting imputers, scalers, and encoders only on training data.
3. Applying the fitted preprocessing pipeline to the test data.
4. Keeping the test set isolated from model training and hyperparameter optimization.

### How would you improve this model in production?

Potential improvements include:

* Better probability calibration
* Cost-sensitive learning
* SHAP-based model explainability
* Hyperparameter optimization with a larger search space
* Cross-validation with temporal validation where appropriate
* Model and data drift monitoring
* Automated retraining pipelines
* More business-specific loss functions
* Credit policy integration
* Monitoring approval rate and bad-rate by risk segment

## Limitations

This project is designed as an end-to-end portfolio and interview demonstration.

The current model should not be treated as a production lending decision engine without additional validation.

Important limitations include:

* The dataset represents historical lending behavior.
* Model performance may change on new populations.
* ROC-AUC is moderate rather than extremely high.
* The probability output has not been formally calibrated.
* The threshold is based on the evaluated dataset rather than an organization's actual cost matrix.
* External economic variables and macroeconomic conditions are not included.
* Production deployment, authentication, monitoring, and automated retraining would require additional infrastructure.

## Future Enhancements

Possible extensions include:

* SHAP model explanations
* MLflow experiment tracking
* Model registry
* Automated CI/CD
* GCP deployment
* Cloud-based monitoring
* Data drift detection
* Model drift detection
* Model calibration
* Power BI risk dashboards
* Automated model retraining
* Role-based API authentication
* Docker-based deployment
