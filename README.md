# Banking-Fraud-Detection-ML

# Task 1: Data Analysis and Preprocessing

This document outlines the steps undertaken for Task 1 of the Banking Fraud Detection ML project, focusing on data analysis, cleaning, feature engineering, and transformation for both the e-commerce (`Fraud_Data.csv`) and bank transaction (`creditcard.csv`) datasets.

---

## Objective

The primary objective of Task 1 is to prepare raw transaction data for machine learning model building. This includes ensuring data quality, enriching datasets with new insights, and transforming features into a format suitable for algorithms.

---

## Datasets Used

- **Fraud_Data.csv:** E-commerce transaction data.
- **IpAddress_to_Country.csv:** Mapping of IP address ranges to countries.
- **creditcard.csv:** Bank credit card transaction data.

---

## Steps Performed

### 1. Handle Missing Values

- **Fraud_Data.csv:**
  - No explicit missing values found in initial checks.
  - If `ip_address` or `device_id` had been missing, those rows would have been dropped.
  - If missing, `sex` would be imputed with the mode, and `age` with the median.
- **IpAddress_to_Country.csv & creditcard.csv:** No missing values found.

### 2. Data Cleaning

- **Remove Duplicates:**
  - `Fraud_Data.csv` & `IpAddress_to_Country.csv`: No duplicates.
  - `creditcard.csv`: 1081 duplicate rows removed.
- **Correct Data Types:**
  - `Fraud_Data.csv`: `signup_time` and `purchase_time` converted to datetime; `ip_address` converted to integer (`ip_address_int`).
  - `IpAddress_to_Country.csv`: IP bounds ensured as integers.
  - `creditcard.csv`: Data types appropriate.

### 3. Exploratory Data Analysis (EDA)

- **Fraud_Data.csv:**
  - **Numerical:** `purchase_value` is right-skewed; `age` is bell-shaped.
  - **Categorical:** Dominant categories for `browser` and `source`.
  - **Class Imbalance:** ~90.6% non-fraud vs. 9.4% fraud.
  - **Bivariate:** No strong separation by individual features.
- **creditcard.csv:**
  - **Numerical:** `Time` has two frequency peaks; `Amount` is heavily right-skewed.
  - **Class Imbalance:** Extreme; very few fraud cases.
  - **Bivariate:** Fraudulent transactions tend to have lower `Amount`; different temporal (`Time`) patterns.
  - **Correlation:** Low among PCA `V` features, but some correlation with `Amount`.

### 4. Merge Datasets for Geolocation Analysis

- `Fraud_Data.csv` was enriched by merging with `IpAddress_to_Country.csv` using `ip_address_int`.
- Custom function mapped each IP address to a country.
- About 14.5% of IPs unmapped; set to `'Unknown'` to retain data.

### 5. Feature Engineering

- **Fraud_Data.csv:**
  - **Time-Based Features:**
    - `hour_of_day`, `day_of_week`, `time_since_signup`
  - **Transaction Frequency/Velocity Features:**
    - `user_transaction_count`, `device_transaction_count`, `ip_transaction_count`
- **creditcard.csv:**
  - No new features; existing PCA V features, `Time`, and `Amount` deemed sufficient.

### 6. Data Transformation

- **Train-Test Split:** 70% training, 30% testing, stratified by class.
- **Handle Class Imbalance:**
  - `Fraud_Data.csv`: SMOTE applied to training data; OneHotEncode categorical features before SMOTE.
  - `creditcard.csv`: Random Undersampler applied to training data.
- **Normalization and Scaling:** StandardScaler fitted on training data and applied to test data.
- **Encode Categorical Features:**
  - `Fraud_Data.csv`: OneHotEncoder for categorical features before/after SMOTE as appropriate.
  - `creditcard.csv`: No categorical encoding required.


## Task 2: Model Building and Training

**Objective:**  
Implement and compare two classification models to detect fraudulent transactions, identifying the best-performing model for each dataset using appropriate metrics for imbalanced classes.

**Models Compared:**
- **Logistic Regression:** Simple, interpretable baseline
- **Random Forest Classifier:** Ensemble method, robust to non-linearities and feature interactions

Both models used `class_weight='balanced'` to mitigate class imbalance.

**Workflow:**
- Load preprocessed datasets (`Fraud_Data_merged.csv`, `creditcard_cleaned.csv`)
- Stratified train-test split (70-30)
- Imbalance handling (SMOTE for e-commerce, random undersampling for bank data)
- Scaling and encoding as required
- Train both models on resampled training data
- Evaluate on untouched test data using AUC-PR, F1-Score, ROC-AUC, and confusion matrix

**Results Summary:**

| Metric          | Logistic Regression | Random Forest |
|-----------------|--------------------|--------------|
|         **E-commerce (Fraud_Data.csv)**         |                    |              |
| AUC-PR          | 0.1192             | 0.5588       |
| F1-Score        | 0.1792             | 0.6591       |
| ROC-AUC         | 0.5560             | 0.7676       |
| Precision       | 0.11               | 0.86         |
| Recall          | 0.51               | 0.54         |
|         **Bank Transactions (creditcard.csv)**         |                    |              |
| AUC-PR          | 0.3401             | 0.7395       |
| F1-Score        | 0.0868             | 0.0889       |
| ROC-AUC         | 0.9626             | 0.9742       |
| Precision       | 0.05               | 0.05         |
| Recall          | 0.88               | 0.88         |

**Model Selection Justification:**

- **E-commerce:**  
  The Random Forest Classifier significantly outperformed Logistic Regression in both AUC-PR and F1-Score, demonstrating a much better balance between precision and recall for fraud detection. Its high precision (0.86) is especially valuable for minimizing false positives.

- **Bank Transactions:**  
  Random Forest again achieved a much higher AUC-PR (0.7395 vs 0.3401) than Logistic Regression, indicating better probability ranking. While F1-scores and precision were similar, Random Forest's higher AUC-PR suggests better performance with threshold tuning.

**Conclusion:**  
In both scenarios, the Random Forest Classifier is considered the best-performing model. Its ensemble nature allows it to capture complex patterns and handle imbalanced data more effectively, as reflected in superior AUC-PR and F1-scores.
