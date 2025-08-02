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
