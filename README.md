# Banking-Fraud-Detection-ML

This repository contains the code and documentation for a machine learning project focused on improving fraud detection models for e-commerce and bank credit card transactions. The project addresses critical challenges such as class imbalance and the trade-off between over-detection (false positives) and missed fraud (false negatives).

---
## Project Structure

```
.
├── README.md
├── data/
│   ├── Fraud_Data.csv
│   ├── IpAddress_to_Country.csv
│   └── creditcard.csv
│
├── notebooks/ 
│   ├── data_preprocessing.ipynb
|   ├── data_transformation.ipynb
|   ├── Exploratory_data_analysis.ipynb
|   ├── Feature_Engineering.ipynb
│   ├── model_training.ipynb
│   └── model_explainability.ipynb
└── src/
    ├── data_preprocessing.py
    ├── model_training.py
    └── model_explainability.py
```

---

## Overall Project Objective

The overarching goal is to develop accurate and robust fraud detection models that effectively handle the unique challenges of both e-commerce and bank transaction data. This includes leveraging geolocation analysis and transaction pattern recognition, while carefully balancing security needs with user experience by minimizing false positives and preventing financial losses from false negatives.

---

## Table of Contents

- [Task 1: Data Analysis and Preprocessing](#task-1-data-analysis-and-preprocessing)
- [Task 2: Model Building and Training](#task-2-model-building-and-training)
- [Task 3: Model Explainability](#task-3-model-explainability)
- [How to Set Up and Run the Code](#how-to-set-up-and-run-the-code)
- [Project Structure](#project-structure)

---

## Task 1: Data Analysis and Preprocessing

**Objective:**  
Prepare raw transaction data for machine learning model building. This includes ensuring data quality, enriching datasets with new insights, and transforming features into a format suitable for algorithms.

**Datasets Used:**
- `Fraud_Data.csv`: E-commerce transaction data.
- `IpAddress_to_Country.csv`: Mapping of IP address ranges to countries.
- `creditcard.csv`: Bank credit card transaction data.

**Steps Performed:**

1. **Handle Missing Values**
    - Robust handling for missing critical identifiers (`ip_address`, `device_id`), with imputation for `sex` (mode) and `age` (median) as needed.
    - No missing values found in `IpAddress_to_Country.csv` and `creditcard.csv`.

2. **Data Cleaning**
    - Removed 1081 duplicates from `creditcard.csv`; no duplicates in other files.
    - Converted data types as appropriate, including datetime and integer conversions for accurate processing.

3. **Exploratory Data Analysis (EDA)**
    - Univariate and bivariate analysis to reveal dataset characteristics and patterns.
    - Severe class imbalance confirmed in both datasets.
    - Identification of dominant categories and key numerical/categorical distributions.

4. **Merge Datasets for Geolocation**
    - Merged `Fraud_Data.csv` with `IpAddress_to_Country.csv` to enrich with country info.
    - Unmapped IPs (~14.5%) set to `'Unknown'` to prevent data loss.

5. **Feature Engineering**
    - E-commerce: Created `hour_of_day`, `day_of_week`, `time_since_signup`, and transaction frequency features.
    - Bank: Relied on rich PCA-transformed features already present.

6. **Data Transformation**
    - Stratified train-test split (70-30).
    - Imbalance handling: SMOTE for e-commerce, random undersampling for bank data.
    - Standard scaling of numerical features.
    - OneHotEncoding of categorical features for e-commerce data.

---

## Task 2: Model Building and Training

**Objective:**  
Implement and compare two classification models to detect fraudulent transactions, identifying the best-performing model for each dataset using appropriate metrics for imbalanced classes.

**Models Compared:**
- **Logistic Regression:** Simple, interpretable baseline.
- **Random Forest Classifier:** Ensemble method, robust to non-linearities and feature interactions.

Both models used `class_weight='balanced'` to account for class imbalance.

**Workflow:**
- Load preprocessed datasets (`Fraud_Data_merged.csv`, `creditcard_cleaned.csv`).
- Stratified train-test split (70-30).
- Apply class balancing (SMOTE/undersampling).
- Apply scaling and encoding as required.
- Train both models; evaluate on untouched test data.

**Evaluation Results Summary:**

| Dataset        | Metric           | Logistic Regression | Random Forest |
|----------------|------------------|--------------------|--------------|
| **E-commerce** | AUC-PR           | 0.1192             | 0.5588       |
|                | F1-Score         | 0.1792             | 0.6591       |
|                | ROC-AUC          | 0.5560             | 0.7676       |
|                | Precision (1)    | 0.11               | 0.86         |
|                | Recall (1)       | 0.51               | 0.54         |
| **Bank**       | AUC-PR           | 0.3401             | 0.7395       |
|                | F1-Score         | 0.0868             | 0.0889       |
|                | ROC-AUC          | 0.9626             | 0.9742       |
|                | Precision (1)    | 0.05               | 0.05         |
|                | Recall (1)       | 0.88               | 0.88         |

**Justification for Model Selection:**

- **E-commerce:**  
  Random Forest Classifier significantly outperformed Logistic Regression, with much higher AUC-PR and F1-Score for the minority fraud class, and a high precision of 0.86 (minimizing false positives).

- **Bank Transactions:**  
  Random Forest again showed much higher AUC-PR, indicating better probability ranking and greater potential for effective fraud detection with threshold tuning.

**Conclusion:**  
Random Forest Classifier is considered the "best" performing model for both scenarios due to its ensemble nature and superior handling of imbalanced classification.

---

## Task 3: Model Explainability

**Objective:**  
Interpret the best-performing models (Random Forest) using SHAP (Shapley Additive exPlanations) to understand feature importance and impact.

**Methodology:**
- Used `shap.Explainer` on Random Forest models for both datasets.
- Generated SHAP summary and beeswarm plots for global and local feature importance.

**Key Findings:**

- **E-commerce Fraud:**
    - Most impactful features: `ip_address_int`, `age`, `source_SEO`, `purchase_value`, `browser_Chrome`, `country_Unknown`.
    - Higher age, certain IP ranges, SEO/Ads source, low purchase value, Chrome browser, and 'Unknown' country all increased fraud likelihood.

- **Bank Transaction Fraud:**
    - Most influential features: PCA-transformed variables `V14`, `V10`, `V4`, `V12`, and `V17`.
    - V14 was particularly dominant; various feature values pushed predictions toward or away from fraud.

**Interpretation:**  
E-commerce fraud is driven by behavioral, transactional, and geolocation anomalies. Bank fraud is best detected using complex latent patterns from PCA features.

---

## How to Set Up and Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/Banking-Fraud-Detection-ML.git
    cd Banking-Fraud-Detection-ML
    ```

2. **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Place Data Files:**
    - Ensure your `Fraud_Data.csv`, `IpAddress_to_Country.csv`, and `creditcard.csv` files are placed in the `/data/` directory (or update file paths in the scripts).
    - If you have intermediate files, include `Fraud_Data_merged.csv` and `creditcard_cleaned.csv` as well.

5. **Run the Scripts:**
    - The project is structured into logical steps. Run the Python scripts corresponding to each task:
        ```bash
        python script_name.py
        ```
    - If running in a notebook environment (e.g., Jupyter/Colab), execute the code cells sequentially.

---


