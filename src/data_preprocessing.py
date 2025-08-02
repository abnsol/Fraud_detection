import pandas as pd
import numpy as np

print("Loading datasets...")
try:
    fraud_data = pd.read_csv('Fraud_Data.csv')
    ip_to_country = pd.read_csv('IpAddress_to_Country.csv')
    creditcard_data = pd.read_csv('creditcard.csv')
    print("Datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Please ensure the CSV files are in the same directory as this script.")
    exit()

print("\n--- Initial Data Info ---")
print("\nFraud_Data.csv Info:")
fraud_data.info()
print("\nIpAddress_to_Country.csv Info:")
ip_to_country.info()
print("\ncreditcard.csv Info:")
creditcard_data.info()

print("\n--- Handling Missing Values ---")

print("\nProcessing Fraud_Data.csv for missing values...")
print("Missing values before handling:\n", fraud_data.isnull().sum())


initial_rows_fraud = fraud_data.shape[0]
fraud_data.dropna(subset=['ip_address', 'device_id'], inplace=True)
print(f"Dropped {initial_rows_fraud - fraud_data.shape[0]} rows from Fraud_Data due to missing ip_address or device_id.")

# Impute 'sex' (categorical) with mode
if fraud_data['sex'].isnull().any():
    mode_sex = fraud_data['sex'].mode()[0]
    fraud_data['sex'].fillna(mode_sex, inplace=True)
    print(f"Imputed missing 'sex' values with mode: {mode_sex}")

# Impute 'age' (numerical) with median
if fraud_data['age'].isnull().any():
    median_age = fraud_data['age'].median()
    fraud_data['age'].fillna(median_age, inplace=True)
    print(f"Imputed missing 'age' values with median: {median_age}")

print("Missing values after handling for Fraud_Data.csv:\n", fraud_data.isnull().sum())


print("\nProcessing IpAddress_to_Country.csv for missing values...")
print("Missing values before handling:\n", ip_to_country.isnull().sum())
print("Missing values after handling for IpAddress_to_Country.csv:\n", ip_to_country.isnull().sum())


print("\nProcessing creditcard.csv for missing values...")
print("Missing values before handling:\n", creditcard_data.isnull().sum())
print("Missing values after handling for creditcard.csv:\n", creditcard_data.isnull().sum())
