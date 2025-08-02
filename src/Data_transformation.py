X_fraud = fraud_data.drop(columns=['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 'class'])
y_fraud = fraud_data['class']

X_creditcard = creditcard_data.drop(columns=['Time', 'Class'])
y_creditcard = creditcard_data['Class']

X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(
    X_fraud, y_fraud, test_size=0.3, random_state=42, stratify=y_fraud
)
print(f"Fraud_Data train shape: {X_fraud_train.shape}, test shape: {X_fraud_test.shape}")

X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test = train_test_split(
    X_creditcard, y_creditcard, test_size=0.3, random_state=42, stratify=y_creditcard
)
print(f"Creditcard_Data train shape: {X_creditcard_train.shape}, test shape: {X_creditcard_test.shape}")


print(f"Original Fraud_Data training set shape: {Counter(y_fraud_train)}")
smote = SMOTE(random_state=42)
X_fraud_train_res, y_fraud_train_res = smote.fit_resample(X_fraud_train.select_dtypes(include=np.number), y_fraud_train)
print(f"Resampled Fraud_Data training set shape (SMOTE): {Counter(y_fraud_train_res)}")

print(f"Original Creditcard_Data training set shape: {Counter(y_creditcard_train)}")
rus = RandomUnderSampler(random_state=42)
X_creditcard_train_res, y_creditcard_train_res = rus.fit_resample(X_creditcard_train, y_creditcard_train)
print(f"Resampled Creditcard_Data training set shape (RandomUnderSampler): {Counter(y_creditcard_train_res)}")

