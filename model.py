# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import numpy as np


DATASET_FILE = "cleaned_dataset.csv"

# Column names
columns = ['temperature','humidity','gasAnalog','flameDetected', 'fireStatus']

# Load dataset
data = pd.read_csv(DATASET_FILE)

# Split features and target
X = data.drop('fireStatus', axis=1)
y = data['fireStatus']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (important for LR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)



def save_logistic_model_as_h(model, scaler, filename="model.h"):
    coefs = model.coef_[0]
    intercept = model.intercept_[0]

    # Optional: scale parameters if you want to include preprocessing
    mean = scaler.mean_
    scale = scaler.scale_

    with open(filename, "w") as f:
        f.write("// Logistic Regression Model Parameters\n\n")
        f.write(f"const float lr_coeffs[{len(coefs)}] = " + "{ " + ", ".join(map(str, coefs)) + " };\n")
        f.write(f"const float lr_intercept = {intercept};\n\n")
        f.write("// StandardScaler Parameters\n")
        f.write(f"const float scaler_mean[{len(mean)}] = " + "{ " + ", ".join(map(str, mean)) + " };\n")
        f.write(f"const float scaler_scale[{len(scale)}] = " + "{ " + ", ".join(map(str, scale)) + " };\n")

save_logistic_model_as_h(model, scaler)
