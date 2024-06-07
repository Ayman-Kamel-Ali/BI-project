import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

# Assuming X_classification is your feature matrix with the 'Cholesterol' column
# Replace this with your actual data
data = {
    'Cholesterol': [289.0, 180.0, 283.0, 214.0, 150.0, 339.0, 237.0, 208.0, 207.0, 284.0, 211.0, 164.0, 204.0, 234.0, 211.0, 273.0, 196.0, 201.0, 248.0, 267.0],
}

X_classification = pd.DataFrame(data)

# Apply Robust Scaling
robust_scaler = RobustScaler()
X_classification['Cholesterol_robust'] = robust_scaler.fit_transform(X_classification[['Cholesterol']])

# Apply Standard Scaling
standard_scaler = StandardScaler()
X_classification['Cholesterol_standard'] = standard_scaler.fit_transform(X_classification[['Cholesterol']])

# Display the results
print("Original Data:")
print(X_classification[['Cholesterol']])

print("\nRobust Scaled Data:")
print(X_classification[['Cholesterol_robust']])

print("\nStandard Scaled Data:")
print(X_classification[['Cholesterol_standard']])
