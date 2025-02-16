import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("cercan.csv")

# Display dataset info
print("Dataset Preview:")
print(df.head())
print("\nDataset Info:")
df.info()

# Handle missing values (fill categorical with mode, numerical with mean)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])  # Assign back to df[col]
    else:
        df[col] = df[col].fillna(df[col].mean())  # Assign back to df[col]

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target (assuming last column is target)
X = df.iloc[:, :-1]  # All columns except last
y = df.iloc[:, -1]   # Last column as target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Ensure sample_data has the correct number of features
sample_data = np.zeros((1, X.shape[1]))  # Create a placeholder with correct shape
sample_values = [20, 1, 1, 1, 1, 1, 2, 1, 2]  # Adjust values accordingly
sample_data[0, :len(sample_values)] = sample_values  # Assign known values

# Convert sample_data into a DataFrame with correct column names
sample_df = pd.DataFrame(sample_data, columns=X.columns)

# Standardize sample data
sample_scaled = scaler.transform(sample_df)

# Predict cervical cancer presence
prediction = model.predict(sample_scaled)
print("\nPrediction for Sample Data:", prediction)

# Extract a high-risk sample (patient with cervical cancer)
high_risk_sample = df[y == 1].iloc[0, :-1].values.reshape(1, -1)  # First diagnosed case

# Convert high_risk_sample to a DataFrame
high_risk_df = pd.DataFrame(high_risk_sample, columns=X.columns)

# Standardize high-risk sample
high_risk_scaled = scaler.transform(high_risk_df)

# Predict cervical cancer presence for high-risk sample
high_risk_prediction = model.predict(high_risk_scaled)
print("\nPrediction for High-Risk Sample:", high_risk_prediction)
