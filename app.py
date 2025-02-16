import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Title of the app
st.title("Cervical Cancer Prediction App")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    # Encode categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Define features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

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
    st.write(f"### Model Accuracy: {accuracy:.2f}")
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))

    # User input for prediction
    st.sidebar.header("Enter Patient Data for Prediction")

    input_data = []
    for i, col in enumerate(X.columns):
        value = st.sidebar.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        input_data.append(value)

    # Predict on new user input
    if st.sidebar.button("Predict Cervical Cancer"):
        input_df = pd.DataFrame([input_data], columns=X.columns)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        st.sidebar.write("### Prediction:", "Cancer Detected" if prediction[0] == 1 else "No Cancer Detected")

