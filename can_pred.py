import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Streamlit UI Title
st.title("🩺 Cervical Cancer Prediction App")

# Upload CSV file
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 **Dataset Preview:**", df.head())
    
    # Display dataset info
    st.write("🔍 **Dataset Info:**")
    st.write(df.info())

    # Handle missing values (fill with mean for numerical, mode for categorical)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    # Encode categorical variables using LabelEncoder
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Define features and target variable
    X = df.iloc[:, :-1]  # All columns except the last one
    y = df.iloc[:, -1]   # The last column as target

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize numerical features to have zero mean and unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize models
    models = {
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42)
    }

    # Train and evaluate models
    st.subheader("📊 Model Training & Evaluation")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.write(f"### {name}")
        st.write(f"✔ **Accuracy:** {accuracy:.4f}")
        st.text(classification_report(y_test, y_pred))

    # Get user input for prediction
    st.subheader("🔍 Predict Cervical Cancer for a New Patient")

    sample_values = []
    for i, col in enumerate(X.columns[:5]):  # Adjust based on important features
        value = st.number_input(f"Enter value for {col}:", min_value=0.0, step=0.1)
        sample_values.append(value)

    if st.button("🔮 Predict"):
        sample_data = np.zeros((1, X.shape[1]))
        sample_data[0, :len(sample_values)] = sample_values

        sample_df = pd.DataFrame(sample_data, columns=X.columns)
        sample_scaled = scaler.transform(sample_df)

        st.subheader("🧪 Prediction Results")
        for name, model in models.items():
            sample_prediction = model.predict(sample_scaled)
            st.write(f"**{name} Prediction:** {'Cancer Detected' if sample_prediction[0] == 1 else 'No Cancer Detected'}")

    # High-risk patient prediction
    st.subheader("⚠️ High-Risk Patient Analysis")
    if st.button("Analyze High-Risk Patient"):
        high_risk_sample = df[y == 1].iloc[0, :-1].values.reshape(1, -1)
        high_risk_df = pd.DataFrame(high_risk_sample, columns=X.columns)
        high_risk_scaled = scaler.transform(high_risk_df)

        for name, model in models.items():
            high_risk_prediction = model.predict(high_risk_scaled)
            st.write(f"**{name} Prediction:** {'Cancer Detected' if high_risk_prediction[0] == 1 else 'No Cancer Detected'}")
