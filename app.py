import streamlit as st
import pandas as pd
import numpy as np
import pickle
from model.dataset_loader import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Classification App", layout="wide")
st.title("Machine Learning Model Deployment")
st.write("""
**BITS Pilani - M.Tech (AIML/DSE) - Assignment 2 - 2025AA05504** This app implements multiple classification models on a chosen dataset.
""")

st.sidebar.header("Model Configuration")
model_options = [
    "Logistic Regression",
    "Decision Tree",
    "K-Nearest Neighbor",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
]
selected_model_name = st.sidebar.selectbox("Select a Model", model_options)

uploaded_file = st.file_uploader("Upload your Test Dataset (CSV)", type=["csv"])

try:
    sample_csv = pd.read_csv("./data/test_dataset.csv")
    csv_data = sample_csv.to_csv(index=False).encode('utf-8')

    # 2. Create the Download Button
    st.sidebar.download_button(
        label="📥 Download Sample Test Data",
        data=csv_data,
        file_name="sample_test_data.csv",
        mime="text/csv",
        help="Click to download a sample CSV file to test this app."
    )

except FileNotFoundError:
    st.warning("⚠️ sample_test_data.csv not found. Please upload your own.")

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())

    st.success("File upload successfully!")

    filename = f"model/{selected_model_name.replace(' ', '_').lower()}.pkl"
    with open(filename, 'rb') as file:
        loaded_model = pickle.load(file)

    st.success(f"Successfully loaded: **{selected_model_name}**")

    X, Y, df = load_dataset('./data/test_dataset.csv')

    #TODO: Set the scalers afterwards
#     scaler = StandardScaler()
#     X_scaled = scaler.transform(X)

    y_pred = loaded_model.predict(X)

    y_prob = loaded_model.predict_proba(X)[:, 1]

    acc = accuracy_score(Y, y_pred)
    auc = roc_auc_score(Y, y_prob) if len(np.unique(Y)) == 2 else 0 # Handle binary only for AUC
    prec = precision_score(Y, y_pred, average='weighted')
    rec = recall_score(Y, y_pred, average='weighted')
    f1 = f1_score(Y, y_pred, average='weighted')
    mcc = matthews_corrcoef(Y, y_pred)

    st.write("### Evaluation Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.4f}")
    col1.metric("AUC Score", f"{auc:.4f}")
    col2.metric("Precision", f"{prec:.4f}")
    col2.metric("Recall", f"{rec:.4f}")
    col3.metric("F1 Score", f"{f1:.4f}")
    col3.metric("MCC Score", f"{mcc:.4f}")

    # --- 7. VISUALIZATION (Confusion Matrix) [cite: 94] ---
    st.write("### Confusion Matrix")
    cm = confusion_matrix(Y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    st.success("Model Predicted")

else:
    st.info("Please upload a CSV dataset to proceed.")