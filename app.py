import streamlit as st
import pandas as pd
import pickle

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
    scaler = StandardScaler()

    X_scaled = scaler.transform(X)

    y_pred = loaded_model.predict(X_scaled)

    y_prob = loaded_model.predict_proba(X_scaled)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(Y, y_pred),
        "AUC Score": roc_auc_score(Y, y_prob),
        "Precision": precision_score(Y, y_pred),
        "Recall": recall_score(Y, y_pred),
        "F1 Score": f1_score(Y, y_pred),
        "MCC (Matthews Correlation)": matthews_corrcoef(Y, y_pred)
    }

    for metric, value in metrics.items():
        print(f"{metric:<25}: {value:.4f}")

    cm = confusion_matrix(Y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

else:
    st.info("Please upload a CSV dataset to proceed.")