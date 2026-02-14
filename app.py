import streamlit as st
import pandas as pd

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
    st.success(f"Successfully loaded: **{selected_model_name}**")

else:
    st.info("Please upload a CSV dataset to proceed.")