import pandas as pd
from sklearn.model_selection import train_test_split
import os

INPUT_FILE = 'qsar-biodeg.csv' 
TEST_SIZE = 0.2  # 20% for testing, 80% for training
RANDOM_STATE = 42

def split_csv_dataset(filename):
    # 1. Check if file exists
    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' was not found.")
        return

    # 2. Load the dataset
    print(f"Loading '{filename}'...")
    try:
        df = pd.read_csv(filename)
        print(f"Original Dataset Shape: {df.shape}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 3. Perform the Split
    # distinct from X/y split, this splits the entire dataframe (rows)
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # 4. Save to new CSV files
    train_filename = 'train_dataset.csv'
    test_filename = 'test_dataset.csv'

    train_df.to_csv(train_filename, index=False)
    test_df.to_csv(test_filename, index=False)

    # 5. Output Summary
    print("\n--- Split Complete ---")
    print(f"Training Data: {train_df.shape} -> Saved as '{train_filename}'")
    print(f"Testing Data:  {test_df.shape} -> Saved as '{test_filename}'")
    print("\n✅ You can now use 'test_dataset.csv' for the upload feature in your Streamlit app.")

if __name__ == "__main__":
    split_csv_dataset(INPUT_FILE)
