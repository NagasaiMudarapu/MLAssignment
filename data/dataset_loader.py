import pandas as pd
import os

# Default configuration
DEFAULT_FILENAME = 'qsar-biodeg.csv'
DEFAULT_TARGET_COL = 'Class'

def load_dataset(file_path=None, target_column=None):
    """
    Loads a dataset from a local CSV file.

    Args:
        file_path (str): Path to the CSV file. If None, uses default.
        target_column (str): Name of the target variable column.
                             If None, assumes the LAST column is the target.

    Returns:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        df (pd.DataFrame): Full dataframe
    """
    # 1. Determine File Path
    if file_path is None:
        file_path = DEFAULT_FILENAME

    # 2. Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Error: The file '{file_path}' was not found in the directory.")

    # 3. Load CSV
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"❌ Error reading CSV file: {e}")

    # 4. Identify Target Column
    if target_column is None:
        # Default behavior: Assume the LAST column is the target
        target_column = df.columns[-1]
        print(f"ℹ️ Note: No target column specified. Using last column: '{target_column}'")

    if target_column not in df.columns:
        raise ValueError(f"❌ Error: Target column '{target_column}' not found in dataset.")

    # 5. Split Features (X) and Target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y, df

# --- Test Block (Runs only when you execute this file directly) ---
if __name__ == "__main__":
    if not os.path.exists(DEFAULT_FILENAME):
        print("⚠️ Test CSV not found. Creating a dummy file for demonstration...")

    try:
        # Test loading
        X, y, df = load_dataset()
        print("\n✅ Dataset Loaded Successfully from CSV!")
        print(f"File Used: {DEFAULT_FILENAME}")
        print(f"Features: {X.shape[1]}")
        print(f"Instances: {X.shape[0]}")
        print(f"Target Column: {y.name}")

    except Exception as e:
        print(e)