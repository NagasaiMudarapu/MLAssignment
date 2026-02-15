# ML Classification Model Deployment

A Streamlit-based web application for deploying and evaluating multiple machine learning classification models on biodegradability prediction tasks.

**BITS Pilani - M.Tech (AIML/DSE) - Assignment 2 - 2025AA05504**

## Problem Statement

### Background

Environmental pollution caused by non-biodegradable chemical compounds poses a significant threat to ecosystems and human health. The ability to predict whether a chemical compound is biodegradable is crucial for:

- **Environmental Risk Assessment**: Identifying chemicals that may persist in the environment
- **Drug Discovery**: Ensuring pharmaceutical compounds break down safely after use
- **Industrial Chemical Design**: Creating eco-friendly products that minimize environmental impact
- **Regulatory Compliance**: Meeting environmental safety standards for new chemical substances

Traditional laboratory testing for biodegradability is time-consuming, expensive, and requires significant resources. There is a critical need for computational methods that can accurately predict biodegradability based on molecular structure.

### Solution Approach

This project implements six different classification algorithms and provides a comparative analysis to determine the optimal model for biodegradability prediction, with a focus on both accuracy and practical deployment considerations.

## Overview

This application provides an interactive interface to test and compare six different classification models trained on the QSAR Biodegradation dataset. Users can upload test datasets and evaluate model performance using multiple metrics.

## Features

- **6 Pre-trained Models**: Logistic Regression, Decision Tree, K Nearest Neighbor, Naive Bayes, Random Forest, and XGBoost
- **Interactive Model Selection**: Choose any model from the sidebar
- **File Upload Support**: Upload CSV test datasets for evaluation
- **Comprehensive Metrics**: Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient
- **Visual Analytics**: Confusion matrix heatmap visualization
- **Sample Data Download**: Built-in sample test dataset for quick testing

## Project Structure

```
MLAssignment/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── data/
│   ├── qsar-biodeg.csv        # Original dataset
│   ├── train_dataset.csv      # Training split (80%)
│   ├── test_dataset.csv       # Test split (20%)
│   └── split_data.py          # Dataset splitting utility
└── model/
    ├── dataset_loader.py      # Dataset loading utility
    ├── models.ipynb           # Model training notebook
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── k_nearest_neighbor.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── xgboost.pkl
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd MLAssignment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Using the App

1. **Select a Model**: Choose from six classification models in the sidebar
2. **Download Sample Data**: (Optional) Use the download button to get sample test data
3. **Upload Dataset**: Upload your CSV file with the same feature structure as the training data
4. **View Results**:
   - Data preview table
   - Performance metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
   - Confusion matrix visualization

### Dataset Requirements

Your CSV file should contain:
- **41 numerical features** (molecular descriptors)
- **1 target column** named "Class" (binary: ready/not ready biodegradable)

The expected format matches the QSAR Biodegradation dataset structure.
Reference - https://www.kaggle.com/datasets/muhammetvarl/qsarbiodegradation

## Dataset

Given 41 molecular descriptor features derived from chemical structures, classify compounds into two categories:
- **Ready Biodegradable (RB)**: Compounds that degrade easily in natural environments
- **Not Ready Biodegradable (NRB)**: Compounds that persist and may cause environmental harm

The challenge involves handling:
- High-dimensional feature space (41 features)
- Class imbalance considerations
- Model selection and comparison across diverse algorithms
- Deployment for practical use by domain experts

The project uses the **QSAR Biodegradation Dataset** which contains:
- **1055 instances** total
- **41 molecular descriptor features**
- **Binary classification target**: Biodegradable (RB) vs Not Biodegradable (NRB)

Data is split into:
- Training set: 844 instances (80%)
- Test set: 211 instances (20%)


### Regenerating Train/Test Split

To recreate the dataset split:

```bash
cd data
python split_data.py
```

## Models

The following pre-trained models are available:

| Model | Algorithm | File Size |
|-------|-----------|-----------|
| Logistic Regression | Linear classifier | 1.3 KB |
| Decision Tree | Tree-based classifier | 17 KB |
| K Nearest Neighbor | Instance-based learning | 285 KB |
| Naive Bayes | Probabilistic classifier | 2.2 KB |
| Random Forest | Ensemble method | 1.6 MB |
| XGBoost | Gradient boosting | 168 KB |

Models are stored as pickle files in the `model/` directory and can be loaded dynamically based on user selection.

## Model Performance

All models were evaluated on the test dataset (211 instances, 20% split). Below are the benchmark results:

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8720 | 0.9323 | 0.8748 | 0.8720 | 0.8729 | 0.7258 |
| Decision Tree | 0.8246 | 0.8191 | 0.8297 | 0.8246 | 0.8262 | 0.6267 |
| Random Forest| 0.8863 | 0.9402 | 0.8855 | 0.8863 | 0.8855 | 0.7494 |
| Naive Bayes | 0.7867 | 0.9353 | 0.8605 | 0.7867 | 0.7905 | 0.6393 |
| K Nearest Neighbor | 0.8294 | 0.8815 | 0.8373 | 0.8294 | 0.8315 | 0.6417 |
| XGBoost | 0.8720 | 0.9204 | 0.8725 | 0.8720 | 0.8722 | 0.7216 |

### Key Observations

| Model | Strengths | Weaknesses | Best Use Case |
|-------|-----------|------------|---------------|
| **Logistic Regression** | - Well-balanced metrics (87.20% accuracy)<br>- Strong AUC (93.23%)<br>- Lightweight (1.3 KB)<br>- Interpretable coefficients | - Linear decision boundary<br>- May miss complex patterns | Fast inference with interpretability needs |
| **Decision Tree** | - Simple interpretable structure<br>- Fast training and prediction<br>- No feature scaling required | - Lowest AUC (81.91%)<br>- Prone to overfitting<br>- Lower generalization (82.46% accuracy) | Quick exploratory analysis and feature importance |
| **Random Forest** | - **Best overall performer (88.63%)**<br>- Highest AUC (94.02%)<br>- Strongest MCC (0.7494)<br>- Stable ensemble predictions | - Large model size (1.6 MB)<br>- Slower inference<br>- Less interpretable | Production deployment requiring best accuracy |
| **Naive Bayes** | - Highest precision (86.05%)<br>- Second-best AUC (93.53%)<br>- Very lightweight (2.2 KB)<br>- Fast training and prediction | - Lowest accuracy (78.67%)<br>- Strong independence assumption<br>- Lower recall | Real-time applications prioritizing precision |
| **K Nearest Neighbor** | - No training phase<br>- Captures local patterns<br>- Good for non-linear boundaries | - Large storage (285 KB)<br>- Slow prediction time<br>- Moderate accuracy (82.94%) | Small datasets with complex decision boundaries |
| **XGBoost** | - Tied 2nd accuracy (87.20%)<br>- Well-balanced F1 (87.22%)<br>- Efficient size (168 KB)<br>- Handles missing values | - Requires hyperparameter tuning<br>- Less interpretable<br>- Moderate training time | Balanced performance with reasonable model size |



## Dependencies

Core libraries:
- `streamlit` - Web application framework
- `scikit-learn` - Machine learning algorithms
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `xgboost` - Gradient boosting framework

See `requirements.txt` for complete list.

## Development

### Training New Models

Models are trained in the Jupyter notebook:

```bash
jupyter notebook model/models.ipynb
```

The notebook handles:
- Data loading and preprocessing
- Model training with various algorithms
- Model serialization to pickle files
- Performance evaluation

### Custom Dataset Loader

The `model/dataset_loader.py` module provides a reusable function to load CSV datasets:

```python
from model.dataset_loader import load_dataset

X, y, df = load_dataset('path/to/data.csv', target_column='Class')
```

## License

This project is part of academic coursework for BITS Pilani M.Tech program.

## Author

**Student ID**: 2025AA05504
**Student Name**: MUDARAPU VEERA VENKATA NAGA SAI
**Program**: M.Tech (AIML/DSE)
**Institution**: BITS Pilani
