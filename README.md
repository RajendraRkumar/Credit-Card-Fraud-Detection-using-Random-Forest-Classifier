# Credit Card Fraud Detection Project

## Overview

This project focuses on building a robust machine learning model to detect fraudulent credit card transactions. Given a dataset of transaction records, the goal is to accurately identify and flag suspicious activities, which is a critical task for financial institutions. The challenge lies in the highly imbalanced nature of the data—fraudulent transactions are typically very rare compared to legitimate ones.

This repository walks through the entire data science lifecycle, including:
*   Exploratory Data Analysis (EDA)
*   Extensive Feature Engineering
*   Data Preprocessing
*   Model Training and Evaluation
*   Interpretation of Model Results

We use a **Random Forest Classifier**, which proves to be highly effective at this task, achieving excellent performance in distinguishing between fraudulent and non-fraudulent transactions.

## Dataset

The model was trained on a dataset containing over 1.2 million transaction records.

*   **Shape**: 1,296,675 rows and 23 columns.
*   **Features**: The data includes transaction details (`amt`, `merchant`, `category`), customer information (`first`, `last`, `gender`, `job`, `dob`), and location data (`lat`, `long`, `merch_lat`, `merch_long`).
*   **Target Variable**: The `is_fraud` column is our target, where `1` indicates a fraudulent transaction and `0` indicates a legitimate one. The dataset is highly imbalanced, with fraud cases representing only about **0.58%** of all transactions.

## Project Workflow

The project was structured into several key stages, from understanding the data to building a predictive model.

### 1. Exploratory Data Analysis (EDA)

The first step was to dive deep into the data to uncover patterns and anomalies. We found that:
*   The distribution of transaction amounts (`amt`) is heavily skewed, with most transactions being small amounts. A log scale was used for better visualization.
*   Certain hours of the day, particularly early morning hours, showed a higher proportion of fraudulent transactions.
*   The `is_fraud` target variable confirmed a severe class imbalance, which heavily influenced our modeling and evaluation strategy.

### 2. Feature Engineering

To enhance the model's predictive power, we engineered several new features from the existing data:

*   **Time-Based Features**: `transaction_hour` and `day_of_week` were extracted from the transaction timestamp to capture temporal patterns.
*   **Customer Age**: The `age` of the cardholder at the time of the transaction was calculated using their date of birth.
*   **Transaction Distance**: A crucial feature, `distance_km`, was created to calculate the geographical distance (in kilometers) between the customer's home location (`lat`, `long`) and the merchant's location (`merch_lat`, `merch_long`). A large distance could be an indicator of fraud.

### 3. Data Preprocessing and Preparation

Before feeding the data to the model, we performed these preprocessing steps:
*   **Dropped unnecessary columns**: Identifiers like `cc_num`, `first`, `last`, `trans_num`, and raw location/date columns were removed.
*   **Handled Data Types**: Date columns were converted to datetime objects for feature engineering.
*   **Scaled Numerical Features**: All numerical features were scaled using `StandardScaler` to normalize their ranges.
*   **Encoded Categorical Features**: Categorical columns like `category` and `merchant` were transformed into a numerical format using `OneHotEncoder`.

All these steps were encapsulated in a `ColumnTransformer` to create a clean and repeatable preprocessing pipeline.

### 4. Model Building

A **Random Forest Classifier** was selected for this task due to its robustness, ability to handle complex interactions, and built-in mechanisms for providing feature importance.

The entire workflow, from preprocessing to modeling, was wrapped in a Scikit-Learn `Pipeline`. This ensures that the same steps are consistently applied to both training and testing data.

To address the class imbalance, we configured the model with `class_weight='balanced'`, which adjusts the weights of samples inversely proportional to class frequencies.

### 5. Model Evaluation

The model's performance was evaluated on a held-out test set. Since accuracy is a misleading metric for imbalanced datasets, we focused on precision, recall, and the ROC AUC score.

**The final results were very strong:**
*   **ROC AUC Score**: **0.986** — This indicates the model has an excellent capability to distinguish between the two classes.
*   **Fraud Class Performance**:
    *   **Precision**: 0.99 (When the model predicts fraud, it's correct 99% of the time).
    *   **Recall**: 0.67 (The model successfully identifies 67% of all actual fraud cases).
    *   **F1-Score**: 0.80 (A healthy balance between precision and recall).

These results show a model that is highly reliable when it flags a transaction but still misses about one-third of fraudulent transactions—a common trade-off in fraud detection systems.

### 6. Feature Importance

To understand *why* the model makes its decisions, we analyzed the feature importances. The visualization revealed that the most influential factors for detecting fraud were:
*   **Transaction Category**: Certain categories like `misc_net` and `shopping_net` were strong predictors.
*   **Transaction Distance (`distance_km`)**: The distance between the customer and the merchant was a top indicator.
*   **Transaction Amount (`amt`)**: The value of the transaction remains a key feature.
*   **Transaction Hour (`transaction_hour`)**: The time of day proved to be highly relevant.

## How to Run This Project

1.  **Prerequisites**: Ensure you have Python 3 installed.
2.  **Install Dependencies**: The project requires the following libraries. You can install them using pip:
    ```
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
3.  **Load Data**: Place your transaction data CSV file in the same directory as your script or notebook.
4.  **Execute the Code**: Run the Python script or Jupyter Notebook cells sequentially to perform the EDA, training, and evaluation.

## Conclusion and Future Work

This project successfully developed a high-performing machine learning model for credit card fraud detection. The Random Forest classifier, combined with thoughtful feature engineering, provides a great balance of precision and recall.

For future improvements, we could explore:
*   **Threshold Tuning**: Adjusting the decision threshold (default is 0.5) to increase recall for the fraud class, which may be desirable depending on business needs.
*   **Advanced Imbalance Handling**: Implementing sampling techniques like SMOTE (Synthetic Minority Over-sampling Technique) on the training data.
*   **Experiment with Other Models**: Testing gradient boosting models (like XGBoost or LightGBM) or building a deep learning model using TensorFlow/Keras could yield further performance gains.
