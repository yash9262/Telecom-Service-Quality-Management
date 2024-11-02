# Telecom-Service-Quality-Management

## Churn Prediction 

This project aims to predict customer churn based on telecom call data using various machine learning algorithms, including Logistic Regression, Support Vector Classifier (SVC), and K-Nearest Neighbors (KNN). The project includes data preprocessing, exploratory data analysis (EDA), model training, and evaluation.

## Features

- Load and preprocess call data from a CSV file.
- Exploratory data analysis with visualizations (histograms, boxplots, heatmaps).
- Implement various machine learning models for churn prediction:
  - Logistic Regression
  - Support Vector Classifier (SVC)
  - K-Nearest Neighbors (KNN)
- Evaluate model performance using accuracy and classification reports.
- Predict churn for new customer data.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- Required Python packages:
  - `pandas`
  - `numpy`
  - `seaborn`
  - `matplotlib`
  - `scikit-learn`

### Data Structure

The dataset should contain the following relevant columns:

- **Rating**: Customer rating (numerical).
- **Call Drop Category**: Categories of call drop, which can include:
  - `Satisfactory`
  - `Poor Voice Quality`
  - `Call Dropped`
- **total day calls**: Total number of calls made during the day (numerical).
- **total eve calls**: Total number of calls made during the evening (numerical).
- **total night calls**: Total number of calls made during the night (numerical).
- **churn**: Target variable indicating whether a customer has churned (represented as `TRUE` or `FALSE`).

### Predicting New Data

You can predict churn for new customer data by providing values in the following format:

```python
new_data = {
    'Rating': [1, 2, 3, 4, 5],
    'Call Drop Category': [0, 1, 2, 2, 1],  # Numeric representation of call drop categories
    'total day calls': [65, 50, 80, 99, 113],
    'total eve calls': [80, 24, 39, 45, 73],
    'total night calls': [20, 45, 99, 100, 10]
}
### Model Evaluation

The models are evaluated based on their performance metrics, including accuracy, precision, recall, and F1-score. During evaluation, the following points will be printed:

- **Training Accuracy**: The accuracy of the model on the training dataset, indicating how well it learned from the training data.
  
- **Testing Accuracy**: The accuracy of the model on the testing dataset, reflecting its ability to generalize to unseen data.

- **Classification Report**: A detailed report that includes:
  - **Precision**: The ratio of true positive predictions to the total predicted positives, indicating how many selected items are relevant.
  - **Recall**: The ratio of true positive predictions to the total actual positives, showing how many relevant items are selected.
  - **F1-Score**: The harmonic mean of precision and recall, providing a single metric that balances both concerns.

These metrics help assess the performance of each model and identify any potential issues such as overfitting or underfitting.

