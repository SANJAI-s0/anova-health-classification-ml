# Anova Insurance – Health Classification ML Model

## Project Overview
This repository contains a comprehensive Machine Learning pipeline designed for **Anova Insurance** to automate the classification of insurance applicants into **'Healthy' (0)** and **'Unhealthy' (1)** categories. The model leverages medical history, lifestyle factors, and physiological data to provide accurate risk assessments, which are critical for determining fair premium pricing.

## Problem Statement
The primary objective is to build a predictive model that can reliably distinguish high-risk health profiles from low-risk ones. The project accounts for real-world data challenges, including:
- **Missing Data**: Handling incomplete records, especially from older demographics, using advanced imputation techniques.
- **Data Quality Issues**: Correcting data entry errors such as negative ages.
- **Predictive Accuracy**: Utilizing a rich feature set of 20+ variables and ensemble methods to maximize model precision and AUC-ROC.
- **Business Impact**: Providing a breakdown of risk drivers (Blood Pressure, BMI, Cholesterol, etc.) to assist in underwriting decisions.

## Features & Data Description
The model uses a diverse set of features including:
- **Physiological Metrics**: Age, BMI, Blood Pressure, Cholesterol, Glucose Level, Heart Rate.
- **Lifestyle Indicators**: Sleep Hours, Exercise Hours, Water Intake, Stress Level, Smoking, Alcohol, Diet.
- **Categorical Factors**: Mental Health status, Physical Activity level, Medical History, Allergies.
- **Engineered Features**: High BP/Cholesterol/Glucose flags, Overweight flag, Composite Risk Score, Lifestyle Score, and cross-feature interactions (e.g., BP x BMI).

## Project Structure
```text
ML_Model/
├── Dataset/                   # Preprocessed healthcare datasets
├── Project/                   # Original problem statement and documentation
├── plots/                     # Visualizations of EDA and model performance
├── health_classification.py   # Core ML pipeline script
├── best_model.pkl             # Serialized best-performing model
├── requirements.txt           # Python dependencies
├── .gitignore                 # Files to ignore in Git
└── LICENSE                    # MIT License
```

## Setup and Usage
### Prerequisites
- Python 3.8+
- Recommended to use a virtual environment (`venv`)

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline
Run the main script to perform data cleaning, EDA, feature engineering, and model training:
```bash
python health_classification.py
```
*Note: The script automatically generates performance plots in the `plots/` directory and saves the best model as `best_model.pkl`.*

## Model Performance
The pipeline evaluates several state-of-the-art classifiers:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost & LightGBM
- Stacking Ensemble (Final Model)

Evaluation metrics include Accuracy, ROC-AUC, F1-Score, and Precision-Recall curves, ensuring robust performance across both healthy and unhealthy classes.

## License
Distributed under the MIT License. See `LICENSE` for more information.
