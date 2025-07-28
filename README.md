# Predictive Maintenance using Big Data Analytics on UR5 Robotic Arm

This repository contains the complete implementation of a predictive maintenance project using machine learning and big data analytics. The study focuses on the NIST PHM 2021 dataset from a UR5 robotic work cell, aiming to detect operational anomalies through sensor data analysis.

## Project Overview

- **Objective**: Leverage big data analytics and machine learning to identify potential failures in an industrial robotic system.
- **Dataset**: UR5 sensor data (NIST PHM 2021) containing joint positions, velocities, and tool coordinates.
- **Methods**:
  - Data cleaning and transformation
  - Feature engineering (rolling statistics, deltas)
  - Synthetic anomaly labeling
  - Classification using Logistic Regression, Random Forest, Gradient Boosting, and K-Nearest Neighbors

## Models Used
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- K-Nearest Neighbors

All models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Cross-Validation

## Key Results

| Model               | Accuracy | Precision | Recall | F1-Score | AUC  |
|--------------------|----------|-----------|--------|----------|------|
| Gradient Boosting  | 1.0000   | 1.0000    | 1.0000 | 1.0000   | 1.00 |
| Random Forest      | 0.9985   | 0.9919    | 0.9866 | 0.9892   | 1.00 |
| K-Nearest Neighbors| 0.9969   | 0.9784    | 0.9758 | 0.9771   | 1.00 |
| Logistic Regression| 0.9932   | 0.9443    | 0.9570 | 0.9506   | 1.00 |

## Files

- `main.py`: The complete Python script containing data processing, modeling, and evaluation.
- `UR5Data.csv`: Input dataset (not included here due to size).
- `README.md`: Project overview and guide.
- `requirements.txt`: Python package dependencies.

## 📦 Requirements

You can install dependencies using:

```bash
pip install -r requirements.txt

