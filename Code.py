
"""
Created on Fri Jul 25 17:59:38 2025

@author: RANSFORD
"""

# FINAL PROJECT - RANSFORD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# -----------------------------
# 1. Load and Preprocess Data
# -----------------------------
df = pd.read_csv("C:/Users/RANSFORD/Desktop/DATA SCIENCE COV/Final Project/UR5Data.csv")
df.dropna(subset=['ToolX', 'ToolY'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Normalizing joint angles and velocities
angle_cols = [col for col in df.columns if 'qactual' in col or 'qdactual' in col]
for col in angle_cols:
    df[col] = ((df[col] + np.pi) % (2 * np.pi)) - np.pi

# -----------------------------
# 2. Feature Engineering
# -----------------------------
for col in ['j1_qactual', 'j2_qactual', 'j3_qactual']:
    df[f'{col}_roll_mean'] = df[col].rolling(window=5).mean()
    df[f'{col}_roll_std'] = df[col].rolling(window=5).std()

df['ToolX_diff'] = df['ToolX'].diff()
df['ToolY_diff'] = df['ToolY'].diff()
df.bfill(inplace=True)
df.ffill(inplace=True)

# -----------------------------
# 3. Label Generation
# -----------------------------
threshold = df['ToolX_diff'].std() * 2
df['label'] = (np.abs(df['ToolX_diff']) > threshold).astype(int)

# -----------------------------
# 4. Drop Highly Correlated Features
filtered_df = df.drop(columns=['label', 'ToolZ'], errors='ignore')
corr_matrix = filtered_df.corr()

threshold_corr = 0.9
high_corr_features = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > threshold_corr:
            col2 = corr_matrix.columns[j]
            high_corr_features.add(col2)

cleaned_features = [col for col in filtered_df.columns if col not in high_corr_features]

# -----------------------------
# 5. Final Feature Set
X = df[cleaned_features]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
X_test_scaled = scaler.transform(imputer.transform(X_test))

# -----------------------------
# 6. Model Training
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}
trained_models = {name: model.fit(X_train_scaled, y_train) for name, model in models.items()}

# -----------------------------
# 7. Classification Reports
for name, model in trained_models.items():
    preds = model.predict(X_test_scaled)
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, preds))

# -----------------------------
# 8. Confusion Matrix Heatmaps
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (name, model) in zip(axes.flat, trained_models.items()):
    preds = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.show()

# -----------------------------
# 9. Evaluation Metrics Heatmap
def get_metrics(y_true, y_pred, name):
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred)
    }

metrics_summary = [
    get_metrics(y_test, model.predict(X_test_scaled), name)
    for name, model in trained_models.items()
]
df_metrics = pd.DataFrame(metrics_summary).set_index("Model")

plt.figure(figsize=(8, 5))
sns.heatmap(df_metrics, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title("Model Evaluation Metrics")
plt.tight_layout()
plt.show()

# -----------------------------
# 10. ROC Curve
plt.figure(figsize=(10, 8))
for name, model in trained_models.items():
    probs = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 11. Cross-Validation
print("\n5-Fold Cross-Validation Scores:")
for name, model in trained_models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"{name}: {np.round(scores, 5)}")

# -----------------------------
# 12. Side-by-side Confusion Matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (name, model) in zip(axes.flat, trained_models.items()):
    preds = model.predict(X_test_scaled)
    ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax, cmap='viridis', values_format='d')
    ax.set_title(f"{name} Confusion Matrix")
plt.tight_layout()
plt.show()

# -----------------------------
# 13. Performance Comparison Bar Plot
df_metrics.plot(kind='bar', figsize=(10, 6))
plt.title("Model Performance Comparison (Test Set)")
plt.ylabel("Score")
plt.ylim(0.9, 1.01)
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# -----------------------------
# 14. Feature Importance for Top Models
top_models = ["Random Forest", "Gradient Boosting"]
X_cols = X.columns

for name in top_models:
    model = trained_models[name]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': X_cols, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feat_df, palette='crest')
        plt.title(f"Top 15 Important Features ({name})")
        plt.tight_layout()
        plt.show()

# -----------------------------
# 15. Precision-Recall Curve
plt.figure(figsize=(10, 8))
for name, model in trained_models.items():
    probs = model.predict_proba(X_test_scaled)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    plt.plot(recall, precision, label=f'{name} (AP = {ap:.2f})')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 16. Correlation Heatmap
corr_features = df[cleaned_features]
plt.figure(figsize=(16, 12))
sns.heatmap(corr_features.corr(), annot=False, cmap='coolwarm', center=0,
            linewidths=0.5, cbar_kws={'label': 'Correlation'})
plt.title('Feature Correlation Heatmap (Cleaned)')
plt.tight_layout()
plt.show()
