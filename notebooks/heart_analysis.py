import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc, f1_score
)
import shap
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------------
local_path = os.path.join(DATA_DIR, 'heart.csv')

try:
    df = pd.read_csv(local_path)
    print(f"Loaded dataset: {df.shape}")
except FileNotFoundError:
    import requests, io
    url = "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"
    r = requests.get(url, verify=False)
    df = pd.read_csv(io.StringIO(r.text))
    df.to_csv(local_path, index=False)
    print(f"Downloaded dataset: {df.shape}")

print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget distribution:\n", df['target'].value_counts())

# ------------------------------------------------------------------
# EDA
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Heart Attack Prediction - EDA', fontsize=16, fontweight='bold')

axes[0, 0].hist(df[df['target'] == 0]['age'], alpha=0.7, label='No Disease', color='#2ecc71', bins=20, edgecolor='white')
axes[0, 0].hist(df[df['target'] == 1]['age'], alpha=0.7, label='Disease',    color='#e74c3c', bins=20, edgecolor='white')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].set_xlabel('Age')
axes[0, 0].legend()

gender_counts = df.groupby(['sex', 'target']).size().unstack()
gender_counts.plot(kind='bar', ax=axes[0, 1], color=['#2ecc71', '#e74c3c'], alpha=0.85, edgecolor='white')
axes[0, 1].set_title('Disease by Gender')
axes[0, 1].set_xticklabels(['Female', 'Male'], rotation=0)
axes[0, 1].legend(['No Disease', 'Disease'])

cp_counts = df.groupby(['cp', 'target']).size().unstack()
cp_counts.plot(kind='bar', ax=axes[1, 0], color=['#2ecc71', '#e74c3c'], alpha=0.85, edgecolor='white')
axes[1, 0].set_title('Chest Pain Type vs Disease')
axes[1, 0].set_xticklabels(['Typical Angina', 'Atypical', 'Non-anginal', 'Asymptomatic'], rotation=20, ha='right')
axes[1, 0].legend(['No Disease', 'Disease'])

axes[1, 1].scatter(df[df['target'] == 0]['age'], df[df['target'] == 0]['thalach'], alpha=0.6, label='No Disease', color='#2ecc71', s=40)
axes[1, 1].scatter(df[df['target'] == 1]['age'], df[df['target'] == 1]['thalach'], alpha=0.6, label='Disease',    color='#e74c3c', s=40)
axes[1, 1].set_title('Age vs Max Heart Rate')
axes[1, 1].set_xlabel('Age')
axes[1, 1].set_ylabel('Max Heart Rate')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'eda_overview.png'), dpi=150, bbox_inches='tight')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 9))
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
sns.heatmap(df.corr(), mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------
X = df.drop('target', axis=1)
y = df['target']
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ------------------------------------------------------------------
# Train models
# ------------------------------------------------------------------
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, random_state=42),
}

results = {}
print(f"\n{'Model':<25} {'Accuracy':>10} {'F1 Score':>10} {'CV Score':>10}")
print("-" * 60)

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    cv  = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='accuracy')

    results[name] = {'accuracy': acc, 'f1': f1, 'cv_mean': cv.mean(), 'model': model, 'y_prob': y_prob}
    print(f"{name:<25} {acc:>10.4f} {f1:>10.4f} {cv.mean():>10.4f}")

# ------------------------------------------------------------------
# Best model evaluation
# ------------------------------------------------------------------
best_name  = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_name]['model']
print(f"\nBest model: {best_name}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})", lw=2)
axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
axes[0].set_title('ROC Curves')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(fontsize=9)

y_pred_best = best_model.predict(X_test_sc)
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
axes[1].set_title(f'Confusion Matrix - {best_name}')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'roc_confusion.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n", classification_report(y_test, y_pred_best, target_names=['No Disease', 'Disease']))

# ------------------------------------------------------------------
# SHAP explainability
# ------------------------------------------------------------------
print("Computing SHAP values...")
explainer = shap.TreeExplainer(best_model)
shap_vals = explainer.shap_values(X_test_sc)

plt.figure(figsize=(10, 6))
if isinstance(shap_vals, list):
    shap.summary_plot(shap_vals[1], X_test_sc, feature_names=feature_names, show=False)
else:
    shap.summary_plot(shap_vals, X_test_sc, feature_names=feature_names, show=False)
plt.title('SHAP Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'shap_summary.png'), dpi=150, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------
# Save model
# ------------------------------------------------------------------
joblib.dump(best_model,    os.path.join(MODELS_DIR, 'heart_model.pkl'))
joblib.dump(scaler,        os.path.join(MODELS_DIR, 'scaler.pkl'))
joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.pkl'))
joblib.dump(explainer,     os.path.join(MODELS_DIR, 'shap_explainer.pkl'))

print(f"Model saved: {best_name}")
