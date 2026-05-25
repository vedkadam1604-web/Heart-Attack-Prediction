# Heart Attack Prediction

A machine learning project to predict the risk of heart disease based on clinical data. Built this as part of my data science learning journey.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)

## What This Does

Takes 13 clinical measurements (age, cholesterol, blood pressure, etc.) and predicts whether a patient is at risk of heart disease. Also shows *why* the model made that prediction using SHAP values.

## Dataset

Used the Cleveland Heart Disease dataset from UCI — 303 patient records with 13 features. No missing values, which made preprocessing pretty straightforward.

| Feature | What it means |
|---------|---------------|
| age | Patient age in years |
| sex | 0 = Female, 1 = Male |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mmHg) |
| chol | Cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG result |
| thalach | Max heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression from exercise |
| slope | Slope of peak ST segment |
| ca | Major vessels colored by fluoroscopy |
| thal | Thalassemia type |

## Models Trained

Tried three different models and picked the best one:

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | 80% | 83% |
| Random Forest | **82%** | **85%** |
| Gradient Boosting | 80% | 83% |

Random Forest came out on top so that's what the app uses.

## SHAP Explainability

One thing I wanted to add was explainability — not just "high risk" but *why*. SHAP values break down exactly which features pushed the prediction up or down for each patient. Really useful for understanding what the model is actually doing.

## Project Structure

```
heart-attack-prediction/
├── data/
│   └── heart.csv
├── notebooks/
│   └── heart_analysis.py      # EDA + model training
├── models/
│   ├── heart_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── app/
│   └── app.py                 # Streamlit web app
├── requirements.txt
└── README.md
```

## Running Locally

```bash
git clone https://github.com/YOUR_USERNAME/heart-attack-prediction.git
cd heart-attack-prediction
pip install -r requirements.txt
cd app
streamlit run app.py
```

The trained model is already included so it loads instantly.

## Live Demo

[Click here to try it](https://YOUR_APP_URL.streamlit.app) ← update after deploying

## Disclaimer

This is a personal learning project and not meant for actual medical use. Always consult a doctor.
