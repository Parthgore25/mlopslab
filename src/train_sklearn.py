#!/usr/bin/env python3
"""Scikit-learn Model Training Script"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import json

def load_data(data_path):
    """Load and preprocess data"""
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    """Train Random Forest model"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions))
    return accuracy

def save_model(model, metrics, output_dir):
    """Save model and metrics"""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'sklearn_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data('data/dataset.csv')
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    save_model(model, {'accuracy': accuracy}, 'models/')
    print("Training completed successfully!")
