#!/usr/bin/env python3
"""
Multi-Class Deep Learning & Machine Learning Model Trainer
==========================================================
Trains and evaluates multi-class models (MLP & Random Forest) to predict
which of the 11 cancer types an individual is susceptible to.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

def train_and_evaluate_multiclass(csv_path="multiclass_cancer_master_dataset.csv"):
    if not os.path.exists(csv_path):
        print(f"Dataset {csv_path} not found.")
        return
        
    print("=" * 70)
    print("TRAINING MULTI-CLASS CANCER PREDICTION DEEP LEARNING MODEL")
    print(f"Dataset File: {csv_path}")
    print("=" * 70)
    
    df = pd.read_csv(csv_path)
    print(f"Loaded Dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    one_hot_cols = [c for c in df.columns if c.startswith("one_hot_")]
    metric_cols = [c for c in ["dp", "af", "ro", "ao", "qual", "gt_code", "gc_content_21bp", "is_transition"] if c in df.columns]
    
    feature_cols = one_hot_cols + metric_cols
    X = df[feature_cols].values
    y = df["cancer_class_id"].values
    
    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)
    
    print(f"Train Set: {len(X_train):,} samples | Test Set: {len(X_test):,} samples")
    print(f"Target Classes in Dataset: {len(np.unique(y))} Unique Cancers")
    
    # --- MODEL 1: Multi-Class MLP (Multi-Layer Perceptron) ---
    print("\n--- Training Multi-Class Neural Network (MLP)... ---")
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation="relu", max_iter=200, random_state=42)
    mlp.fit(X_train, y_train)
    
    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "=" * 70)
    print("MULTI-CLASS NEURAL NETWORK EVALUATION REPORT")
    print("=" * 70)
    print(f"Top-1 Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report across 11 Cancer Types:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("=" * 70)

if __name__ == "__main__":
    c_path = "multiclass_cancer_master_dataset.csv"
    if len(sys.argv) > 1:
        c_path = sys.argv[1]
    train_and_evaluate_multiclass(c_path)
