#!/usr/bin/env python3
"""
First Model Build: Dual-Branch 1D-CNN + Tabular Deep Learning Pipeline
=====================================================================
Target: 11-Class Pure Germline Cancer Risk & Susceptibility Prediction
Features: 176 Input Features (168 DNA One-Hot Channels + 8 Quality/Bio Metrics)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, top_k_accuracy_score

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "multiclass_cancer_dataset", "master_dataset", "germline_cancer_master_dataset.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "best_germline_model.pth")
SCALER_SAVE_PATH = os.path.join(BASE_DIR, "germline_scaler.pkl")
CONFUSION_MATRIX_PATH = os.path.join(BASE_DIR, "germline_confusion_matrix.png")
TRAINING_CURVES_PATH = os.path.join(BASE_DIR, "germline_training_curves.png")
METRICS_JSON_PATH = os.path.join(BASE_DIR, "germline_model_metrics.json")


# ---------------------------------------------------------
# 1. Dual-Branch Deep Learning Architecture
# ---------------------------------------------------------
class DualBranchGermlineNet(nn.Module):
    """
    Branch 1 (1D-CNN): Scans 21-bp DNA Sequence (168 one-hot features -> 21 positions x 8 channels)
    Branch 2 (Dense Block): Evaluates 8 Sequencing Quality & Biophysical Metrics
    Fusion Head: Combines both branches (128 + 32 = 160) -> 11 Cancer Classes
    """
    def __init__(self, num_classes=11):
        super(DualBranchGermlineNet, self).__init__()
        
        # Branch 1: 1D-CNN Sequence Scanner
        self.conv_branch = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Branch 2: Dense Metrics Block
        self.metric_branch = nn.Sequential(
            nn.Linear(8, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Fusion Layer & Classifier Head
        self.fusion_head = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x_dna, x_metrics):
        # Reshape DNA one-hot from (Batch, 168) -> (Batch, 8 channels, 21 positions)
        x_dna_spatial = x_dna.view(-1, 21, 8).permute(0, 2, 1)
        
        dna_features = self.conv_branch(x_dna_spatial)
        metric_features = self.metric_branch(x_metrics)
        
        fused = torch.cat([dna_features, metric_features], dim=1)
        return self.fusion_head(fused)


# ---------------------------------------------------------
# 2. Main Training & Evaluation Pipeline
# ---------------------------------------------------------
def train_first_model():
    print("=" * 75)
    print("FIRST MODEL BUILD: DUAL-BRANCH GERMLINE CANCER RISK PREDICTION")
    print("=" * 75)
    
    # 1. Load Dataset
    print(f"Loading Pure Germline Dataset: {DATASET_PATH}...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
        
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(df):,} pure germline variants with {len(df.columns)} columns.")
    
    # Extract Feature Subsets
    one_hot_cols = [c for c in df.columns if c.startswith("one_hot_")]
    metric_cols = ["dp", "af", "ro", "ao", "qual", "gt_code", "gc_content_21bp", "is_transition"]
    
    X_dna = df[one_hot_cols].values.astype(np.float32)       # (N, 168)
    X_metrics = df[metric_cols].values.astype(np.float32)   # (N, 8)
    y = df["cancer_class_id"].values.astype(np.int64)        # (N,)
    
    ALL_CLASS_NAMES = [
        "Oral Cavity Carcinoma",
        "Lung Cancer",
        "Female Breast Cancer",
        "Colorectal Cancer",
        "Prostate Cancer",
        "Stomach Cancer",
        "Thyroid Cancer",
        "Liver Cancer",
        "Bladder Cancer",
        "Cervical Cancer",
        "Non-Hodgkin Lymphoma"
    ]
    num_classes = 11
    
    print(f"Features: {X_dna.shape[1]} DNA One-Hot Channels + {X_metrics.shape[1]} Quality Metrics = {X_dna.shape[1] + X_metrics.shape[1]} Total")
    print(f"Target Classes: {len(np.unique(y))} unique represented cancer categories across {num_classes} total categories.")
    
    # 2. Stratified 70 / 15 / 15 Split
    print("\nSplitting Dataset (70% Train, 15% Validation, 15% Test)...")
    indices = np.arange(len(df))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices, y, test_size=0.30, random_state=42, stratify=y
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"   Train Set:      {len(idx_train):,} samples (70.0%)")
    print(f"   Validation Set: {len(idx_val):,} samples (15.0%)")
    print(f"   Test Set:       {len(idx_test):,} samples (15.0%)")
    
    # 3. Selective Scaling (Scale ONLY Metrics; DNA One-Hot stays untouched 0/1)
    scaler = StandardScaler()
    X_metrics_train = scaler.fit_transform(X_metrics[idx_train])
    X_metrics_val = scaler.transform(X_metrics[idx_val])
    X_metrics_test = scaler.transform(X_metrics[idx_test])
    
    with open(SCALER_SAVE_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved fitted StandardScaler to: {SCALER_SAVE_PATH}")
    
    X_dna_train = X_dna[idx_train]
    X_dna_val = X_dna[idx_val]
    X_dna_test = X_dna[idx_test]
    
    # 4. Balanced Class Weighting (Handles Imbalance across all 11 classes)
    present_train_classes = np.unique(y_train)
    computed_weights = compute_class_weight(class_weight='balanced', classes=present_train_classes, y=y_train)
    weights = np.ones(num_classes, dtype=np.float32)
    for cls, w in zip(present_train_classes, computed_weights):
        weights[cls] = w
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
    
    # 5. DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_dna_train, dtype=torch.float32),
        torch.tensor(X_metrics_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_dna_val, dtype=torch.float32),
        torch.tensor(X_metrics_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_dna_test, dtype=torch.float32),
        torch.tensor(X_metrics_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 6. Model, Optimizer, and Scheduler Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nInitializing DualBranchGermlineNet on Device: {device}")
    
    model = DualBranchGermlineNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-5)
    
    # 7. Training Loop with Early Stopping & Model Checkpointing
    MAX_EPOCHS = 35
    PATIENCE = 8
    best_val_loss = float("inf")
    patience_counter = 0
    
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }
    
    print("\n" + "=" * 75)
    print("TRAINING PROGRESS")
    print("=" * 75)
    
    start_time = time.time()
    
    for epoch in range(1, MAX_EPOCHS + 1):
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch_dna, batch_met, batch_y in train_loader:
            batch_dna = batch_dna.to(device)
            batch_met = batch_met.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_dna, batch_met)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(batch_y)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == batch_y).sum().item()
            train_total += len(batch_y)
            
        train_loss /= train_total
        train_acc = (train_correct / train_total) * 100.0
        scheduler.step()
        
        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for batch_dna, batch_met, batch_y in val_loader:
                batch_dna = batch_dna.to(device)
                batch_met = batch_met.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_dna, batch_met)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * len(batch_y)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == batch_y).sum().item()
                val_total += len(batch_y)
                
        val_loss /= val_total
        val_acc = (val_correct / val_total) * 100.0
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch [{epoch:02d}/{MAX_EPOCHS:02d}] | Train Loss: {train_loss:.4f} (Acc: {train_acc:5.1f}%) | Val Loss: {val_loss:.4f} (Acc: {val_acc:5.1f}%) | LR: {scheduler.get_last_lr()[0]:.6f}")
        sys.stdout.flush()
        
        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly Stopping triggered at epoch {epoch} (Validation loss did not improve for {PATIENCE} epochs).")
                break
                
    elapsed = time.time() - start_time
    print(f"\nTraining Complete in {elapsed:.1f} seconds! Best Val Loss: {best_val_loss:.4f}")
    
    # 8. Final Test Set Evaluation
    print("\n" + "=" * 75)
    print("EVALUATING BEST MODEL ON UNSEEN TEST SET (15% HOLDOUT)")
    print("=" * 75)
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
    model.eval()
    
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_dna, batch_met, batch_y in test_loader:
            batch_dna = batch_dna.to(device)
            batch_met = batch_met.to(device)
            
            outputs = model(batch_dna, batch_met)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    top1_acc = accuracy_score(all_targets, all_preds) * 100.0
    top3_acc = top_k_accuracy_score(all_targets, all_probs, k=3, labels=np.arange(num_classes)) * 100.0
    
    print(f"\n[*] FINAL TEST TOP-1 ACCURACY: {top1_acc:.2f}%")
    print(f"[*] FINAL TEST TOP-3 ACCURACY: {top3_acc:.2f}%")
    
    # Classification Report
    print("\nClassification Report across Hereditary Cancer Classes:")
    present_labels = np.sort(np.unique(all_targets))
    present_names = [ALL_CLASS_NAMES[i] for i in present_labels]
    report_dict = classification_report(all_targets, all_preds, labels=present_labels, target_names=present_names, output_dict=True, zero_division=0)
    print(classification_report(all_targets, all_preds, labels=present_labels, target_names=present_names, zero_division=0))
    sys.stdout.flush()
    
    # 9. Plot Confusion Matrix Heatmap (if matplotlib is available)
    if HAS_PLOTTING:
        cm = confusion_matrix(all_targets, all_preds, labels=np.arange(num_classes))
        plt.figure(figsize=(11, 9))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f"C{i}" for i in range(num_classes)],
                    yticklabels=[f"C{i}" for i in range(num_classes)])
        plt.title(f"11-Class Germline Cancer Predisposition Confusion Matrix\nTop-1 Accuracy: {top1_acc:.2f}% | Top-3 Accuracy: {top3_acc:.2f}%", fontsize=12, fontweight='bold')
        plt.xlabel("Predicted Cancer Class", fontsize=11)
        plt.ylabel("True Cancer Class", fontsize=11)
        plt.tight_layout()
        plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
        plt.close()
        print(f"Saved Confusion Matrix Plot to: {CONFUSION_MATRIX_PATH}")
        
        # 10. Plot Training & Validation Curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss", color="#2563eb", linewidth=2)
        plt.plot(history["val_loss"], label="Val Loss", color="#dc2626", linewidth=2)
        plt.title("Cross-Entropy Loss vs. Epochs", fontweight='bold')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history["train_acc"], label="Train Acc", color="#2563eb", linewidth=2)
        plt.plot(history["val_acc"], label="Val Acc", color="#16a34a", linewidth=2)
        plt.title("Classification Accuracy (%) vs. Epochs", fontweight='bold')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(TRAINING_CURVES_PATH, dpi=300)
        plt.close()
        print(f"Saved Training Curves Plot to: {TRAINING_CURVES_PATH}")
    else:
        print("\n[Notice] matplotlib/seaborn not installed; skipped plot PNG export.")
    
    # 11. Save Structured Metrics JSON
    metrics_summary = {
        "model_architecture": "DualBranchGermlineNet (1D-CNN + Tabular Dense Fusion)",
        "dataset": "Pure Mendelian Germline Cancer Susceptibility Dataset",
        "total_samples": len(df),
        "train_samples": len(idx_train),
        "val_samples": len(idx_val),
        "test_samples": len(idx_test),
        "top1_test_accuracy": top1_acc,
        "top3_test_accuracy": top3_acc,
        "best_validation_loss": best_val_loss,
        "training_time_seconds": elapsed,
        "classification_report": report_dict
    }
    
    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved Model Metrics JSON to: {METRICS_JSON_PATH}")
    
    print("\n" + "=" * 75)
    print("SUCCESS: FIRST GERMLINE MODEL BUILD COMPLETE!")
    print(f"Saved Weights:  {MODEL_SAVE_PATH}")
    print(f"Saved Metrics:  {METRICS_JSON_PATH}")
    print("=" * 75)
    return metrics_summary


if __name__ == "__main__":
    train_first_model()
