import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# -----------------------------------------------------------------------------
# 1. Dataset & Feature Engineering
# -----------------------------------------------------------------------------
class GermlineVariantDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_data(csv_path="brca_germline_pilot_dataset.csv"):
    print(f"[*] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[+] Loaded {len(df):,} variants.")
    
    # Gene one-hot
    df['is_brca1'] = (df['gene'] == 'BRCA1').astype(float)
    df['is_brca2'] = (df['gene'] == 'BRCA2').astype(float)
    
    # Nucleotide ref/alt one-hot
    for base in ['A', 'C', 'G', 'T']:
        df[f'ref_{base}'] = (df['ref'] == base).astype(float)
        df[f'alt_{base}'] = (df['alt'] == base).astype(float)
        
    feature_cols = [
        'is_brca1', 'is_brca2',
        'ref_A', 'ref_C', 'ref_G', 'ref_T',
        'alt_A', 'alt_C', 'alt_G', 'alt_T',
        'rel_protein_pos',
        'is_transition',
        'is_missense',
        'is_nonsense',
        'is_synonymous',
        'delta_mw',
        'delta_logp',
        'delta_tpsa',
        'delta_charge',
        'tanimoto_chem_dist',
        'grantham_score'
    ]
    
    X = df[feature_cols].copy()
    y = df['label'].values
    
    # Train / Val / Test split (70 / 15 / 15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    # Scale continuous features
    cont_cols = ['delta_mw', 'delta_logp', 'delta_tpsa', 'delta_charge', 'grantham_score']
    scaler = StandardScaler()
    X_train[cont_cols] = scaler.fit_transform(X_train[cont_cols])
    X_val[cont_cols] = scaler.transform(X_val[cont_cols])
    X_test[cont_cols] = scaler.transform(X_test[cont_cols])
    
    print(f"[+] Feature vector dimension: {len(feature_cols)}")
    print(f"    Train size: {len(X_train):,} ({y_train.sum():,} positive)")
    print(f"    Val size:   {len(X_val):,} ({y_val.sum():,} positive)")
    print(f"    Test size:  {len(X_test):,} ({y_test.sum():,} positive)")
    
    return (
        X_train.values, y_train,
        X_val.values, y_val,
        X_test.values, y_test,
        feature_cols, scaler
    )

# -----------------------------------------------------------------------------
# 2. Tabular MLP Architecture
# -----------------------------------------------------------------------------
class GermlineTabularMLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------
# 3. Training & Validation Loop
# -----------------------------------------------------------------------------
def train_model(epochs=35, batch_size=128, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Training on Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, scaler = prepare_data()
    
    train_dataset = GermlineVariantDataset(X_train, y_train)
    val_dataset = GermlineVariantDataset(X_val, y_val)
    test_dataset = GermlineVariantDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Handle Class Imbalance (Benign: ~80%, Pathogenic: ~20%)
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()], device=device)
    print(f"[*] Class weighting (pos_weight for Pathogenic): {pos_weight.item():.2f}")
    
    model = GermlineTabularMLP(in_features=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    
    best_val_auc = 0.0
    best_model_path = "germline_brca_tabular_mlp.pth"
    
    print("\n" + "=" * 70)
    print(f"{'Epoch':<7} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val Acc':<9} | {'Val ROC-AUC':<11} | {'Val F1':<8}")
    print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_x)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(batch_y.cpu().numpy())
                
        val_loss /= len(val_dataset)
        val_preds = np.array(val_preds).flatten()
        val_targets = np.array(val_targets).flatten()
        val_binary = (val_preds >= 0.5).astype(int)
        
        val_acc = accuracy_score(val_targets, val_binary)
        val_auc = roc_auc_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_binary)
        
        scheduler.step(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            mark = "[BEST]"
        else:
            mark = ""
            
        if epoch % 5 == 0 or epoch == 1 or mark:
            print(f"{epoch:<7} | {train_loss:<12.4f} | {val_loss:<10.4f} | {val_acc:<9.4f} | {val_auc:<11.4f} | {val_f1:<8.4f} {mark}")
            
    print("=" * 70)
    print(f"\n[+] Best Validation ROC-AUC: {best_val_auc:.4f}")
    
    # -------------------------------------------------------------------------
    # 4. Final Evaluation on Unseen Test Set
    # -------------------------------------------------------------------------
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_preds.extend(probs)
            test_targets.extend(batch_y.cpu().numpy())
            
    test_preds = np.array(test_preds).flatten()
    test_targets = np.array(test_targets).flatten()
    test_binary = (test_preds >= 0.5).astype(int)
    
    test_acc = accuracy_score(test_targets, test_binary)
    test_prec = precision_score(test_targets, test_binary)
    test_rec = recall_score(test_targets, test_binary)
    test_f1 = f1_score(test_targets, test_binary)
    test_auc = roc_auc_score(test_targets, test_preds)
    cm = confusion_matrix(test_targets, test_binary)
    
    print("\n" + "=" * 70)
    print(">>> FINAL TEST SET PERFORMANCE METRICS (UNSEEN 15% TEST SPLIT)")
    print("=" * 70)
    print(f"Accuracy:           {test_acc * 100:.2f}%")
    print(f"ROC-AUC Score:      {test_auc:.4f}")
    print(f"F1 Score:           {test_f1:.4f}")
    print(f"Precision:          {test_prec:.4f}")
    print(f"Recall (Sensitivity): {test_rec:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Benign (TN):      {cm[0, 0]:<5} | False Pathogenic (FP): {cm[0, 1]}")
    print(f"  False Benign (FN):     {cm[1, 0]:<5} | True Pathogenic (TP):  {cm[1, 1]}")
    print("=" * 70)
    print(f"[+] Model checkpoint saved to: {best_model_path}")

if __name__ == "__main__":
    train_model()
