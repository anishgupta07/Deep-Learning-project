#!/usr/bin/env python3
"""
Germline Cancer Susceptibility Inference Tool
=============================================
Runs deep learning multi-class risk predictions for patient germline variants
using the trained Dual-Branch 1D-CNN + Tabular Fusion Network.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Model Definition
class DualBranchGermlineNet(nn.Module):
    def __init__(self, num_classes=11):
        super(DualBranchGermlineNet, self).__init__()
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
        self.metric_branch = nn.Sequential(
            nn.Linear(8, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
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
        x_dna_spatial = x_dna.view(-1, 21, 8).permute(0, 2, 1)
        dna_features = self.conv_branch(x_dna_spatial)
        metric_features = self.metric_branch(x_metrics)
        fused = torch.cat([dna_features, metric_features], dim=1)
        return self.fusion_head(fused)


CLASS_NAMES = [
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


def run_sample_inference(sample_count=5):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "best_germline_model.pth")
    scaler_path = os.path.join(base_dir, "germline_scaler.pkl")
    dataset_path = os.path.join(base_dir, "..", "multiclass_cancer_dataset", "master_dataset", "germline_cancer_master_dataset.csv")
    
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}.")
        print("Please train the model first by running: python train_first_model.py")
        sys.exit(1)
        
    if not os.path.exists(scaler_path):
        print(f"Error: Fitted scaler not found at {scaler_path}.")
        print("Please train the model first to generate the scaler.")
        sys.exit(1)
        
    print(f"Loading trained weights from: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualBranchGermlineNet(num_classes=11).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    print(f"Loading random test samples from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    sample_df = df.sample(n=sample_count, random_state=42).reset_index(drop=True)
    
    one_hot_cols = [c for c in df.columns if c.startswith("one_hot_")]
    metric_cols = ["dp", "af", "ro", "ao", "qual", "gt_code", "gc_content_21bp", "is_transition"]
    
    X_dna = sample_df[one_hot_cols].values.astype(np.float32)
    X_metrics = sample_df[metric_cols].values.astype(np.float32)
    X_metrics_scaled = scaler.transform(X_metrics)
    
    tensor_dna = torch.tensor(X_dna, dtype=torch.float32).to(device)
    tensor_metrics = torch.tensor(X_metrics_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model(tensor_dna, tensor_metrics)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        
    print("\n" + "=" * 80)
    print("GERMLINE MULTI-CLASS CANCER RISK PREDICTION INFERENCE REPORT")
    print("=" * 80)
    
    for i, row in sample_df.iterrows():
        pred_idx = preds[i]
        true_idx = int(row["cancer_class_id"])
        pred_name = CLASS_NAMES[pred_idx]
        true_name = str(row["cancer_class_name"])
        top_prob = probs[i][pred_idx] * 100.0
        
        status = "CORRECT" if pred_idx == true_idx else "MISMATCH"
        print(f"\nVariant #{i+1}: ID={row['variant_id']} (chr{row['chrom']}:{row['pos']} {row['ref']}>{row['alt']})")
        print(f"   Zygosity:    {row.get('germline_zygosity', 'N/A')} (AF: {row['af']:.3f}, DP: {int(row['dp'])})")
        print(f"   True Class:  [{true_idx}] {true_name}")
        print(f"   Pred Class:  [{pred_idx}] {pred_name} (Confidence: {top_prob:.2f}%) -> {status}")
        print("   Top-3 Predisposition Risk Vector:")
        top3_indices = np.argsort(probs[i])[::-1][:3]
        for rank, c_idx in enumerate(top3_indices, 1):
            bar = "#" * int(probs[i][c_idx] * 25)
            print(f"      {rank}. {CLASS_NAMES[c_idx]:<25}: {probs[i][c_idx]*100:5.2f}% {bar}")
            
    print("\n" + "=" * 80)


if __name__ == "__main__":
    count = 5
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            count = 5
    run_sample_inference(count)
