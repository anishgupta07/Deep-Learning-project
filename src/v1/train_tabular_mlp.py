# ==============================================================================
# GenomeDL v2: Deep Tabular Residual Network (ResNet-Tabular)
# Pan-Cancer Hereditary Cancer Predisposition Engine (80+ Genes)
# ==============================================================================

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Set deterministic random seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -----------------------------------------------------------------------------
# 1. PyTorch Dataset with Entity Embeddings
# -----------------------------------------------------------------------------
class PanCancerDataset(Dataset):
    def __init__(self, X_cont, gene_ids, syndrome_ids, y):
        self.X_cont = torch.tensor(X_cont, dtype=torch.float32)
        self.gene_ids = torch.tensor(gene_ids, dtype=torch.long)
        self.syndrome_ids = torch.tensor(syndrome_ids, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X_cont[idx], self.gene_ids[idx], self.syndrome_ids[idx], self.y[idx]

def prepare_pan_cancer_data(csv_path=None):
    if csv_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, "data", "pan_cancer_germline_dataset.csv")
        if os.path.exists(data_path):
            csv_path = data_path
        else:
            csv_path = os.path.join(script_dir, "pan_cancer_germline_dataset.csv")
            
    print(f"[*] Loading Pan-Cancer dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[+] Loaded {len(df):,} variants across {df['gene'].nunique()} genes.")
    
    # Categorical encodings
    genes = sorted(df['gene'].unique())
    gene_to_id = {g: i for i, g in enumerate(genes)}
    df['gene_id'] = df['gene'].map(gene_to_id)
    
    syndromes = sorted(df['cancer_syndrome'].unique()) if 'cancer_syndrome' in df.columns else ['PanCancer']
    syn_to_id = {s: i for i, s in enumerate(syndromes)}
    df['syndrome_id'] = df['cancer_syndrome'].map(syn_to_id) if 'cancer_syndrome' in df.columns else 0
    
    # Nucleotide ref/alt one-hot
    for base in ['A', 'C', 'G', 'T']:
        df[f'ref_{base}'] = (df['ref'] == base).astype(float)
        df[f'alt_{base}'] = (df['alt'] == base).astype(float)
        
    cont_cols = [
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
    
    X_cont = df[cont_cols].astype(float).copy()
    gene_ids = df['gene_id'].values
    syndrome_ids = df['syndrome_id'].values
    y = df['label'].values
    
    # Stratified Train (70%) / Val (15%) / Test (15%)
    idx = np.arange(len(df))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=SEED, stratify=y[temp_idx])
    
    # Scale continuous biophysical features
    scale_cols = ['delta_mw', 'delta_logp', 'delta_tpsa', 'delta_charge', 'grantham_score']
    scaler = StandardScaler()
    X_cont.loc[train_idx, scale_cols] = scaler.fit_transform(X_cont.loc[train_idx, scale_cols])
    X_cont.loc[val_idx, scale_cols] = scaler.transform(X_cont.loc[val_idx, scale_cols])
    X_cont.loc[test_idx, scale_cols] = scaler.transform(X_cont.loc[test_idx, scale_cols])
    
    print(f"[+] Total Continuous Features: {len(cont_cols)}")
    print(f"[+] Gene Categories: {len(genes)} (Embedded into 16-D vectors)")
    print(f"[+] Syndrome Categories: {len(syndromes)} (Embedded into 8-D vectors)")
    print(f"    Train size: {len(train_idx):,} ({y[train_idx].sum():,} positive)")
    print(f"    Val size:   {len(val_idx):,} ({y[val_idx].sum():,} positive)")
    print(f"    Test size:  {len(test_idx):,} ({y[test_idx].sum():,} positive)")
    
    splits = {
        'train': (X_cont.iloc[train_idx].values, gene_ids[train_idx], syndrome_ids[train_idx], y[train_idx]),
        'val':   (X_cont.iloc[val_idx].values, gene_ids[val_idx], syndrome_ids[val_idx], y[val_idx]),
        'test':  (X_cont.iloc[test_idx].values, gene_ids[test_idx], syndrome_ids[test_idx], y[test_idx]),
        'num_genes': len(genes),
        'num_syndromes': len(syndromes),
        'gene_to_id': gene_to_id,
        'scaler': scaler
    }
    return splits

# -----------------------------------------------------------------------------
# 2. ResNet Tabular Architecture with Skip Connections
# -----------------------------------------------------------------------------
class TabularResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout)
        )
        self.act = nn.GELU()
        
    def forward(self, x):
        return self.act(x + self.block(x)) # Residual Skip Connection: x + F(x)

class GenomeDLResNet(nn.Module):
    def __init__(self, num_cont, num_genes, num_syndromes, hidden_dim=256, gene_emb_dim=16, syn_emb_dim=8):
        super().__init__()
        # Entity Embeddings
        self.gene_embed = nn.Embedding(num_genes, gene_emb_dim)
        self.syndrome_embed = nn.Embedding(num_syndromes, syn_emb_dim)
        
        in_dim = num_cont + gene_emb_dim + syn_emb_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Residual Blocks
        self.res1 = TabularResidualBlock(hidden_dim, dropout=0.2)
        self.res2 = TabularResidualBlock(hidden_dim, dropout=0.2)
        
        # Output Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1) # Calibrated Pathogenicity Logit
        )
        
    def forward(self, x_cont, gene_id, syn_id):
        g_emb = self.gene_embed(gene_id)
        s_emb = self.syndrome_embed(syn_id)
        x = torch.cat([x_cont, g_emb, s_emb], dim=1)
        x = self.input_proj(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.head(x)

# -----------------------------------------------------------------------------
# 3. Training & Validation Engine
# -----------------------------------------------------------------------------
def train_pan_cancer_model(epochs=25, batch_size=256, lr=1e-3, weight_decay=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Training on Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    data = prepare_pan_cancer_data()
    train_ds = PanCancerDataset(*data['train'])
    val_ds   = PanCancerDataset(*data['val'])
    test_ds  = PanCancerDataset(*data['test'])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True if torch.cuda.is_available() else False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    num_cont = data['train'][0].shape[1]
    model = GenomeDLResNet(
        num_cont=num_cont,
        num_genes=data['num_genes'],
        num_syndromes=data['num_syndromes'],
        hidden_dim=256
    ).to(device)
    
    # Class imbalance compensation
    y_train = data['train'][3]
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / max(1, pos_count)], dtype=torch.float32).to(device)
    print(f"[+] Class Imbalance Ratio: 1:{pos_weight.item():.2f} (Applying Weighted BCE Loss)")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_auc = 0.0
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "pan_cancer_germline_resnet.pth")
    t0 = time.time()
    
    print("\n" + "="*80)
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Val Loss':<14}{'Val AUC':<14}{'Val Recall':<14}{'LR':<10}")
    print("="*80)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for x_cont, g_id, s_id, y_batch in train_loader:
            x_cont, g_id, s_id, y_batch = x_cont.to(device), g_id.to(device), s_id.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_cont, g_id, s_id)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            
        train_loss = total_loss / len(train_ds)
        
        # Validation pass
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for x_cont, g_id, s_id, y_batch in val_loader:
                x_cont, g_id, s_id, y_batch = x_cont.to(device), g_id.to(device), s_id.to(device), y_batch.to(device)
                logits = model(x_cont, g_id, s_id)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(y_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(y_batch.cpu().numpy())
                
        val_loss = val_loss / len(val_ds)
        val_preds = np.array(val_preds).ravel()
        val_targets = np.array(val_targets).ravel()
        val_auc = roc_auc_score(val_targets, val_preds)
        val_recall = recall_score(val_targets, (val_preds >= 0.5).astype(int))
        
        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]
        
        print(f"{epoch:<8}{train_loss:<14.4f}{val_loss:<14.4f}{val_auc:<14.4f}{val_recall*100:<13.2f}%{cur_lr:<10.6f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': data['scaler'],
                'gene_to_id': data['gene_to_id'],
                'best_val_auc': best_val_auc
            }, best_model_path)
            
    elapsed = time.time() - t0
    print("="*80)
    print(f"[+] Training completed in {elapsed:.1f} seconds! Best Val ROC-AUC: {best_val_auc:.4f}")
    
    # Final Unseen Test Evaluation
    print(f"\n[*] Evaluating on Completely Unseen Test Split ({len(test_ds):,} variants)...")
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_preds, test_targets = [], []
    with torch.no_grad():
        for x_cont, g_id, s_id, y_batch in test_loader:
            x_cont, g_id, s_id = x_cont.to(device), g_id.to(device), s_id.to(device)
            probs = torch.sigmoid(model(x_cont, g_id, s_id)).cpu().numpy()
            test_preds.extend(probs)
            test_targets.extend(y_batch.numpy())
            
    test_preds = np.array(test_preds).ravel()
    test_targets = np.array(test_targets).ravel()
    
    test_auc = roc_auc_score(test_targets, test_preds)
    test_acc = accuracy_score(test_targets, (test_preds >= 0.5).astype(int))
    test_recall = recall_score(test_targets, (test_preds >= 0.5).astype(int))
    test_prec = precision_score(test_targets, (test_preds >= 0.5).astype(int))
    test_f1 = f1_score(test_targets, (test_preds >= 0.5).astype(int))
    
    print("\n" + "="*50)
    print(">>> FINAL PAN-CANCER TEST BENCHMARKS <<<")
    print("="*50)
    print(f"  • ROC-AUC Score : {test_auc:.4f} ({test_auc*100:.2f}%)")
    print(f"  • Sensitivity   : {test_recall:.4f} ({test_recall*100:.2f}%)")
    print(f"  • Precision     : {test_prec:.4f} ({test_prec*100:.2f}%)")
    print(f"  • Accuracy      : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  • F1-Score      : {test_f1:.4f}")
    print("="*50)

if __name__ == '__main__':
    train_pan_cancer_model()
