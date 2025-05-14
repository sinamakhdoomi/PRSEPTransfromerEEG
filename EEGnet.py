#!/usr/bin/env python
"""
EEG source-space classification with a EEGnet.

* Requires: Python ≥3.9, NumPy, SciPy, scikit-learn, PyTorch, seaborn, matplotlib.

Example:
$ python EEGnet.py \
    --data_root ./source_arrays \
    --bands Delta Theta Alpha Beta Gamma \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --n_channels 1884 \
    --n_samples 257 \
    --folds 5 \
    --patience 10 \
    --output_dir ./results_eegnet

Each band expects two NumPy arrays in data_root:
    <Band>_source_data.npy   – shape (N, D)
    <Band>_source_labels.npy – shape (N,)

Outputs:
    - Metrics and confusion matrices per band
    - Aggregated results in eegnet_band_results.npy
"""

import os
import gc
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, cohen_kappa_score, classification_report)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------
# Argument Parser
# ----------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="EEGNet band-wise classification")
    parser.add_argument("--data_root", type=Path, required=True, help="Directory with *_source_data.npy and *_source_labels.npy")
    parser.add_argument("--bands", nargs="+", default=["Delta", "Theta", "Alpha", "Beta", "Gamma"], help="Frequency bands to process")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_channels", type=int, default=1884)
    parser.add_argument("--n_samples", type=int, default=257)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output_dir", type=Path, default=Path("./results_eegnet"))
    return parser.parse_args()

# ----------------------------------------------------
# Dataset
# ----------------------------------------------------
class EEGDataset(Dataset):
    def __init__(self, data, labels, n_channels=1884, n_samples=257):
        self.data = data
        self.labels = labels
        self.n_channels = n_channels
        self.n_samples = n_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        x_tensor = torch.tensor(x, dtype=torch.float32).view(1, self.n_channels, self.n_samples)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return x_tensor, y_tensor

# ----------------------------------------------------
# EEGNet Definition
# ----------------------------------------------------
class EEGNet(nn.Module):
    def __init__(self, num_classes=3, n_channels=1884, n_samples=257):
        super(EEGNet, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8)
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(n_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=0.25)
        )
        self.separable = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.25)
        )
        self.classifier = nn.Linear(16 * 1 * (n_samples // 32), num_classes)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ----------------------------------------------------
# Band Classification Function
# ----------------------------------------------------
def run_band_classification(band_name, data, labels, cfg, device):
    skf = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=42)
    metrics = {k: [] for k in ('acc', 'prec', 'rec', 'f1', 'kappa')}
    cms, times = [], []

    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    for fold, (tr, te) in enumerate(skf.split(data, labels), 1):
        print(f"\nFold {fold}/{cfg.folds} — {band_name}")
        model = EEGNet(num_classes=len(np.unique(labels)), n_channels=cfg.n_channels, n_samples=cfg.n_samples).to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        train_ds = EEGDataset(data[tr], labels[tr], cfg.n_channels, cfg.n_samples)
        test_ds  = EEGDataset(data[te], labels[te], cfg.n_channels, cfg.n_samples)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size)

        best_loss, no_imp = float('inf'), 0
        t0 = time.time()
        for ep in range(cfg.epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            model.eval()
            val_loss = 0.0; preds, truths = [], []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    val_loss += criterion(logits, yb).item()
                    preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    truths.extend(yb.cpu().numpy())
            if val_loss < best_loss:
                best_loss, no_imp = val_loss, 0
            else:
                no_imp += 1
                if no_imp >= cfg.patience:
                    break
        times.append(time.time() - t0)
        cms.append(confusion_matrix(truths, preds))
        metrics['acc'].append(accuracy_score(truths, preds))
        metrics['prec'].append(precision_score(truths, preds, average='weighted', zero_division=0))
        metrics['rec'].append(recall_score(truths, preds, average='weighted', zero_division=0))
        metrics['f1'].append(f1_score(truths, preds, average='weighted', zero_division=0))
        metrics['kappa'].append(cohen_kappa_score(truths, preds))
        gc.collect(); torch.cuda.empty_cache()

    # Report and save
    m = {k: 100*np.mean(v) if k != 'time' else np.mean(v) for k, v in metrics.items()}
    s = {k: 100*np.std(v) if k != 'time' else np.std(v) for k, v in metrics.items()}
    m['time'], s['time'] = np.mean(times), np.std(times)

    print(f"\n=== {band_name} Band Results ===")
    for k in metrics: print(f"{k.capitalize()}: {m[k]:.2f}% ± {s[k]:.2f}%")

    cm_mean = sum(cms).astype(float)
    cm_mean = (cm_mean.T / cm_mean.sum(1)).T * 100
    df_cm = pd.DataFrame(cm_mean,
                         index=[f"Class {i}" for i in range(cm_mean.shape[0])],
                         columns=[f"Class {i}" for i in range(cm_mean.shape[1])])
    df_cm.to_csv(cfg.output_dir / f"{band_name}_confusion_matrix.csv")
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt=".1f", cmap="Blues")
    plt.title(f"{band_name} Confusion Matrix (%)")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(cfg.output_dir / f"{band_name}_confusion_matrix.png")
    plt.close()

    with open(cfg.output_dir / f"{band_name}_metrics.txt", 'w') as f:
        for k in metrics:
            f.write(f"{k.capitalize()}: {m[k]:.2f}% ± {s[k]:.2f}%\n")
        f.write(f"Mean Runtime: {m['time']:.2f}s ± {s['time']:.2f}s\n")

    return m

# ----------------------------------------------------
# Main
# ----------------------------------------------------
def main():
    cfg = get_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    results = {}
    for band in cfg.bands:
        data_file = cfg.data_root / f"{band}_source_data.npy"
        label_file = cfg.data_root / f"{band}_source_labels.npy"
        if not (data_file.exists() and label_file.exists()):
            print(f"Skipping {band} — missing files")
            continue
        data = np.load(data_file)
        labels = np.load(label_file)
        results[band] = run_band_classification(band, data, labels, cfg, device)

    np.save(cfg.output_dir / "eegnet_band_results.npy", results)
    print("\nAll bands processed. Results saved.")

if __name__ == "__main__":
    main()
