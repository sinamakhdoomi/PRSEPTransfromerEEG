#!/usr/bin/env python
"""
EEG source-space classification with a ResNet-18 backbone.

* Requires: Python ≥3.9, NumPy, SciPy, scikit-learn, PyTorch, seaborn, matplotlib.

Example:
$ python resnet18.py \
    --data_root ./source_arrays \
    --bands Delta Theta Alpha Beta Gamma \
    --output_dir ./results_resnet18 \
    --epochs 100 \
    --batch_size 32

Each band expects two NumPy arrays in data_root:
    <Band>_source_data.npy   – shape (N, D)
    <Band>_source_labels.npy – shape (N,)

Outputs:
    - Metrics and confusion matrices per band
    - Aggregated results in final_band_results_resnet18.npy
"""

import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, cohen_kappa_score)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time

# ----------------------------------------------------
# Argument Parser
# ----------------------------------------------------
def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results_resnet18')
    parser.add_argument('--bands', nargs='+', default=['Delta','Theta','Alpha','Beta','Gamma'])
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=1)
    parser.add_argument('--folds', type=int, default=2)
    return parser.parse_args()

# ----------------------------------------------------
# Dataset
# ----------------------------------------------------
class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = self.data[idx].reshape(1, 1884, 257).astype(np.float32)
        y = self.labels[idx]
        return torch.tensor(x), torch.tensor(y)

# ----------------------------------------------------
# ResNet-18 Definition
# ----------------------------------------------------
class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

def resnet18(num_classes):
    return ResNet(ResidualBlock, [2,2,2,2], num_classes)

# ----------------------------------------------------
# Main Execution
# ----------------------------------------------------
def main():
    args = build_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    band_results = {}

    for band in args.bands:
        x_path = os.path.join(args.data_root, f"{band}_source_data.npy")
        y_path = os.path.join(args.data_root, f"{band}_source_labels.npy")
        if not os.path.exists(x_path) or not os.path.exists(y_path):
            print(f"[WARN] {band} data missing. Skipping.")
            continue

        print(f"\n=== Processing {band} ===")
        x = np.load(x_path)
        y = np.load(y_path)
        n_classes = len(np.unique(y))

        weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        accs, precs, recs, f1s, kaps, times, confs = [], [], [], [], [], [], []

        for fold, (train_idx, test_idx) in enumerate(skf.split(x, y), 1):
            print(f"Fold {fold}/{args.folds}")
            model = resnet18(n_classes).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            criterion = nn.CrossEntropyLoss(weight=weights_tensor)

            train_ds = EEGDataset(x[train_idx], y[train_idx])
            test_ds  = EEGDataset(x[test_idx],  y[test_idx])
            train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            test_dl  = DataLoader(test_ds,  batch_size=args.batch_size)

            best_loss, stop_count = float('inf'), 0
            start = time.time()
            for ep in range(args.epochs):
                model.train()
                for xb, yb in train_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()

                if loss.item() < best_loss:
                    best_loss, stop_count = loss.item(), 0
                else:
                    stop_count += 1
                if stop_count >= args.patience:
                    break

            model.eval()
            all_preds, all_trues = [], []
            with torch.no_grad():
                for xb, yb in test_dl:
                    xb = xb.to(device)
                    out = model(xb)
                    preds = torch.argmax(out, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_trues.extend(yb.numpy())

            conf = confusion_matrix(all_trues, all_preds)
            accs.append(accuracy_score(all_trues, all_preds))
            precs.append(precision_score(all_trues, all_preds, average='weighted', zero_division=0))
            recs.append(recall_score(all_trues, all_preds, average='weighted', zero_division=0))
            f1s.append(f1_score(all_trues, all_preds, average='weighted', zero_division=0))
            kaps.append(cohen_kappa_score(all_trues, all_preds))
            times.append(time.time() - start)
            confs.append(conf)

        mean_cm = sum(confs).astype(float)
        for i in range(mean_cm.shape[0]):
            mean_cm[i, :] /= mean_cm[i, :].sum() or 1
        mean_cm *= 100

        plt.figure(figsize=(6,5))
        sns.heatmap(mean_cm, annot=True, fmt=".2f", cmap="Blues")
        plt.title(f"{band} Mean Confusion Matrix (%)")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.savefig(os.path.join(args.output_dir, f"{band}_cm.png"), dpi=150)
        plt.close()

        pd.DataFrame(mean_cm).to_csv(os.path.join(args.output_dir, f"{band}_cm.csv"))

        # Save text results
        m_acc, s_acc = np.mean(accs)*100, np.std(accs)*100
        m_prec,s_prec= np.mean(precs)*100, np.std(precs)*100
        m_rec, s_rec = np.mean(recs)*100, np.std(recs)*100
        m_f1, s_f1   = np.mean(f1s)*100, np.std(f1s)*100
        m_kap,s_kap  = np.mean(kaps)*100, np.std(kaps)*100
        m_t, s_t     = np.mean(times), np.std(times)

        with open(os.path.join(args.output_dir, f"{band}_metrics.txt"), 'w') as f:
            f.write(f"{band} Band - K-Fold Results:\n\n")
            f.write(f"Mean Accuracy:  {m_acc:.2f}% ± {s_acc:.2f}\n")
            f.write(f"Mean Precision: {m_prec:.2f}% ± {s_prec:.2f}\n")
            f.write(f"Mean Recall:    {m_rec:.2f}% ± {s_rec:.2f}\n")
            f.write(f"Mean F1 Score:  {m_f1:.2f}% ± {s_f1:.2f}\n")
            f.write(f"Mean Kappa:     {m_kap:.2f}% ± {s_kap:.2f}\n")
            f.write(f"Mean Fold Runtime: {m_t:.2f}s ± {s_t:.2f}s\n")

        band_results[band] = {
            "mean_acc": m_acc, "std_acc": s_acc,
            "mean_prec": m_prec, "std_prec": s_prec,
            "mean_rec": m_rec, "std_rec": s_rec,
            "mean_f1": m_f1, "std_f1": s_f1,
            "mean_kappa": m_kap, "std_kappa": s_kap,
            "mean_time": m_t, "std_time": s_t
        }

    np.save(os.path.join(args.output_dir, "final_band_results_resnet18.npy"), band_results)
    print("\nAll bands processed. Results saved.")

if __name__ == '__main__':
    main()
