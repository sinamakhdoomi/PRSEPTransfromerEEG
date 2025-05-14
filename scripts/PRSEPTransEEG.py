#!/usr/bin/env python
"""
EEG band–specific classification with SimCLR pre–training.

Python ≥3.9, PyTorch, NumPy, pandas, scikit–learn, seaborn, and
matplotlib are required.
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    classification_report)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

# -----------------------------------------------------------------------------
# argument parsing
# -----------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser("EEG band classifier")
    p.add_argument("--data_root",         type=Path,   default=Path("."),              help="Folder with *_source_data.npy")
    p.add_argument("--bands",             nargs="+",   default=["Delta","Theta","Alpha","Beta","Gamma"])
    p.add_argument("--pretrain_band",     type=str,    default="Theta",                   help="Band for SimCLR pretraining")
    p.add_argument("--folds",             type=int,    default=5,                         help="Number of cross‐val folds")
    p.add_argument("--pretrain_epochs",   type=int,    default=50,                        help="SimCLR pretrain epochs")
    p.add_argument("--class_epochs",      type=int,    default=100,                       help="Classifier train epochs")
    p.add_argument("--batch_size",        type=int,    default=32,                        help="Batch size for both stages")
    p.add_argument("--lr",                type=float,  default=1e-4,                      help="Learning rate for classifier")
    p.add_argument("--simclr_lr",         type=float,  default=1e-6,                      help="Learning rate for SimCLR")
    p.add_argument("--pretrain_patience", type=int,    default=3,                         help="Patience for SimCLR early stop")
    p.add_argument("--train_patience",    type=int,    default=10,                        help="Patience for classifier early stop")
    p.add_argument("--n_channels",        type=int,    default=1884,                      help="Number of channels (flattened dim part 1)")
    p.add_argument("--n_timepoints",      type=int,    default=257,                       help="Number of timepoints (flattened dim part 2)")
    p.add_argument("--seed",              type=int,    default=42)
    return p.parse_args()

# -----------------------------------------------------------------------------
# dataset wrappers
# -----------------------------------------------------------------------------
class VectorDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.x = torch.from_numpy(data).float()
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx]

class EEGDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray, cfg):
        self.x = torch.from_numpy(data).float()
        self.y = torch.from_numpy(labels).long()
        self.C, self.T = cfg.n_channels, cfg.n_timepoints
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        v = self.x[idx].view(1, self.C, self.T)
        return v, self.y[idx]

# -----------------------------------------------------------------------------
# SimCLR components
# -----------------------------------------------------------------------------
def random_augment(x: torch.Tensor, noise_std=0.01, dropout=0.02):
    x_noisy = x + noise_std * torch.randn_like(x)
    mask = (torch.rand_like(x_noisy) > dropout).float()
    return x_noisy * mask

class SimCLRModel(nn.Module):
    def __init__(self, dim_in, hidden=1024, proj=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden, proj),
            nn.ReLU(inplace=True),
            nn.Linear(proj, proj),
        )
    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z

def nt_xent(z1, z2, t=0.07):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    sim = torch.matmul(z1, z2.T) / t
    target = torch.arange(z1.size(0), device=z1.device)
    return nn.functional.cross_entropy(sim, target)

def pretrain_simclr(x: np.ndarray, cfg, device):
    loader = DataLoader(VectorDataset(x), batch_size=cfg.batch_size, shuffle=True)
    model = SimCLRModel(x.shape[1]).to(device)
    opt   = optim.Adam(model.parameters(), lr=cfg.simclr_lr)
    best, no_imp = float("inf"), 0
    for ep in range(1, cfg.pretrain_epochs+1):
        total = 0.0
        for xb in loader:
            xb = xb.to(device)
            z1, z2 = model(random_augment(xb)), model(random_augment(xb))
            loss = nt_xent(z1[1], z2[1])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        avg = total/len(loader)
        print(f"[SimCLR] epoch {ep}/{cfg.pretrain_epochs} loss={avg:.4f}")
        if avg < best:
            best, no_imp = avg, 0
        else:
            no_imp += 1
            if no_imp >= cfg.pretrain_patience:
                break
    return model

# -----------------------------------------------------------------------------
# PRSEPTrans‑EEG model
# -----------------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(c, c//r, bias=False), nn.ReLU(inplace=True),
            nn.Linear(c//r, c, bias=False),   nn.Sigmoid()
        )
    def forward(self, x):
        b,c,_,_ = x.shape
        w = self.avg(x).view(b,c)
        w = self.fc(w).view(b,c,1,1)
        return x * w

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.down  = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            ) if (stride!=1 or in_ch!=out_ch) else nn.Identity()
        )
        self.se    = SEBlock(out_ch)
    def forward(self, x):
        idt = self.down(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out) + idt
        return self.relu(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d, maxlen=3000):
        super().__init__()
        pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d,2).float()*-(np.log(10000.0)/d))
        pe  = torch.zeros(maxlen,d)
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:,:x.size(1)]

class PRSEPTransEEG(nn.Module):
    def __init__(self, n_classes=3, d_model=64, nhead=4, layers=1, ff=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1,16,3,1,1,bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.res  = nn.Sequential(
            ResidualBlock(16,32,2),
            ResidualBlock(32,64,2),
            ResidualBlock(64,64,1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1,None))
        self.embed= nn.Linear(64, d_model)
        self.pos  = PositionalEncoding(d_model)
        enc_layer= nn.TransformerEncoderLayer(d_model,nhead,ff,batch_first=True)
        self.tf   = nn.TransformerEncoder(enc_layer, layers)
        self.cls  = nn.Linear(d_model, n_classes)
    def forward(self, x):
        x = self.pool(self.res(self.stem(x))).squeeze(2)       # (B,C,T)
        x = self.embed(x.permute(0,2,1))                       # (B,T,d)
        x = self.tf(self.pos(x)).mean(1)                       # (B,d)
        return self.cls(x)

# -----------------------------------------------------------------------------
# run classification
# -----------------------------------------------------------------------------
def run_classification(x, y, simclr, band, cfg, device):
    kf = StratifiedKFold(cfg.folds, shuffle=True, random_state=cfg.seed)
    cms, times = [], []
    accs, precs, recs, f1s, kaps = [], [], [], [], []

    for tr, te in kf.split(x,y):
        ds_tr = EEGDataset(x[tr], y[tr], cfg)
        ds_te = EEGDataset(x[te], y[te], cfg)
        dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True)
        dl_te = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False)

        model = PRSEPTransEEG(n_classes=len(np.unique(y))).to(device)
        opt   = optim.Adam(model.parameters(), lr=cfg.lr)
        crit  = nn.CrossEntropyLoss()

        best, no_imp = float("inf"), 0
        t0 = time.time()
        for ep in range(cfg.class_epochs):
            model.train()
            for xb,yb in dl_tr:
                xb,yb = xb.to(device), yb.to(device)
                loss = crit(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
            if loss.item() < best:
                best, no_imp = loss.item(), 0
            else:
                no_imp += 1
                if no_imp >= cfg.train_patience:
                    break

        model.eval()
        preds, targs = [], []
        with torch.no_grad():
            for xb,yb in dl_te:
                xb,yb = xb.to(device), yb.to(device)
                out = model(xb)
                preds.extend(torch.argmax(out,1).cpu().numpy())
                targs.extend(yb.cpu().numpy())

        cms.append(confusion_matrix(targs, preds))
        accs.append(accuracy_score(targs, preds))
        precs.append(precision_score(targs, preds, average="weighted", zero_division=0))
        recs.append(recall_score(targs, preds, average="weighted", zero_division=0))
        f1s.append(f1_score(targs, preds, average="weighted", zero_division=0))
        kaps.append(cohen_kappa_score(targs, preds))
        times.append(time.time()-t0)
        gc.collect()

    # aggregate & print
    m_acc, s_acc = np.mean(accs)*100, np.std(accs)*100
    m_prec,s_prec= np.mean(precs)*100, np.std(precs)*100
    m_rec, s_rec   = np.mean(recs)*100, np.std(recs)*100
    m_f1, s_f1     = np.mean(f1s)*100, np.std(f1s)*100
    m_kap,s_kap    = np.mean(kaps)*100, np.std(kaps)*100
    m_t, s_t       = np.mean(times), np.std(times)

    print(f"\n=======================")
    print(f"K-Fold Results for band")
    print("=======================")
    print(f"Mean Accuracy:  {m_acc:.2f}% ± {s_acc:.2f}")
    print(f"Mean Precision: {m_prec:.2f}% ± {s_prec:.2f}")
    print(f"Mean Recall:    {m_rec:.2f}% ± {s_rec:.2f}")
    print(f"Mean F1 Score:  {m_f1:.2f}% ± {s_f1:.2f}")
    print(f"Mean Kappa:     {m_kap:.2f}% ± {s_kap:.2f}")
    print(f"Mean Fold Runtime: {m_t:.2f}s ± {s_t:.2f}s")


    n_classes = 3
    if len(cms) > 0:
        mean_cm = np.zeros_like(cms[0], dtype=float)
        for cm in cms:
            mean_cm += cm
        for i in range(mean_cm.shape[0]):
            row_sum = mean_cm[i].sum()
            if row_sum > 0:
                mean_cm[i] = mean_cm[i] / row_sum * 100

        plt.figure(figsize=(6, 5))
        sns.heatmap(mean_cm,
                    annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=[f"Task {i+1}" for i in range(n_classes)],
                    yticklabels=[f"Task {i+1}" for i in range(n_classes)])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"{band} Mean Confusion Matrix (Row-wise %)")
        plt.savefig(f"{band}_mean_confusion_matrix.png", dpi=120)
        plt.close()

    final_report = classification_report(targs, preds, zero_division=0)

    txt_filename = f"{band}_metrics.txt"
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write(f"{band} Band - K-Fold Results:\n\n")
        f.write(f"Mean Accuracy:  {m_acc:.2f}% ± {s_acc:.2f}\n")
        f.write(f"Mean Precision: {m_prec:.2f}% ± {s_prec:.2f}\n")
        f.write(f"Mean Recall:    {m_rec:.2f}% ± {s_rec:.2f}\n")
        f.write(f"Mean F1 Score:  {m_f1:.2f}% ± {s_f1:.2f}\n")
        f.write(f"Mean Kappa:     {m_kap:.2f}% ± {s_kap:.2f}\n\n")
        f.write(f"Mean Fold Runtime: {m_t:.2f} s ± {s_t:.2f} s\n\n")
        f.write("=== Last Fold Classification Report ===\n")
        f.write(final_report)
    print(f"Metrics saved to {txt_filename}")

    csv_filename = f"{band}_mean_confusion_matrix.csv"
    mean_cm_df = pd.DataFrame(mean_cm,
                              index=[f"Task {i+1}" for i in range(n_classes)],
                              columns=[f"Task {i+1}" for i in range(n_classes)])
    mean_cm_df.to_csv(csv_filename)
    print(f"Confusion matrix saved to {csv_filename}")

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def load_band_arrays(root: Path, band: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(root / f"{band}_source_data.npy")
    y = np.load(root / f"{band}_source_labels.npy")
    return x, y

# -----------------------------------------------------------------------------
# main entrypoint
# -----------------------------------------------------------------------------
if __name__=="__main__":
    cfg    = get_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=== Pretraining SimCLR on {cfg.pretrain_band} band ===")
    x_rep = np.load(cfg.data_root / f"{cfg.pretrain_band}_source_data.npy")
    simclr = pretrain_simclr(x_rep, cfg, DEVICE)

    for b in cfg.bands:
        dp = cfg.data_root / f"{b}_source_data.npy"
        lp = cfg.data_root / f"{b}_source_labels.npy"
        if not dp.exists() or not lp.exists():
            print(f"[WARN] {b} missing, skipping.")
            continue
        x, y = load_band_arrays(cfg.data_root, b)
        print(f"\n--- Band: {b} ---")
        run_classification(x, y, simclr, b, cfg, DEVICE)

    print("\n✓ All bands processed.")
